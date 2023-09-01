import torch
import numpy as np
from torch.distributed.pipeline.sync.stream import get_device

from my_utils import rotate_images, load_model
import pandas as pd
from sklearn.metrics import roc_curve, auc


def test_AutoEncoder(model, val_dl, load_path=None, device=None):

    device = get_device() if device is None else device

    if load_path:
        load_model(model, load_path)
        print("loading model successfully")

    mse_loss = torch.nn.MSELoss()
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(device)
            latent = model(images)
            reconstructed = model.decoder(latent)
            loss = mse_loss(images, reconstructed)
            loss_total += loss.item()

    return loss_total / len(val_dl)


def compute_baccu(model, device, criterion, known_dl, unknown_dl, threshold, options=None):
    correct_known = 0
    total_known = 0
    correct_unknown = 0
    total_unknown = 0

    with torch.no_grad():
        for (images, labels) in known_dl:

            images = images.to(device)
            labels = labels.to(device)

            if options['qchannel']:
                im90, im180, im270 = rotate_images(images)
                feature, output = model(images, True, im2=im90, im3=im180, im4=im270)
            else:
                feature, output = model(images, True)
            logits, _ = criterion(feature, output)

            final_dis = logits.data.max(1)[0]
            final_dis = final_dis.where(final_dis >= threshold, torch.tensor(0.0).to(device))
            final_dis = final_dis.where(final_dis < threshold, torch.tensor(1.0).to(device))

            total_known += labels.size(0)
            correct_known += (final_dis == 1).sum().item()

        for (images, labels) in unknown_dl:

            images = images.to(device)
            labels = labels.to(device)

            if options['qchannel']:
                im90, im180, im270 = rotate_images(images)
                feature, output = model(images, True, im2=im90, im3=im180, im4=im270)
            else:
                feature, output = model(images, True)

            logits, _ = criterion(feature, output)

            final_dis = logits.data.max(1)[0]
            final_dis = final_dis.where(final_dis >= threshold, torch.tensor(0.0).to(device))
            final_dis = final_dis.where(final_dis < threshold, torch.tensor(1.0).to(device))

            total_unknown += labels.size(0)
            correct_unknown += (final_dis == 0).sum().item()

        return (correct_known / total_known + correct_unknown / total_unknown) / 2

def AUROC_Sklearn(model, device, criterion, loader_known, loader_unknown,
                  path=None, return_threshold=False, options=None):
    if path is not None:
        model.load_state_dict(torch.load(path))

    correct, total = 0, 0
    y_pred = np.array([], dtype="int32")
    y_label = np.array([], dtype="int32")
    with torch.no_grad():
        for (images, labels) in loader_known:
            images = images.to(device)
            labels = labels.to(device)

            if options['qchannel']:
                im90, im180, im270 = rotate_images(images)
                feature, output =  model(images, True, im2=im90, im3=im180, im4=im270)
            else:
                feature, output = model(images, True)

            logits, _ = criterion(feature, output)
            predictions = logits.data.max(1)[1]
            correct += (predictions == labels.data).sum()
            final_dis = logits.data.max(1)[0]
            y_pred = np.concatenate((y_pred, final_dis.cpu().numpy()))
            true_label = np.ones(predictions.size(0))
            y_label = np.concatenate((y_label, true_label))
            total += labels.size(0)

        total = 0
        for (images, labels) in loader_unknown:
            images = images.to(device)
            labels = labels.to(device)

            if options['qchannel']:
                im90, im180, im270 = rotate_images(images)
                feature, output = model(images, True, im2=im90, im3=im180, im4=im270)
            else:
                feature, output = model(images, True)

            logits, _ = criterion(feature, output)
            predictions = logits.data.max(1)[1]
            correct += (predictions == labels.data).sum()

            final_dis = logits.data.max(1)[0]
            y_pred = np.concatenate((y_pred, final_dis.cpu().numpy()))

            true_label = np.zeros(predictions.size(0))
            y_label = np.concatenate((y_label, true_label))

            total += labels.size(0)

        fpr, tpr, threshold = roc_curve(y_true=y_label, y_score=y_pred, pos_label=1)
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
        roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
        optimal_threshold = roc_df_maxj.iloc[0]['threshold']
        auroc = auc(fpr, tpr)


    if return_threshold:
        return auroc, optimal_threshold

    return auroc

def test(net, criterion, testloader, outloader=None, device=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            if options['qchannel']:
                im90, im180, im270 = rotate_images(data)

            with torch.set_grad_enabled(False):
                if options['qchannel']:
                    x, y = net(data, True, im2=im90, im3=im180, im4=im270)
                else:
                    x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

    # Accuracy
    acc = float(correct) * 100. / float(total)

    # _pred_k = np.concatenate(_pred_k, 0)
    # _pred_u = np.concatenate(_pred_u, 0)
    # _labels = np.concatenate(_labels, 0)

    # Out-of-Distribution detction evaluation
    # x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    # results = evaluation.metric_ood(x1, x2, verbose=verbose)['Bas']
    results = {}
    # AUROC from sklearn
    auroc_sklearn, thre_sklearn = AUROC_Sklearn(net, device, criterion, testloader, outloader, return_threshold=True, options=options)
    baccu = compute_baccu(net, device, criterion, testloader, outloader, thre_sklearn, options)

    results['ACC'] = acc
    results['AUROC_Sklearn'] = auroc_sklearn * 100
    results['Threshold'] = thre_sklearn
    results['BACCU'] = baccu * 100

    return results