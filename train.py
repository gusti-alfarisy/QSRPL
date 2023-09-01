import logging
import torch
import os

from RPLoss import RPLoss
from my_utils import load_model, today_str, make_dir, rotate_images, save_dict_csv, get_device, save_model
from time import time
import os.path as osp
from test import test, test_AutoEncoder
import errno
import pandas as pd



logger = logging.getLogger("train")


def train_autoencoder(model, num_epochs, train_dl,
                            test_dl=None,
                            lr=0.001,
                            path='_checkpoints/autoencoder/',
                            load_path=None,
                            start_epoch=0,
                            path_log="autoencoder",
                            save_each_epoch=False,
                      device=None
                            ):

    device = get_device() if device is None else device

    load_model(model, load_path)
    make_dir(path)

    path_log = os.path.join("_logs", path_log)
    make_dir(path_log)

    log_dictionary = {
        'Epoch': [],
        'Train_MSE': [],
    }

    if test_dl:
        log_dictionary['Test_MSE'] = []

    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    total_step = len(train_dl)

    save = lambda epoch, train_loss, val_loss: save_model(model, path, "AutoEncoder", epoch, train_loss, val_loss,
                                                          verbose=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        loss_step = 0
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            optimizer.zero_grad()

            latent = model(images)
            reconstructed = model.decoder(latent)

            loss = mse_loss(images, reconstructed)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_step += 1

            # if (i + 1) % 100 == 0 or (i + 1) == total_step:
            #     print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], MSE Loss: {loss.item():.4f}")

        # print(f"Epoch [{epoch + 1}/{num_epochs}]: MSE Loss: {total_loss / loss_step}")

        train_loss = total_loss / loss_step
        test_loss = test_AutoEncoder(model, test_dl, device=device)
        model.train()
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Test MSE Loss: {test_loss}')

        if save_each_epoch:
            path = save(epoch + 1, train_loss, test_loss)

        print(f"--- PERFORMANCE EPOCH: {epoch + 1} ---")
        print(f"Train MSE Loss: {train_loss}")
        print(f"Test MSE Loss: {test_loss}")

        # Store the log to the dictionary
        log_dictionary['Epoch'].append(epoch + 1)
        log_dictionary['Train_MSE'].append(train_loss)
        if test_dl:
            log_dictionary['Test_MSE'].append(test_loss)

    if not save_each_epoch:
        path = save(epoch + 1, train_loss, test_loss)

    save_dict_csv(log_dictionary, path_log)
    log_dictionary['saved_path'] = path
    return log_dictionary





class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_qsrpl(model, num_class, train_dl, epoch,
                dataset_name="dsname",
                temperature=1.0, lamb=1.0,
                test_dl=None,
                openset_dl=None,
                load_path=None,
                postfix_file="",
                path=None,
                device=None):

    device = get_device() if device is None else device
    model = model.to(device)

    path = f"_checkpoints/qsrpl/{dataset_name}" if path is None else path

    if load_path is not None:
        load_model(model, load_path[0])
        res = train_RPL(model, num_class, train_dl, epoch, test_dl=test_dl, openset_dl=openset_dl, nfeature=model.num_features, T=temperature, lamb=lamb, device=device,
                        postfix_file=postfix_file, path=path, load_path_criterion=load_path[1], ood_dc=openset_dl)
    else:
        res = train_RPL(model, num_class, train_dl, epoch, test_dl=test_dl, openset_dl=openset_dl, nfeature=model.num_features, T=temperature, lamb=lamb, device=device,
                        postfix_file=postfix_file, path=path, ood_dc=openset_dl)


    if load_path is None:
        logger.debug(f"res: {res}")
        print(f"SAVED PATH: {res['saved_path']}")

    logger.debug(f"res: {res}")
    print(f"accuracy: {res['Test Accuracy'][-1]}")
    print(f"auroc: {res['AUROC'][-1]}")
    print(f"baccu: {res['BACCU'][-1]}")
    print(f"thresh: {res['Threshold'][-1]}")


    logger.info(f"Done Training")

    return {
        'accuracy': res['Test Accuracy'][-1],
        'auroc': res['AUROC'][-1],
        'baccu': res['BACCU'][-1],
        'threshold': res['Threshold'][-1],
        'saved path': res['saved_path'] if load_path is None else None,
        'criterion': res['criterion']
    }


def train_per_epoch(net, criterion, optimizer, trainloader,  **options):
    net.train_qsrpl()
    losses = AverageMeter()

    torch.cuda.empty_cache()

    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        if options['qchannel']:
            im90, im180, im270 = rotate_images(data)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            if options['qchannel']:
                x, y = net(data, True, im2=im90, im3=im180, im4=im270)
            else:
                x, y = net(data, True)

            logits, loss = criterion(x, y, labels)

            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))
        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

        loss_all += losses.avg

    return loss_all
def train_RPL(model, num_class, train_dl, num_epochs,
              dataset_name=None,
              test_dl=None,
              openset_dl=None,
              nfeature=1024,
              options=None,
              path=None,
              postfix_file="",
              device=None,
              T=1.0, lamb=1.0,
              load_path_criterion=None,
              ood_dc=None):

    if options is None:
        options = {
            'num_classes': num_class,
            'feat_dim': nfeature,
            'num_centers': 1,
            # lambdaa
            'weight_pl': lamb,
            'outf': './logs',
            'lr': 0.001,
            'max_epoch': num_epochs,
            'model': 'MnetV3',
            'loss': 'RPLoss',
            'temp': T,
            'dataset': dataset_name,
            'use_gpu': True,
            'print_freq': 100,
            'eval_freq': 100,
            'qchannel': True,
        }

    criterion = RPLoss(**options)
    criterion = criterion.to(device)

    if load_path_criterion is not None:
        criterion.load_state_dict(torch.load(load_path_criterion))
    model_path = path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params_list = [{'params': model.parameters()},
                   {'params': criterion.parameters()}]

    optimizer = torch.optim.Adam(params_list, lr=options['lr'])

    file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], postfix_file, today_str())
    res_map = {
        "Epoch": [],
        "Test Accuracy": [],
        "AUROC": [],
        "Threshold": [],
        "BACCU": [],
    }


    if load_path_criterion is None:
        logger.info("Starts Training...")
        logger.info(f"Total Epoch: {options['max_epoch']}")
        logger.info(f"openset_dl: {openset_dl}")
        for epoch in range(options['max_epoch']):
            train_per_epoch(model, criterion, optimizer, train_dl, epoch=num_epochs, **options)

            results = test(model, criterion, test_dl, openset_dl, epoch=epoch,
                           verbose=True, device=device, **options)
            logger.debug(results)

            print("Epoch {} | Acc (%): {:.3f}\t AUROC (%): {:.3f}\t BACCU (%): {:.3f}\t".format(epoch+1,
                                                                                                   results['ACC'],
                                                                                                   results[
                                                                                                       'AUROC_Sklearn'],
                                                                                                   results[
                                                                                                       'BACCU']))



            res_map['Epoch'].append(epoch + 1)
            res_map['Test Accuracy'].append(results['ACC'])
            res_map['AUROC'].append(results['AUROC_Sklearn'])
            res_map['Threshold'].append(results['Threshold'])
            res_map['BACCU'].append(results['BACCU'])

            saved_path, saved_path_criterion = save_networks(model, model_path, file_name, criterion=criterion)

    else:
        logger.debug("no training.. move to test")
        results = test(model, criterion, test_dl, openset_dl, epoch=num_epochs, verbose=True, device=device, **options)
        print(
            "Acc (%): {:.3f}\t AUROC (%): {:.3f}\t BACCU (%): {:.3f}\t".format(results['ACC'],
                                                                                           results['AUROC_Sklearn'],
                                                                                           results['BACCU']))
        logger.debug(f"AUROC: {results['AUROC']}")

        res_map['Epoch'].append(num_epochs)
        res_map['Test Accuracy'].append(results['ACC'])
        res_map['AUROC'].append(results['AUROC_Sklearn'])
        res_map['Threshold'].append(results['Threshold'])
        res_map['BACCU'].append(results['BACCU'])

    logger.debug(res_map)

    df = pd.DataFrame(res_map)
    # csv_path = f"logs/train/{dataset_name}/QSRPL_{postfix_file}_{today_str()}{'' if ood_dc is None else '_' + ood_dc.name}"
    csv_path = f"_logs/train/{dataset_name}/QSRPL_{postfix_file}_{today_str()}"
    make_dir(csv_path)
    df.to_csv(csv_path + ".csv")
    res_map['model'] = model
    res_map['saved_path'] = (saved_path, saved_path_criterion) if load_path_criterion is None else None
    res_map['criterion'] = criterion

    return res_map

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_networks(networks, result_dir, name='', loss='', criterion=None):
    mkdir_if_missing(osp.join(result_dir))
    weights = networks.state_dict()
    filename = '{}/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename_criterion = '{}/{}_{}_criterion_{}.pth'.format(result_dir, name, loss, criterion.name)
        torch.save(weights, filename_criterion)
        return filename, filename_criterion

    return filename