import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from backbone import get_backbone

logger = logging.getLogger("model")
class DSOSR(nn.Module):
    # TODO adding this feature later
    def __init__(self, osr_model, autoencoder, threshold1, threshold2, criterion):
        super(DSOSR, self).__init__()
        self.osr_model = osr_model
        self.autoencoder = autoencoder
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.criterion = criterion

    # def forward(self, x):
    #     x = self.osr_model(x)
    #     prob = torch.nn.functional.softmax(x, dim=1)
    #     prob, idx_pred = torch.max(prob)
    #
    #     if prob >= self.threshold1:
    #         return prob, idx_pred
    #     elif prob <= self.threshold2:
    #         return None

    # def predict(self, x):
    #     x = self.osr_model(x)


class QSRPL(nn.Module):
    def __init__(self, n_class,
                 n_feature=1240,
                 pretrained_model="mobilenet_v3_large"):

        super(QSRPL, self).__init__()
        self.num_class = n_class
        self.num_features = n_feature
        self.backbone, backbone_fea = get_backbone(pretrained_model)
        logger.debug(f"feature backbone: {backbone_fea}")
        logger.debug(f"num_features: {self.num_features}")
        self.feature = nn.Linear(backbone_fea, self.num_features)

        self.feature2 = nn.Linear(backbone_fea, self.num_features)

        self.final_layer = nn.Linear(self.num_features*4, self.num_class)

        self.weight_K = torch.nn.Parameter(torch.randn(n_feature, n_feature))
        self.weight_Q = torch.nn.Parameter(torch.randn(n_feature, n_feature))
        self.weight_V = torch.nn.Parameter(torch.randn(n_feature, n_feature))

        self.weight_K.requires_grad = True
        self.weight_Q.requires_grad = True
        self.weight_V.requires_grad = True


        self.last_layer1 = nn.Linear(n_feature, n_feature)
        self.last_layer2 = nn.Linear(n_feature, n_feature)
        self.last_layer3 = nn.Linear(n_feature, n_feature)
        self.last_layer4 = nn.Linear(n_feature, n_feature)

        self.last_layer11 = nn.Linear(n_feature, n_feature)
        self.last_layer22 = nn.Linear(n_feature, n_feature)
        self.last_layer33 = nn.Linear(n_feature, n_feature)
        self.last_layer44 = nn.Linear(n_feature, n_feature)

    def forward(self, x, return_feature=False, im2=None, im3=None, im4=None):

        x_feature = self.forward_feature1(x)
        x_fea2 = self.forward_feature2(im2)
        x_fea3 = self.forward_feature2(im3)
        x_fea4 = self.forward_feature2(im4)


        x1_K = torch.matmul(x_feature, self.weight_K)
        x2_K = torch.matmul(x_fea2, self.weight_K)
        x3_K = torch.matmul(x_fea3, self.weight_K)
        x4_K = torch.matmul(x_fea4, self.weight_K)
        X_K = torch.stack((x1_K, x2_K, x3_K, x4_K), 1)
        X_KT = torch.transpose(X_K, 1, 2)

        x1_V = torch.matmul(x_feature, self.weight_V)
        x2_V = torch.matmul(x_fea2, self.weight_V)
        x3_V = torch.matmul(x_fea3, self.weight_V)
        x4_V = torch.matmul(x_fea4, self.weight_V)

        X_V = torch.stack((x1_V, x2_V, x3_V, x4_V), 1)

        x1_Q = torch.matmul(x_feature, self.weight_Q)
        x2_Q = torch.matmul(x_fea2, self.weight_Q)
        x3_Q = torch.matmul(x_fea3, self.weight_Q)
        x4_Q = torch.matmul(x_fea4, self.weight_Q)

        x1_Q = torch.unsqueeze(x1_Q, 1)
        x2_Q = torch.unsqueeze(x2_Q, 1)
        x3_Q = torch.unsqueeze(x3_Q, 1)
        x4_Q = torch.unsqueeze(x4_Q, 1)


        x1_attention = F.softmax(torch.matmul(x1_Q, X_KT), 2)
        x1_attention = torch.matmul(x1_attention, X_V)
        x1_attention = torch.sum(x1_attention, 1)

        x1_attention = F.relu(self.last_layer1(x1_attention)) + x_feature
        x_feature = self.last_layer11(x1_attention)

        x2_attention = F.softmax(torch.matmul(x2_Q, X_KT), 2)
        x2_attention = torch.matmul(x2_attention, X_V)
        x2_attention = torch.sum(x2_attention, 1)
        x2_attention = F.relu(self.last_layer2(x2_attention)) + x_fea2
        x_fea2 = self.last_layer22(x2_attention)

        x3_attention = F.softmax(torch.matmul(x3_Q, X_KT), 2)
        x3_attention = torch.matmul(x3_attention, X_V)
        x3_attention = torch.squeeze(x3_attention, 1)
        x3_attention = F.relu(self.last_layer3(x3_attention)) + x_fea3
        x_fea3 = self.last_layer33(x3_attention)

        x4_attention = F.softmax(torch.matmul(x4_Q, X_KT), 2)
        x4_attention = torch.matmul(x4_attention, X_V)
        x4_attention = torch.squeeze(x4_attention, 1)
        x4_attention = F.relu(self.last_layer4(x4_attention)) + x_fea4
        x_fea4 = self.last_layer44(x4_attention)

        x_feature = torch.cat((x_feature, x_fea2, x_fea3, x_fea4), 1)

        x = self.final_layer(x_feature)
        if return_feature:
            return x_feature, x
        else:
            return x

    def forward_feature1(self, x):
        x = self.backbone(x)
        x = F.relu(self.feature(x))
        return x

    def forward_feature2(self, x):
        x = self.backbone(x)
        x = F.relu(self.feature2(x))
        return x

    def backbone_feature(self, x):
        return self.backbone(x)


class AutoEncoderModel(nn.Module):
    def __init__(self,
                 n_feature=1024,
                 pretrained=True,
                 selfattention=False,
                 pretrained_model="mobilenet_v3_large"
                 ):

        super(AutoEncoderModel, self).__init__()
        self.selfattention = selfattention
        self.n_feature = n_feature
        self.backbone, self.backbone_features = get_backbone(pretrained_model, pretrained=pretrained)
        logger.debug(f"Backbone Features: {self.backbone_features}")
        logger.debug(f"N Features: {self.n_feature}")
        self.last_layer1 = nn.Linear(self.backbone_features, self.n_feature)
        self.last_layer2 = nn.Linear(self.n_feature, self.n_feature)

        # Decoder architecture
        self.decoder_fcl = nn.Linear(self.n_feature, 729)
        self.transform_channel = lambda x, bs: x.view((bs, 1, 27, 27))
        # self.transform_channel = lambda x, bs: x.view((bs, 1, 32, 32))
        self.decoder_convt = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 3, 2),
            nn.ConvTranspose2d(32, 32, 3, 2),
            nn.ConvTranspose2d(32, 3, 4, 2),
        )

    def encoder(self, x):
        x_backbone = self.backbone(x)
        x_backbone = F.relu(x_backbone)
        x = F.relu(self.last_layer1(x_backbone))
        x_enc = self.last_layer2(x)
        if self.selfattention:
            x_enc = self.SAModule(x_enc)

        return x_enc

    def forward(self, x):
        return self.encoder(x)

    def decoder(self, x):
        x = self.decoder_fcl(x)
        x = self.transform_channel(x, x.size(0))
        x = self.decoder_convt(x)
        return x
    def get_negative_MSE(self, x_img):
        x = self.encoder(x_img)
        recons = self.decoder(x)
        score = torch.tensor([]).cuda()
        for i, rec in enumerate(recons):
            recon_loss = F.mse_loss(x_img[i], rec, reduction='mean') * -1
            recon_loss = torch.unsqueeze(recon_loss, 0)
            score = torch.cat((score, recon_loss), 0)
        return score

    def get_MSE(self, x_img):
        x = self.encoder(x_img)
        recons = self.decoder(x)
        score = torch.tensor([]).cuda()
        for i, rec in enumerate(recons):
            recon_loss = F.mse_loss(x_img[i], rec, reduction='mean')
            recon_loss = torch.unsqueeze(recon_loss, 0)
            score = torch.cat((score, recon_loss), 0)
        return score