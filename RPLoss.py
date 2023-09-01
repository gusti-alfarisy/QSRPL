import torch
import torch.nn as nn
import torch.nn.functional as F

class Dist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random5'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random5':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)

        return dist

class RPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(RPLoss, self).__init__()
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        # print(f"Num centers: {options['num_centers']}")
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'] * 4 if options['qchannel'] else options['feat_dim'], num_centers=options['num_centers'])
        self.radius = 1

        self.radius = nn.Parameter(torch.Tensor(self.radius))
        self.radius.data.fill_(0)
        self.name = 'RPLoss'

    def forward(self, x, y, labels=None):
        dist = self.Dist(x)
        output = F.softmax(dist, dim=1)
        if labels is None: return output, 0
        # loss = F.cross_entropy(dist / self.temp, labels)
        loss = F.cross_entropy(dist * self.temp, labels)
        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, self.radius.repeat(_dis.size(0)))
        loss = loss + self.weight_pl * loss_r

        return output, loss
    

