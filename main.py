import logging

from dataloader_pool import cifar100_domain_osr_dl
from model import QSRPL
from train import train

if __name__ == '__main__':
    domain = "vehicle"
    domain = "household_items"
    dc = cifar100_domain_osr_dl(domain=domain, download=True)
    model = QSRPL(dc.num_class, n_feature=256, pretrained_model="mobilenet_v3_large")
    result = train(model, dc.num_class, dc.train_dl, 5, temperature=1.0, lamb=1.0, test_dl=dc.test_dl, openset_dl=dc.openset_dl, dataset_name=dc.name)
    print(result)
