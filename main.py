import logging

from dataloader_pool import cifar100_domain_osr_dl
from model import QSRPL, AutoEncoderModel
from train import train_qsrpl, train_autoencoder

if __name__ == '__main__':

    # logging.getLogger("model").setLevel(logging.DEBUG)

    domain = "vehicle"
    domain = "household_items"
    dc = cifar100_domain_osr_dl(domain=domain, download=True, batch_size=32)
    # model = QSRPL(dc.num_class, n_feature=256, pretrained_model="mobilenet_v3_large")
    # result = train_qsrpl(model, dc.num_class, dc.train_dl, 5, temperature=1.0, lamb=1.0, test_dl=dc.test_dl, openset_dl=dc.openset_dl, dataset_name=dc.name)
    # print(result)

    autoencoder = AutoEncoderModel(pretrained_model="mobilenet_v3_large")
    train_autoencoder(autoencoder, 3, dc.train_dl, test_dl=dc.test_dl)

