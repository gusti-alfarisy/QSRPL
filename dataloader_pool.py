from collections import namedtuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import logging
import platform

logger = logging.getLogger("qsrpl")

logging.debug(f'Loading dataset at platform {platform.node()}')
OSRDataContainer = namedtuple("OSRDataContainer", "name num_class train_dl val_dl test_dl openset_dl test_ds")

class CIFAR100_Domain(CIFAR100):
    def __init__(self, root: str,
                 train: bool = True,
                 transform=None,
                 target_transform=None,
                 download: bool = False, domain='vehicle', index_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

        super(CIFAR100_Domain, self).__init__(root, train, transform, target_transform, download)

        if domain == 'vehicle':
            self.class_name = ['bicycle', 'pickup_truck', 'train', 'bus', 'motorcycle', 'streetcar', 'rocket',
                                 'lawn_mower',
                                 'tractor', 'tank']
        elif domain == 'household_items':
            self.class_name = ['lamp', 'keyboard', 'clock', 'television', 'telephone', 'chair', 'couch', 'wardrobe',
                                 'table',
                                 'bed']
        else:
            raise Exception('domain is not available')

        self.class_domain = [self.class_name[e] for e in index_class]

        the_targets = np.array(self.targets)
        df_targets = pd.DataFrame({"idx_label": the_targets, "idx": list(range(len(the_targets)))},
                                  dtype=np.int32)
        idx_cls = []
        for cls in self.class_domain:
            idx_cls.append(self.classes.index(cls))
        idx_cls_srs = pd.Series(idx_cls)
        self.map_new_idx_cls = {e: i for i, e in enumerate(idx_cls)}
        df_targets = df_targets[df_targets['idx_label'].isin(idx_cls_srs)]
        self.domain_idx = df_targets['idx'].tolist()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = self.domain_idx[index]
        img, target = self.data[index], self.targets[index]
        target = self.map_new_idx_cls[target]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.domain_idx)


def cifar100_domain_osr_dl(DATASET_PATH="./dataset/", batch_size=32, class_known=[0, 1, 2, 3, 4, 5], class_unknown=[6, 7, 8, 9], domain='vehicle', shuffle=True, num_workers=2, img_size=224, download=True):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = CIFAR100_Domain(DATASET_PATH, train=True, domain=domain, index_class=class_known, transform=transform, download=download)
    test_ds = CIFAR100_Domain(DATASET_PATH, train=False, domain=domain, index_class=class_known, transform=transform, download=download)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if class_unknown is not None:
        ood_test_ds = CIFAR100_Domain(DATASET_PATH, train=False, domain=domain, index_class=class_unknown, transform=transform, download=download)
        ood_dl = DataLoader(ood_test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        ood_dl = None

    return OSRDataContainer(name=f"cifar100_{domain}",
                            num_class=len(class_known),
                            train_dl=train_dl,
                            val_dl=None,
                            test_dl=test_dl,
                            openset_dl=ood_dl,
                            test_ds=test_ds
                            )