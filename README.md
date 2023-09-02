

# Open Domain-Specific Recognition (ODSR) using QSRPL and AutoEncoder
Implementation of Quad-channel Self-attention Reciprocal Point Learning (QSRPL) and AutoEncoder for ODSR.

To use QSRPL model, we just import from the models.py as shown below. You can use this model to train with your own problems.

```python
from model import QSRPL

model = QSRPL(6) # 6 is the number of classes
# In complete arguments
model = QSRPL(n_class=6, n_feature=256, pretrained_model="mobilenet_v3_large")
```

Pretrained models that is supported in this code is listed below:

| Pretrained models  |
|--------------------|
| mobilenet_v3_large |
| mobilenet_v3_small |
| efficientnet_b0    |
| efficientnet_b1    |
| efficientnet_b2    |
| densenet121        |    

We also provided training function requiring data loader for train set, test set, and open set as shown below:

```python
from train import train_qsrpl
result = train_qsrpl(model, num_class, train_dl, 5, temperature=1.0, lamb=1.0, test_dl=test_dl, openset_dl=openset_dl, dataset_name="cifar100_vehicle")
```

We provided the dataset for Cifar100 in domains of vehicle and household items that can be used through:

```python
from dataloader_pool import cifar100_domain_osr_dl
# dc is dataset container in form of named tuple that has properties: name, num_class, train_dl, test_dl, openset_dl | dl is data loadder
dc_vehicle = cifar100_domain_osr_dl(domain="vehicle", download=True, batch_size=32, class_known=[0, 1, 2, 3, 4, 5], class_unknown=[6, 7, 8, 9])
dc_houshold = cifar100_domain_osr_dl(domain="household_items", download=True, batch_size=32, class_known=[0, 1, 2, 3, 4, 5], class_unknown=[6, 7, 8, 9])
```

To train the model using our data container, you can utilize the previous function which also shown below:


```python
from model import QSRPL
from train import train_qsrpl
from dataloader_pool import cifar100_domain_osr_dl
domain = "vehicle"
dc = cifar100_domain_osr_dl(domain=domain, download=True, batch_size=32)
model = QSRPL(dc.num_class, n_feature=256, pretrained_model="mobilenet_v3_large")
# 50 is the number of epoch
result = train_qsrpl(model, dc.num_class, dc.train_dl, 50, temperature=1.0, lamb=1.0, test_dl=dc.test_dl, openset_dl=dc.openset_dl, dataset_name=dc.name)
print(result)
```

To train the autoencoder, you can use the autoecnoder module that we have provided as shown below:

```python
from model import AutoEncoderModel
from train import train_autoencoder
from dataloader_pool import cifar100_domain_osr_dl
domain = "vehicle"
dc = cifar100_domain_osr_dl(domain=domain, download=True, batch_size=32)
autoencoder = AutoEncoderModel(pretrained_model="mobilenet_v3_large")
# 50 is the number of epoch
train_autoencoder(autoencoder, 50, dc.train_dl, test_dl=dc.test_dl)
```

Please feel free to push an issue or your own source code. Thank you! :)
