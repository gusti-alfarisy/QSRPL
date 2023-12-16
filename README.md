

# Open Domain-Specific Recognition (ODSR) using QSRPL and AutoEncoder
Implementation of Quad-channel Self-attention Reciprocal Point Learning (QSRPL) and AutoEncoder for ODSR.

[Journal link](https://www.sciencedirect.com/science/article/pii/S0950705123010109)

> Open-Set recognition (OSR) emphasizes its ability to reject unknown classes and maintain closed-set performance simultaneously. The primary objective of OSR is to minimize the risk of unknown classes being predicted as one of the known classes. OSR operates under the assumption that unknown classes will be present during testing, and it identifies a single distribution for these unknowns. Recognizing unknowns both within and outside the domain of interest can enhance future learning efforts. The rejected unknown samples within the domain of interest can be used to refine deep learning models further. We introduced a new challenge within OSR called Open Domain-Specific Recognition (ODSR). This approach formalizes the risk in the open domain-specific space to address the recognition of two distinct unknown distributions, i.e., in-domain (ID) and out-of-domain (OOD) unknowns. To address this, we proposed an initial baseline that employs Quad-channel Self-attention Reciprocal Point Learning (QSRPL) to mitigate open-space risk. Additionally, we used an Autoencoder to handle open domain-specific space risk. We harnessed the knowledge embedded in pre-trained models and optimized the open-set hyperparameters before benchmarking against other methods. We also explored how different pre-trained models influence open-set recognition performance. To validate our method, we tested our model across various domains, including garbage, vehicles, household items, and pets. Experimental results indicate that our approach is effective at rejecting unseen classes while maintaining closed-set accuracy. Furthermore, the Autoencoder shows potential in addressing open domain-specific space risks for future development. The choice of pre-trained models significantly affects the performance of open-set recognition in rejecting unknowns. The source code of the project is available at https://github.com/gusti-alfarisy/QSRPL.


To use QSRPL model, we need to import a module from the models.py as shown below. You can use this model to train with your problems.


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

We also provided a training function requiring a data loader for a train set, test set, and open set as shown below:

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

To train the autoencoder, you can use the autoencoder module that we have provided as shown below:

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

To cite our paper:

```
@article{ALFARISY2024111261,
title = {Towards open domain-specific recognition using Quad-Channel Self-Attention Reciprocal Point Learning and Autoencoder},
journal = {Knowledge-Based Systems},
volume = {284},
pages = {111261},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.111261},
url = {https://www.sciencedirect.com/science/article/pii/S0950705123010109},
author = {Gusti Ahmad Fanshuri Alfarisy and Owais Ahmed Malik and Ong Wee Hong},
keywords = {Deep learning, Open-set recognition, Open domain-specific recognition, Out-of-distribution detection, Prototype-based neural networks, Prototype learning, Reciprocal point learning, Autoencoder},
abstract = {Open-Set recognition (OSR) emphasizes its ability to reject unknown classes and maintain closed-set performance simultaneously. The primary objective of OSR is to minimize the risk of unknown classes being predicted as one of the known classes. OSR operates under the assumption that unknown classes will be present during testing, and it identifies a single distribution for these unknowns. Recognizing unknowns both within and outside the domain of interest can enhance future learning efforts. The rejected unknown samples within the domain of interest can be used to refine deep learning models further. We introduced a new challenge within OSR called Open Domain-Specific Recognition (ODSR). This approach formalizes the risk in the open domain-specific space to address the recognition of two distinct unknown distributions, i.e., in-domain (ID) and out-of-domain (OOD) unknowns. To address this, we proposed an initial baseline that employs Quad-channel Self-attention Reciprocal Point Learning (QSRPL) to mitigate open-space risk. Additionally, we used an Autoencoder to handle open domain-specific space risk. We harnessed the knowledge embedded in pre-trained models and optimized the open-set hyperparameters before benchmarking against other methods. We also explored how different pre-trained models influence open-set recognition performance. To validate our method, we tested our model across various domains, including garbage, vehicles, household items, and pets. Experimental results indicate that our approach is effective at rejecting unseen classes while maintaining closed-set accuracy. Furthermore, the Autoencoder shows potential in addressing open domain-specific space risks for future development. The choice of pre-trained models significantly affects the performance of open-set recognition in rejecting unknowns. The source code of the project is available at https://github.com/gusti-alfarisy/QSRPL.}
}
``` 


Please feel free to push an issue or your own source code. Thank you! :)
