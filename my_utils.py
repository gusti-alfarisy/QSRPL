import torch
import os
import pandas as pd
from datetime import date
import torchvision.transforms.functional as Fvision


get_device = lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model(model, load_path):
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))


def make_dir(path):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def save_dict_csv(dict_file, path, verbose=True):
    log_df = pd.DataFrame(dict_file)
    today = today_str()
    saved_path = f"{path}_{today}.csv"
    log_df.to_csv(saved_path)
    if verbose:
        print("CSV file successfully saved at", saved_path)

    return saved_path

def today_str():
    return str(date.today()).replace('-', '_')

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def rotate_images(images, rotation_degree=90, device=None):
    im_1 = Fvision.rotate(images, rotation_degree)
    rotation_degree += rotation_degree
    im_2 = Fvision.rotate(images, rotation_degree)
    rotation_degree += rotation_degree
    im_3 = Fvision.rotate(images, rotation_degree)
    if device:
        im_1 = im_1.to(device)
        im_2 = im_2.to(device)
        im_3 = im_3.to(device)
    return im_1, im_2, im_3

def rotate_image(images, rotation_degree=90):
    im = Fvision.rotate(images, rotation_degree)
    return im

def hflip(images):
    return Fvision.hflip(images)

def vflip(images):
    return Fvision.vflip(images)

def save_model(model, current_path, model_name, epoch, train_loss, val_loss, verbose=True):
    if verbose:
        print("Saving the model...")

    today = str(date.today()).replace('-', '_')
    saved_path = f"{current_path}Model_{model_name}_Epoch_{epoch}_Loss_{train_loss:.3f}_valloss_{val_loss:.3f}_{today}.pth"
    torch.save(model.state_dict(), saved_path)

    if verbose:
        print("Successfully saved the model at", saved_path)

    return saved_path