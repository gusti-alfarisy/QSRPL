import torchvision
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights, \
    EfficientNet_B1_Weights, EfficientNet_B2_Weights, DenseNet121_Weights


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def adjust_final_layers(layers):
    all_layers = []
    for ll in layers:
        all_layers.append(ll)
    all_layers.pop()
    return all_layers

def mobilenet_v3_large(pretrained, freeze_backbone, only_remove_last=True):
    backbone = torchvision.models.mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)

    if freeze_backbone:
        freeze_parameters(backbone)

    n_feature = backbone.classifier[0].in_features

    if only_remove_last:
        clf_layers = adjust_final_layers(backbone.classifier)
        backbone.classifier = nn.Sequential(*clf_layers)
        n_feature = backbone.classifier[0].out_features
    else:
        backbone.classifier = nn.Sequential()

    return backbone, n_feature


def mobilenet_v3_small(pretrained, freeze_backbone, only_remove_last=True):
    backbone = torchvision.models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_backbone:
        freeze_parameters(backbone)

    n_feature = backbone.classifier[0].in_features
    if only_remove_last:
        clf_layers = adjust_final_layers(backbone.classifier)
        backbone.classifier = nn.Sequential(*clf_layers)
        n_feature = backbone.classifier[0].out_features
    else:
        backbone.classifier = nn.Sequential()

    return backbone, n_feature


def efficientnet_b0(pretrained, freeze_backbone, only_remove_last=True):
    backbone = torchvision.models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_backbone:
        freeze_parameters(backbone)

    n_feature = backbone.classifier[-1].in_features
    if only_remove_last:
        clf_layers = adjust_final_layers(backbone.classifier)
        backbone.classifier = nn.Sequential(*clf_layers)
    else:
        backbone.classifier = nn.Sequential()

    return backbone, n_feature

def efficientnet_b1(pretrained, freeze_backbone, only_remove_last=True):
    backbone = torchvision.models.efficientnet_b1(
        weights=EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_backbone:
        freeze_parameters(backbone)

    n_feature = backbone.classifier[-1].in_features
    if only_remove_last:
        clf_layers = adjust_final_layers(backbone.classifier)
        backbone.classifier = nn.Sequential(*clf_layers)
    else:
        backbone.classifier = nn.Sequential()

    return backbone, n_feature

def efficientnet_b2(pretrained, freeze_backbone, only_remove_last=True):
    backbone = torchvision.models.efficientnet_b2(
        weights=EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_backbone:
        freeze_parameters(backbone)

    n_feature = backbone.classifier[-1].in_features
    if only_remove_last:
        clf_layers = adjust_final_layers(backbone.classifier)
        backbone.classifier = nn.Sequential(*clf_layers)
    else:
        backbone.classifier = nn.Sequential()

    return backbone, n_feature

def densenet121(pretrained, freeze_backbone, only_remove_last=True):
    backbone = torchvision.models.densenet121(
        weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_backbone:
        freeze_parameters(backbone)
    n_feature = backbone.classifier.in_features
    backbone.classifier = nn.Sequential()

    return backbone, n_feature


def get_backbone(name, pretrained=True, freeze_backbone=True, only_remove_last=True):
    try:
        _func = globals()[name]
        backbone, n_feature = _func(pretrained, freeze_backbone, only_remove_last)
    except Exception:
        raise Exception("backbone not implemented yet")

    return backbone, n_feature


if __name__ == "__main__":
    backbone, n_feature = get_backbone("efficientnet_b2")
    print(backbone)
    print("NFEAtURE", n_feature)
