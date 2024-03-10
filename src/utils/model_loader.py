from torchvision import models
import pretrainedmodels
import clip
import torch
import os, copy
from torch import nn
from transformers import ViTImageProcessor, ViTForImageClassification
from src.torch_nets import (
    tf2torch_inception_v3,
    tf2torch_inception_v4,
    tf2torch_resnet_v2_50,
    tf2torch_resnet_v2_101,
    tf2torch_resnet_v2_152,
    tf2torch_inc_res_v2,
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
    )
from src.GN.inception_v3_GN import ghost_Inc_v3
from src.GN.inception_v4_GN import ghost_Inc_v4
from src.GN.densenet121_GN import ghost_DenseNet121
from src.GN.vgg16_GN import ghost_VGG16_BN
from src.GN.IncRes_v2_GN import ghost_inceptionresnet_v2
from src.GN.resnet_GN import ghost_ResNet50, ghost_ResNet101, ghost_ResNet152
from src.utils.scale_weight import all_scale
from src.utils.utils import get_idx2label


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class TfNormalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(TfNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class Permute(nn.Module):
    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]
    

def load_named_model(model_name: str, weight_path=None, lgv=False):
    if model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits=False
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    elif model_name == "inc_res_v2":
        model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    elif model_name == "inception_v4":
        model = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    elif model_name == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    else:
        raise KeyError("Unsupported Model!")
    if weight_path is not None:
        if lgv:
            weight = torch.load(weight_path, map_location='cpu')['state_dict']
        else:
            weight = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(weight)
    return model


def get_src_model(model_type, model_name, scaling_ratio=1, device='cuda'):
    func_get_model = None
    weight_dict = {'resnet50': 7, 'densenet121': 7, 'inception_v3': 9, 'vgg16_bn': 8}
    global src_model
    
    if model_type == 'pretrain':
        src_model = load_named_model(model_name).eval().to(device)
        func_get_model = lambda t: src_model
    elif model_type == 'SASD':    
        weight_path = f"./results/distilled_models/single_teacher_distill_{model_name}_min_val_loss_SAM,lr=0.05,t=1,{weight_dict[model_name]}-12.pth"
        src_model = load_named_model(model_name, weight_path).eval().to(device)
        func_get_model = lambda t: src_model
    elif model_type == 'GN':
        if model_name == "resnet50":
            src_model = ghost_ResNet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
        elif model_name == "inception_v3":
            src_model = ghost_Inc_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).eval().to(device)
            src_model.aux_logits = False
        elif model_name == "densenet121":
            src_model = ghost_DenseNet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).eval().to(device)
        elif model_name == "vgg16_bn":
            src_model = ghost_VGG16_BN(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).eval().to(device)
        def func_get_model(t):
            src_model.reset_GN()
            return src_model
    elif model_type == 'LGV':
        weight_paths = ["./results/lgv_models/%s/lgv_model_%05d.pt" % (model_name, i) for i in range(40)]
        src_model = [load_named_model(model_name, w, lgv=True) for w in weight_paths]
        def func_get_model(t):
            sur_idx = torch.randint(high=len(src_model), size=(1,)).item()
            return [src_model[sur_idx].eval().to(device)]
    else:
        raise NotImplementedError()
    
    if scaling_ratio != 1:
        assert (model_type == 'SASD' or model_type == 'pretrain')
        src_model = all_scale(src_model, scaling_ratio)
    
    return src_model, func_get_model


def get_src_models(model_type, scaling_ratio=1, device='cuda'):
    func_get_models = None
    global src_models
    src_models = []
    src_model_names = ['inception_v3', 'inception_v4', 'inc_res_v2', 'resnet50', 'resnet101', 'resnet152']
    
    if model_type == 'pretrain':
        for src_model_name in src_model_names:
            src_models.append(load_named_model(src_model_name).eval().to(device))
        func_get_models = lambda t: src_models
    elif model_type == 'SASD':
        src_ckpts = [9, 8, 8, 7, 8, 9]
        for src_model_name, src_ckpt in zip(src_model_names, src_ckpts):
            weight_path = f"./results/distilled_models/single_teacher_distill_{src_model_name}_min_val_loss_SAM,lr=0.05,t=1,{src_ckpt}-12.pth"
            model = load_named_model(src_model_name, weight_path)
            src_models.append(model.eval().to(device))
        func_get_models = lambda t: src_models
    elif model_type == 'GN':
        src_models.append(ghost_Inc_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).eval().to(device))
        src_models[-1].aux_logits = False
        src_models.append(ghost_Inc_v4(num_classes=1000, pretrained='imagenet').eval().to(device))
        src_models.append(ghost_ResNet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().to(device))
        src_models.append(ghost_ResNet101(weights=models.ResNet101_Weights.IMAGENET1K_V1).eval().to(device))
        src_models.append(ghost_ResNet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).eval().to(device))
        src_models.append(ghost_inceptionresnet_v2(num_classes=1000, pretrained='imagenet').eval().to(device))
        def func_get_models(t):
            for model in src_models:
                model.reset_GN()
            return src_models
    elif model_type == 'LGV':
        for src_model_name in src_model_names:
            weight_paths = ["./results/lgv_models/%s/lgv_model_%05d.pt" % (src_model_name, i) for i in range(40)]
            single_type_src_models = [load_named_model(src_model_name, w, lgv=True) for w in weight_paths]
            src_models.append(single_type_src_models)
        def func_get_models(t):
            selected_models = []
            for i in range(len(src_models)):
                sur_idx = torch.randint(high=len(src_models[0]), size=(1,)).item()
                selected_models.append(src_models[i][sur_idx])
            return [copy.deepcopy(m).eval().to(device) for m in selected_models]
    else:
        raise NotImplementedError()
    
    if scaling_ratio != 1:
        assert (model_type == 'SASD' or model_type == 'pretrain')
        for model in src_models:
            model = all_scale(model, scaling_ratio)
    
    return src_models, func_get_models


def get_adv_CNNs_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf2torch_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf2torch_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf2torch_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf2torch_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf2torch_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf2torch_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
    else:
        raise KeyError("Unsupported adv CNNs type")

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval(),)
    return model


def get_ViTs_models(model_name):
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, resume_download=True)
    return processor, model


def get_CLIP_models(model_name, device):
    model, _ = clip.load(model_name, device=device,  jit=False)
    return model


def get_target_models(device='cuda'):
    processors = []
    adv_CNNs = []
    ViT_models = []
    CLIP_models = []
    
    adv_CNN_model_names = [
        'tf2torch_resnet_v2_152',
        'tf2torch_ens3_adv_inc_v3',
        'tf2torch_ens4_adv_inc_v3',
        'tf2torch_ens_adv_inc_res_v2',
        'tf2torch_adv_inception_v3',
        ]
    adv_CNNs = [get_adv_CNNs_model(model_name, "./tf2torch_models/") for model_name in adv_CNN_model_names]
    
    ViT_model_names = ['google/vit-base-patch16-224', 'google/vit-large-patch16-224']
    for model_name in ViT_model_names:
        processor, target_model = get_ViTs_models(model_name)
        processors.append(processor)
        ViT_models.append(target_model)
    
    CLIP_model_names = ['RN50', 'ViT-B/32', 'ViT-L/14']
    CLIP_models = [get_CLIP_models(model_name, device) for model_name in CLIP_model_names]
    idx2label = get_idx2label()
    text = clip.tokenize([f"{idx2label[t]}" for t in range(1000)]).to(device)

    for target_model in adv_CNNs:
        for param in target_model.parameters():
            param.requires_grad = False
        target_model = target_model.eval().to(device)
    
    for target_model in ViT_models:
        for param in target_model.parameters():
            param.requires_grad = False
        target_model = target_model.eval().to(device)
    
    for target_model in CLIP_models:
        for param in target_model.parameters():
            param.requires_grad = False
        target_model = target_model.eval().to(device)

    return adv_CNNs, ViT_models, CLIP_models, processors, text