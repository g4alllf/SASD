import os
import numpy as np
import torch
from torchvision import transforms
from torch import Tensor
from PIL import Image


def restore_from_normalized(image):
    """
    Input tensor shape (3,299,299)
    Return numpy shape (299,299,3)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return (image.permute(1, 2, 0).cpu() * torch.tensor(imagenet_std) + torch.tensor(imagenet_mean)).detach().numpy()


def transform_to_PIL(image):
    """
    Transform the tensor into PIL image
    """
    T = transforms.ToPILImage()
    image = T(image)
    return image


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def calculate_weight(kernel_width, distances):
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))


def input_transform(img) -> Tensor:
    """
    ### Args:
        img: PIL image
    ### returns:
        return tensor
    """
    return __get_input_transform()(img)


def __get_input_transform():
    """
    In fact, return get_pil_transform(get_preprocess_transform())
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transf = transforms.Compose([
        transforms.Resize((342, 342)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])

    return transf


def pil_transform(img):
    return __get_pil_transform()(img)


def __get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((342, 342)),
        transforms.CenterCrop(299)
    ])

    return transf


def preprocess_transform(img):
    return __get_preprocess_transform()(img)


def __get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


def get_input_tensors(img):
    # unsqeeze converts single image to batch of 1
    return preprocess_transform(pil_transform(img)).unsqueeze(0)