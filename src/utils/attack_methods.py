import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from torch import nn
from torchvision import transforms


##define TI
def TI(device):
    def gkern(kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    kernel_size=5
    kernel = gkern(kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)
    return gaussian_kernel


##define DI
def DI(X_in):
    rnd = np.random.randint(299, 330,size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out 
    else:
        return  X_in


def compute_cost(args, outputs, labels, loss, targeted):
    if args.adv_loss_function == 'CE':
        if targeted:
            return loss(outputs, labels)
        else:
            return -1 * loss(outputs, labels)
    elif args.adv_loss_function == 'MaxLogit':
        real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        if targeted:
            return (-1 * real).sum()
        else:
            return real.sum()


def pgd(args, models, get_raw_logits, data, labels, targeted, epsilon, k, a, random_start=True, device='cuda'):
    data_max = torch.clamp(data + epsilon, 0, 1)
    data_min = torch.clamp(data - epsilon, 0, 1)

    perturbed_data = data.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        perturbed_data = perturbed_data + torch.empty_like(perturbed_data).uniform_(-epsilon, epsilon)
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1)

    loss = nn.CrossEntropyLoss(reduction='sum')

    for _ in range(k):
        perturbed_data.requires_grad = True
        outputs = get_raw_logits(perturbed_data, models)
        cost = compute_cost(args, outputs, labels, loss, targeted)

        # Update adversarial images
        cost.backward()

        gradient = perturbed_data.grad.clone().to(device)
        perturbed_data.grad.zero_()

        with torch.no_grad():
            perturbed_data.data -= a * torch.sign(gradient)
            perturbed_data.data = torch.max(torch.min(perturbed_data.data, data_max), data_min)
    return perturbed_data.detach()