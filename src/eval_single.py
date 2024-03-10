import torch
from torchvision import transforms
from tqdm.auto import tqdm
import torch.nn.functional as F
import os, copy
import torch.utils.data as Data
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pynvml import *
from src.utils.image_loader import Nips17, load_nips17_metadata
from src.utils.model_loader import get_src_model, get_adv_CNNs_model, load_named_model
from src.utils.args import eval_single_parser
from src.utils.attack_methods import DI, TI
from src.utils.utils import log_to_file


def attack(args, func_get_model, target_models, loader, device):
    """
    Generate targeted perturbations for clean images and test on the target models
    ### Args:
        func_get_model: functions to get source model
        target_models: pretrain and adv CNNs
        loader: dataloader of the input images
    """
    lr = args.learning_rate
    epsilon = args.epsilon
    max_iterations = args.max_iterations
    ckpt = 10
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trn = transforms.Compose([transforms.ToTensor(), ])
    image_id_list, label_ori_list, _ = load_nips17_metadata('./NIPS17/images.csv')

    pos = np.zeros((len(target_models), args.max_iterations // ckpt))

    if args.TI:
        print("TI")
        gaussian_kernel = TI(device)

    if args.DI:
        print("DI")
        get_logit = lambda x, model: model(norm(DI(x)))
    else:
        get_logit = lambda x, model: model(norm(x))
    if args.MI:
        print("MI")

    if args.lossfunc == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.lossfunc == 'logit':
        criterion = lambda logits, labels: (-1 * logits.gather(1, labels.unsqueeze(1)).squeeze(1)).sum()
    else:
        raise KeyError("loss function not supported")

    for images, images_ID, true_labels, target_labels in tqdm(loader, ncols=80):
        grad_pre = 0
        images = images.to(device)
        delta = torch.zeros_like(images,requires_grad=True).to(device)
        true_labels = true_labels.to(device)
        target_labels = target_labels.to(device)
        for t in range(max_iterations):
            src_model = func_get_model(t)
            adv_images = images + delta

            logits = get_logit(adv_images, src_model)
            
            loss = criterion(logits, target_labels)
            loss.backward()

            grad_c = delta.grad.clone()
            if args.TI:
                grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            if args.MI:
                if args.lossfunc == "ce":
                    grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                elif args.lossfunc == "logit":
                    grad_a = grad_c + 1 * grad_pre
                grad_pre = grad_a
            else:
                grad_a = grad_c
            delta.grad.zero_()

            # Manually update the perturbation
            delta.data = delta.data - lr * torch.sign(grad_a)
            delta.data = delta.data.clamp(-epsilon / 255,epsilon / 255)
            delta.data = ((images + delta.data).clamp(0,1)) - images
            
            # Record results every [ckpt] iterations
            if t % ckpt == ckpt - 1:
                for i, target_model in enumerate(target_models):
                    if i < 4:
                        if args.targeted:
                            pos[i, t // ckpt] += sum(torch.argmax(target_model(norm(images + delta)), dim=1) == target_labels).cpu().numpy()
                        else:
                            pos[i, t // ckpt] += sum(torch.argmax(target_model(norm(images + delta)), dim=1) != true_labels).cpu().numpy()
                    else:
                        if args.targeted:
                            pos[i, t // ckpt] += sum(torch.argmax(target_model(images + delta), dim=1) == target_labels + 1).cpu().numpy()
                        else:
                            pos[i, t // ckpt] += sum(torch.argmax(target_model(images + delta), dim=1) != true_labels + 1).cpu().numpy()

    print(str(pos))
    log_to_file(f"{str(pos)}", f'./results/{args.src_type}_single_eval_results.txt')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = eval_single_parser()
    log_to_file('Hyper-parameters: {}'.format(args.__dict__), f'./results/{args.src_type}_single_eval_results.txt')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"📷 Loading {args.src_type} {args.src_name}...")
    src_model, func_get_model = get_src_model(args.src_type, args.src_name, args.scaling_ratio, device)

    print(f"📷 Loading target models...")
    adv_CNN_model_names = [
        'tf2torch_resnet_v2_152',
        'tf2torch_ens3_adv_inc_v3',
        'tf2torch_ens4_adv_inc_v3',
        'tf2torch_ens_adv_inc_res_v2',
        'tf2torch_adv_inception_v3',
    ]
    model_1 = load_named_model('inception_v3').eval().to(device)
    model_2 = load_named_model('resnet50').eval().to(device)
    model_3 = load_named_model('densenet121').eval().to(device)
    model_4 = load_named_model('vgg16_bn').eval().to(device)

    target_models = [model_1, model_2, model_3, model_4]
    target_models += [get_adv_CNNs_model(model_name, "./tf2torch_models/").eval().to(device) for model_name in adv_CNN_model_names]

    for target_model in target_models:
        for param in target_model.parameters():
            param.requires_grad = False

    print(f"📷 Loading Nips17 (num_worker: {args.workers}, batch_size: {args.batch_size})...")
    nips17_images_dir = r"./NIPS17/images"
    csv_dir = r"./NIPS17/images.csv"
    nips17_data = Nips17(nips17_images_dir, csv_dir, T.Compose([T.ToTensor()]))
    nips17_loader = Data.DataLoader(nips17_data, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.workers)

    print(f"🪴 Start evaluating {args.src_type} {args.src_name} -> pretrain and adv CNNs")
    attack(args, func_get_model, target_models, nips17_loader, device)