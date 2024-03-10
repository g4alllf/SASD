import torch
from torchvision import transforms
from tqdm.auto import tqdm
import torch.nn.functional as F
import os
import torch.utils.data as Data
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from pynvml import *
from src.utils.image_loader import Nips17, load_nips17_metadata
from src.utils.model_loader import get_target_models, get_src_models
from src.utils.args import eval_parser
from src.utils.attack_methods import DI, TI
from src.utils.utils import log_to_file


def eval_adv_CNNs(targeted, target_models, adv_images, true_labels, target_labels, results, t, ckpt):
    with torch.no_grad():
        for i, target_model in enumerate(target_models):
            if targeted:
                results[i, t // ckpt] += sum(torch.argmax(target_model(adv_images), dim=1) == target_labels + 1).cpu().numpy()
            else:
                results[i, t // ckpt] += sum(torch.argmax(target_model(adv_images), dim=1) != true_labels + 1).cpu().numpy()


def eval_ViTs(targeted, target_models, adv_images, true_labels, target_labels, processors, results, t, ckpt, device):
    with torch.no_grad():
        for i, (target_model, processor) in enumerate(zip(target_models, processors)):
            inputs = processor(images=adv_images, return_tensors="pt").to(device)
            if targeted:
                results[i, t // ckpt] += sum(torch.argmax(target_model(**inputs).logits, dim=1) == target_labels).cpu().numpy()
            else:
                results[i, t // ckpt] += sum(torch.argmax(target_model(**inputs).logits, dim=1) != true_labels).cpu().numpy()


def eval_CLIP(targeted, target_models, adv_images, true_labels, target_labels, text, preprocess, results, t, ckpt):
    with torch.no_grad():
        for i, target_model in enumerate(target_models):
            logits, _ = target_model(preprocess(adv_images), text)
            probs = logits.softmax(dim=-1).detach()
            if targeted:
                results[i, t // ckpt] += sum(torch.argmax(probs, dim=1) == target_labels).cpu().numpy()
            else:
                results[i, t // ckpt] += sum(torch.argmax(probs, dim=1) != true_labels).cpu().numpy()


def attack(args, func_get_models, adv_CNNs, ViT_models, CLIP_models, processors, text, preprocess, loader, device):
    """
    Generate targeted perturbations for clean images and test on the target models
    ### Args:
        func_get_models: functions to get source models
        adv_CNNs: adversarial-trained convolutional networks
        ViT_models: vision transformers
        CLIP_models: openai pretrained clip models
        processors: processors to process the input of vision transformers
        text: image labels encoded with transformer
        loader: dataloader of the input images
    """
    lr = args.learning_rate
    epsilon = args.epsilon
    max_iterations = args.max_iterations
    ckpt = 10
    save_path = args.save_path
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trn = transforms.Compose([transforms.ToTensor(), ])
    image_id_list, label_ori_list, _ = load_nips17_metadata('./NIPS17/images.csv')

    adv_CNNs_results = np.zeros((len(adv_CNNs), max_iterations // ckpt))
    ViT_models_results = np.zeros((len(ViT_models), max_iterations // ckpt))
    CLIP_models_results = np.zeros((len(CLIP_models), max_iterations // ckpt))

    if save_path is not None:
        save_path += args.src_type 

    if args.TI:
        print("TI")
        gaussian_kernel = TI(device)
        if save_path is not None:
            save_path += "_TI"
    if args.DI:
        print("DI")
        get_logits = lambda x, models: torch.mean(torch.stack([m(norm(DI(x))) for m in models]), 0)
        if save_path is not None:
            save_path += "_DI"
    else:
        get_logits = lambda x, models: torch.mean(torch.stack([m(norm(x)) for m in models]), 0)
    if args.MI:
        print("MI")
        if save_path is not None:
            save_path += "_MI"

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
            src_models = func_get_models(t)
            adv_images = images + delta

            logits = get_logits(adv_images, src_models)
            
            if args.targeted:
                loss = criterion(logits, target_labels)
            else:
                loss = -criterion(logits, true_labels)
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
                with torch.no_grad():
                    perturbed = images + delta
                    eval_adv_CNNs(args.targeted, adv_CNNs, perturbed, true_labels, target_labels, adv_CNNs_results, t, ckpt)
                    eval_CLIP(args.targeted, CLIP_models, perturbed, true_labels, target_labels, text, preprocess, CLIP_models_results, t, ckpt)
                    ViT_images = [to_pil_image(img) for img in perturbed.cpu()]
                    eval_ViTs(args.targeted, ViT_models, ViT_images, true_labels, target_labels, processors, ViT_models_results, t, ckpt, device)
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(images.shape[0]):
                x_np = transforms.ToPILImage()((images + delta)[j].detach().cpu())
                x_np.save(os.path.join(save_path, images_ID[j]))
    print(f"Adv CNNs results: ", adv_CNNs_results.squeeze())
    print(f"ViT models results: ", ViT_models_results.squeeze())
    print(f"CLIP models results: ", CLIP_models_results.squeeze())
    log_to_file(f"Adv CNNs results: {adv_CNNs_results.squeeze()}", f'./results/{args.src_type}_ens6_eval_results.txt')
    log_to_file(f"ViT models results: {ViT_models_results.squeeze()}", f'./results/{args.src_type}_ens6_eval_results.txt')
    log_to_file(f"CLIP models results: {CLIP_models_results.squeeze()}", f'./results/{args.src_type}_ens6_eval_results.txt')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = eval_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    preprocess = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    print(f"📷 Loading {args.src_type} models...")
    src_models, func_get_models = get_src_models(args.src_type, args.scaling_ratio, device)

    print(f"📷 Loading target models...")
    adv_CNNs, ViT_models, CLIP_models, processors, text = get_target_models(device)

    print(f"📷 Loading Nips17 (num_worker: {args.workers}, batch_size: {args.batch_size})...")
    nips17_images_dir = r"./NIPS17/images"
    csv_dir = r"./NIPS17/images.csv"
    nips17_data = Nips17(nips17_images_dir, csv_dir, T.Compose([T.ToTensor()]))
    nips17_loader = Data.DataLoader(nips17_data, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.workers)

    print(f"🪴 Start evaluating ensemble {args.src_type} 'incn_v3', 'inc_v4', 'inc_res_v2', 'res50', 'res101', 'res152' -> adv CNNs, ViTs and CLIP")
    log_to_file('Hyper-parameters: {}'.format(args.__dict__), f'./results/{args.src_type}_ens6_eval_results.txt')
    attack(args, func_get_models, adv_CNNs, ViT_models, CLIP_models, processors, text, preprocess, nips17_loader, device)