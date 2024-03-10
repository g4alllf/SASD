'''
This part of code comes from "Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation"
@article{qin2022boosting,
  title={Boosting the transferability of adversarial attacks with reverse adversarial perturbation},
  author={Qin, Zeyu and Fan, Yanbo and Liu, Yi and Shen, Li and Zhang, Yong and Wang, Jue and Wu, Baoyuan},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={29845--29858},
  year={2022}
}
'''


import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
import torch.utils.data as Data
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from src.utils.attack_methods import DI, TI, pgd
from src.utils.args import rap_parser
from src.utils.utils import logging, rap_initialize
from src.utils.model_loader import get_src_models, get_target_models
from src.utils.image_loader import Nips17, load_nips17_metadata
from src.eval_ensemble import eval_adv_CNNs, eval_ViTs, eval_CLIP


def ensemble_rap_attack(args, func_get_models, adv_CNNs, ViT_models, CLIP_models, processors, text, preprocess, loader, device, exp_name):
    adv_activate = 0

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trn = transforms.Compose([transforms.ToTensor(), ])
    image_id_list, label_ori_list, _ = load_nips17_metadata('./NIPS17/images.csv')

    input_path = './NIPS17/images/'
    lr = 2 / 255  # step size
    epsilon = 16  # L_inf norm bound
    max_iterations = args.max_iterations

    ##define TI
    gaussian_kernel = TI(device)

    if args.loss_function == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_function == 'MaxLogit':
        criterion = lambda logits, labels: (-1 * logits.gather(1, labels.unsqueeze(1)).squeeze(1)).sum()
    else:
        raise KeyError("loss function not supported")
    
    get_raw_logits = lambda x, models: torch.mean(torch.stack([m(norm(x)) for m in models]), 0)
        
    if args.DI:
        get_logits = lambda x, models: torch.mean(torch.stack([m(norm(DI(x))) for m in models]), 0)
    else:
        get_logits = lambda x, models: torch.mean(torch.stack([m(norm(x)) for m in models]), 0)

    ckpt = 10
    adv_CNNs_results = np.zeros((len(adv_CNNs), max_iterations // ckpt))
    ViT_models_results = np.zeros((len(ViT_models), max_iterations // ckpt))
    CLIP_models_results = np.zeros((len(CLIP_models), max_iterations // ckpt))

    for images, images_ID, true_labels, target_labels in tqdm(loader, ncols=80):
        grad_pre = 0
        images = images.to(device)
        delta = torch.zeros_like(images, requires_grad=True).to(device)
        true_labels = true_labels.to(device)
        target_labels = target_labels.to(device)

        if args.random_start:
            # Starting at a uniformly random point
            delta.requires_grad_(False)
            delta = delta + torch.empty_like(images).uniform_(-epsilon/255, epsilon/255)
            delta = torch.clamp(images+delta, min=0, max=1) - images
            delta.requires_grad_(True)

        for t in range(max_iterations):
            src_models = func_get_models(t)
            if t < args.transpoint:
                adv_activate = 0
            else:
                if args.adv_perturbation:
                    adv_activate = 1
                else:
                    adv_activate = 0
            grad_list = []

            for _ in range(args.m1):
                delta.requires_grad_(False)

                if args.strength == 0:
                    X_addin = torch.zeros_like(images).to(device)
                else:
                    X_addin = torch.zeros_like(images).to(device)
                    random_labels = torch.zeros(images.shape[0]).to(device)
                    stop = False
                    while stop == False:
                        random_indices = np.random.randint(0, 1000, images.shape[0])
                        for i in range(images.shape[0]):
                            X_addin[i] = trn(Image.open(input_path + image_id_list[random_indices[i]] + '.png'))
                            random_labels[i] = label_ori_list[random_indices[i]]
                        if torch.sum(random_labels==true_labels).item() == 0:
                            stop = True
                    X_addin = args.strength * X_addin
                    X_addin = torch.clamp(images+delta+X_addin, min=0, max=1) - (images+delta)
                

                delta.requires_grad_(True)

                for j in range(args.m2):

                    delta.requires_grad_(False)

                    if adv_activate:
                        if args.adv_targeted:
                            label_pred = true_labels
                        else:
                            label_pred = target_labels
                        
                        X_advaug = pgd(args, src_models, get_raw_logits, images+delta+X_addin, label_pred, args.adv_targeted, args.adv_epsilon, args.adv_steps, args.adv_alpha, device=device)
                        X_aug = X_advaug - (images+delta+X_addin)

                    else:
                        X_aug = torch.zeros_like(images).to(device)
                    delta.requires_grad_(True)

                    logits = get_logits(images + delta + X_addin + X_aug, src_models)

                    if args.targeted:
                        loss = criterion(logits, target_labels)
                    else:
                        loss = -criterion(logits, true_labels)

                    loss.backward()
                    grad_cc = delta.grad.clone().to(device)

                    if args.TI:  # TI
                        grad_cc = F.conv2d(grad_cc, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                    grad_list.append(grad_cc)
                    delta.grad.zero_()

            grad_c = 0

            for j in range(args.m1 * args.m2):
                grad_c += grad_list[j]
            grad_c = grad_c / (args.m1 * args.m2)

            if args.MI:  # MI
                grad_c = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                grad_pre = grad_c

            delta.data = delta.data - lr * torch.sign(grad_c)
            delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
            delta.data = ((images + delta.data).clamp(0, 1)) - images

            if t % ckpt == ckpt - 1:
                with torch.no_grad():
                    perturbed = images + delta
                    eval_adv_CNNs(args.targeted, adv_CNNs, perturbed, true_labels, target_labels, adv_CNNs_results, t, ckpt)
                    eval_CLIP(args.targeted, CLIP_models, perturbed, true_labels, target_labels, text, preprocess, CLIP_models_results, t, ckpt)
                    ViT_images = [to_pil_image(img) for img in perturbed.cpu()]
                    eval_ViTs(args.targeted, ViT_models, ViT_images, true_labels, target_labels, processors, ViT_models_results, t, ckpt, device)

        if args.save_path is not None:
            # save_path = './perturbed_images/' + exp_name
            save_path = args.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(images.shape[0]):
                x_np = T.ToPILImage()((images + delta)[j].detach().cpu())
                x_np.save(os.path.join(save_path, images_ID[j]))

    logging(args, "Final result")
    logging(args, 'Source models: Ensemble 6 models --> Target models: adv CNNs, ViTs, and CLIP ')
    logging(args, f"Adv CNNs results: "+str(adv_CNNs_results))
    logging(args, f"ViT models results: "+str(ViT_models_results))
    logging(args, f"CLIP models results: "+str(CLIP_models_results))

    logging(args, "Experiment finished")
    logging(args, 50*"#")


if __name__ == "__main__":
    args = rap_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_name = rap_initialize(args)

    preprocess = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    logging(args, "setting up the source and target models")

    print(f"📷 Loading source models...")
    src_models, func_get_models = get_src_models('pretrain', 1, device)

    print(f"📷 Loading target models...")
    adv_CNNs, ViT_models, CLIP_models, processors, text = get_target_models(device)

    print(f"📷 Loading Nips17 (num_worker: {args.workers}, batch_size: {args.batch_size})...")
    nips17_images_dir = r"./NIPS17/images"
    csv_dir = r"./NIPS17/images.csv"
    nips17_data = Nips17(nips17_images_dir, csv_dir, T.Compose([T.ToTensor()]))
    nips17_loader = Data.DataLoader(nips17_data, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.workers)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    ensemble_rap_attack(args, func_get_models, adv_CNNs, ViT_models, CLIP_models, processors, text, preprocess, nips17_loader, device, exp_name)

