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
from src.utils.attack_methods import DI, TI, pgd
from src.utils.args import rap_parser
from src.utils.utils import logging, rap_initialize
from src.utils.model_loader import load_named_model, get_adv_CNNs_model
from src.utils.image_loader import Nips17, load_nips17_metadata


def rap_attack(args, src_model, target_models, loader, device):
    adv_activate = 0

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trn = transforms.Compose([transforms.ToTensor(), ])
    image_id_list, label_ori_list, _ = load_nips17_metadata('./NIPS17/images.csv')

    input_path = './NIPS17/images/'
    lr = 2 / 255  # step size
    epsilon = 16  # L_inf norm bound

    ##define TI
    gaussian_kernel = TI(device)

    if args.loss_function == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_function == 'MaxLogit':
        criterion = lambda logits, labels: (-1 * logits.gather(1, labels.unsqueeze(1)).squeeze(1)).sum()
    else:
        raise KeyError("loss function not supported")
    
    get_raw_logits = lambda x, model: model(norm(x))
        
    if args.DI:
        get_logits = lambda x, model: model(norm(DI(x)))
    else:
        get_logits = lambda x, model: model(norm(x))

    ckpt = 10
    pos = np.zeros((len(target_models), args.max_iterations // ckpt))

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

        for t in range(args.max_iterations):
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
                        
                        X_advaug = pgd(args, src_model, get_raw_logits, images+delta+X_addin, label_pred, args.adv_targeted, args.adv_epsilon, args.adv_steps, args.adv_alpha)
                        X_aug = X_advaug - (images+delta+X_addin)

                    else:
                        X_aug = torch.zeros_like(images).to(device)
                    delta.requires_grad_(True)

                    logits = get_logits(images + delta + X_addin + X_aug, src_model)

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
        if args.save_path is not None:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            for j in range(images.shape[0]):
                x_np = transforms.ToPILImage()((images + delta)[j].detach().cpu())
                x_np.save(os.path.join(args.save_path, images_ID[j]))


    logging(args, "Final result")
    logging(args, f'Source model: {args.source_model} --> Target model: Inception-v3 | ResNet50 | DenseNet121 | VGG16bn | resnet_v2_152 | ens3_adv_inc_v3 | ens4_adv_inc_v3 | ens_adv_inc_res_v2 | tf2torch_adv_inception_v3 ')
    logging(args, str(pos))

    logging(args, "Experiment finished")
    logging(args, 50*"#")


if __name__ == "__main__":
    args = rap_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rap_initialize(args)

    adv_CNN_model_names = [
        'tf2torch_resnet_v2_152',
        'tf2torch_ens3_adv_inc_v3',
        'tf2torch_ens4_adv_inc_v3',
        'tf2torch_ens_adv_inc_res_v2',
        'tf2torch_adv_inception_v3',
        ]

    logging(args, "setting up the source and target models")

    model_1 = load_named_model('inception_v3').eval().to(device)
    model_2 = load_named_model('resnet50').eval().to(device)
    model_3 = load_named_model('densenet121').eval().to(device)
    model_4 = load_named_model('vgg16_bn').eval().to(device)

    target_models = [model_1, model_2, model_3, model_4]
    target_models += [get_adv_CNNs_model(model_name, "./tf2torch_models/").eval().to(device) for model_name in adv_CNN_model_names]

    for target_model in target_models:
        for param in target_model.parameters():
            param.requires_grad = False

    if args.source_model == 'inception-v3':
        src_model = load_named_model('inception_v3').eval().to(device)
    elif args.source_model == 'resnet50':
        src_model = load_named_model('resnet50').eval().to(device)
    elif args.source_model == 'densenet121':
        src_model = load_named_model('densenet121').eval().to(device)
    elif args.source_model == 'vgg16bn':
        src_model = load_named_model('vgg16_bn').eval().to(device)

    for param in src_model.parameters():
        param.requires_grad = False

    nips17_images_dir = r"./NIPS17/images"
    csv_dir = r"./NIPS17/images.csv"
    nips17_data = Nips17(nips17_images_dir, csv_dir, T.Compose([T.ToTensor()]))
    nips17_loader = Data.DataLoader(nips17_data, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.workers)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    rap_attack(args, src_model, target_models, nips17_loader, device)
