import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from pynvml import *
from src.utils.SAM import SAM
from src.utils.image_loader import load_imagenet
from src.utils.args import SASD_parser
from src.utils.model_loader import load_named_model
from src.utils.utils import auto_file_name


def KL_Divergence(scores, targets, temperature):
    soft_pred = F.log_softmax(scores / temperature, dim=1)
    soft_target = F.softmax(targets / temperature, dim=1)
    loss = torch.nn.KLDivLoss(reduction='batchmean')(soft_pred, soft_target)
    return loss


def distill(args, student_model, teacher_model, save_path, device='cuda'):
    """
    Distill the knowledge of the finetuned model into the student model
    """

    epochs = args.epochs
    lr = args.learning_rate
    temperature = args.temperature

    print(f"📷 Loading imagenet (num_worker: {args.workers}, batch_size: {args.batch_size})...")
    train_loader, _ = load_imagenet(
        r"./imagenet",
        num_worker=args.workers,
        batch_size=args.batch_size,
        sets=["train"]
    )

    student_model = student_model.train().to(device)
    teacher_model = teacher_model.to(device)

    optimizer = None
    if args.sharpness_aware:
        print(f'⏳ Start sharpness aware self distillation (epochs: {args.epochs}, lr: {args.learning_rate})...')
        optimizer = SAM(student_model.parameters(), torch.optim.SGD, lr=lr)
    else:
        print(f'⏳ Start normal distillation (epochs: {args.epochs}, lr: {args.learning_rate})...')
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    count = 0
    for i in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=80)
        train_num = 0
        loss_mean_epoch = 0
        student_target_mean_epoch = 0
        count_images = 0
        for _, batch in loop:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scores = student_model(batch_x)
            targets = teacher_model(batch_x)
            student_target_loss = criterion(scores, batch_y)
            loss = KL_Divergence(scores, targets, temperature)
            if not args.sharpness_aware:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)
                loss = KL_Divergence(student_model(batch_x), targets, temperature)
                loss.backward()
                optimizer.second_step(zero_grad=True)
                
            # calculate mean loss
            loss_mean_epoch = (loss_mean_epoch * train_num + loss.item()) / (train_num + 1)
            student_target_mean_epoch = (student_target_mean_epoch * train_num + student_target_loss.item()) / (train_num + 1)
            train_num += 1
            loop.set_description(f'Epoch [{i + 1}/{epochs}]')
            loop.set_postfix(trloss=loss.item())

            count_images += batch_x.shape[0]
            if count_images > 100000:
                count_images = 0
                count += 1
                file_name = f'single_teacher_distill_{args.model}_min_val_loss_SAM,lr={args.learning_rate},t={args.temperature},{count}-12.pth'
                print(f'{file_name} saved.')
                torch.save(student_model.state_dict(), auto_file_name(save_path, file_name))
            
        if args.sharpness_aware:
            torch.save(student_model.state_dict(),
                auto_file_name(save_path, f'pretrain_teacher_distill_{args.model}_{i + args.offset}_SAM,lr={args.learning_rate},t={args.temperature}.pth'))
        else:
            torch.save(student_model.state_dict(),
                auto_file_name(save_path, f'single_teacher_distill_{args.model}_{i + args.offset},raw.pth'))


if __name__ == "__main__":
    args = SASD_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert args.model is not None
    student_model = load_named_model(args.model)
    teacher_weight_path = f'./results/raw_SAM_lr_0.05_models/{args.model}_epochs_4-4_SAM_lr_0.05,raw.pth'
    teacher_model = load_named_model(args.model, teacher_weight_path)
    save_path = './results/pretrain_SAM_distilled_models'

    distill(args, student_model, teacher_model, save_path, device)