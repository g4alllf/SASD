import torch
from tqdm.auto import tqdm
import os
from pynvml import *
from src.utils.SAM import SAM
from src.utils.image_loader import load_imagenet
from src.utils.args import finetune_parser
from src.utils.utils import auto_file_name
from src.utils.model_loader import load_named_model


def finetune(args, model, save_path, device='cuda'):
    """
    Finetune the pretrained model
    ### Args:
        model: pretrained source model
        save_path: path to save the finetuned model's state dict
    """

    if args.weight_path is not None:
        print(f"💾 Use wegiht from '{args.weight}'.")
    print(f"📷 Loading imagenet (num_worker: {args.workers}, batch_size: {args.batch_size}, lr: {args.learning_rate})...")
    train_loader, _ = load_imagenet(
        r"./imagenet",
        num_worker=args.workers,
        batch_size=args.batch_size,
        sets=["train"]
    )

    print(f'⏳ Fine tune (epochs: {args.epochs})...')
    
    if args.optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    elif args.optimizer == "SAM":
        base_opt = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_opt, lr=args.learning_rate)
    else:
        raise KeyError("Unsupported Optimizer Type!")
    
    criterion = torch.nn.CrossEntropyLoss()
    model = model.train().to(device)
    for epoch in range(args.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=80)
        checkpoint_num = len(train_loader) // 4
        for step, batch in loop:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            if optimizer is None:
                optimizer.step()
                optimizer.zero_grad()
            elif optimizer == "SAM":
                optimizer.first_step(zero_grad=True)
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.second_step(zero_grad=True)

            loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}]')
            loop.set_postfix(trloss=loss.item())
            if (step % checkpoint_num == checkpoint_num - 1):
                try:
                    weight_file, _ = os.path.splitext(os.path.split(args.weight_path)[1])
                except Exception:
                    weight_file = "raw"
                cal_saved_num = (epoch)*4 + (step + 1) // checkpoint_num

                torch.save(model.state_dict(), auto_file_name(
                    save_path, f'{args.model}_epochs_{cal_saved_num}-4_{optimizer}_lr_{args.learning_rate},{weight_file}.pth'))
                print(f"Fine tuned model {cal_saved_num}/4 Saved.")


if __name__ == "__main__":
    args = finetune_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert args.model is not None
    model = load_named_model(args.model, args.weight_path)
    save_path = f'./results/raw_SAM_lr_{args.learning_rate}_models'
    finetune(args, model, save_path, device)
