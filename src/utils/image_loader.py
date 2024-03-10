import torch, torchvision
import pandas as pd
import sys
import os
import torchvision.transforms as T
import torch.utils.data as Data
import src.utils.image_process as process
import csv

sys.path.append("..")
sys.path.append(".")

current_path = os.path.dirname(__file__) + "/"


def load_imagenet(imagenet_path: str, batch_size=16, num_worker=0, 
                  sets=["train", "val"], transform=process.__get_input_transform()):
    """

    This function will load `num_imgs_per_class` images from every folder of `classes_name`
    as the order of their file_name. `num_skip_imgs` determined how many imgs to skip of
    each class. If `class_names` is not passed, it will be set to all 1000 classes by default.

    `sets` is a list consist of the datasets you want to load. "train", "val" and "filtered_val" 
    and `100_val` are supported.

    `imagenet_path` should be like this:
    ```
        imagenet_path
        ├── ILSVRC2012_devkit_t12.tar.gz
        ├── ILSVRC2012_img_train.tar
        └── ILSVRC2012_img_val.tar
    ```
    """

    train_loader = None
    val_loader = None

    if "train" in sets:
        train_folder = torchvision.datasets.ImageNet(
            root=os.path.join(imagenet_path),
            split="train",
            transform=transform,
        )
        train_loader = Data.DataLoader(
            train_folder, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    if "val" in sets:
        val_folder = torchvision.datasets.ImageNet(
            root=os.path.join(imagenet_path),
            split="val",
            transform=transform,
        )
        val_loader = Data.DataLoader(dataset=val_folder, shuffle=True,
                                     batch_size=batch_size, num_workers=num_worker)

    return train_loader, val_loader


class Nips17(Data.Dataset):
    def __init__(self, image_path, csv_path, transforms = None):
        self.dir = image_path   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        data = None
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        pil_img = process.get_image(os.path.join(self.dir, ImageID))
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel, TargetClass

    def __len__(self):
        return len(self.csv)


def load_nips17_metadata(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list