import re
from typing import Dict, Type, List

import torch
from torch.utils.data import Dataset

dataset_registry: Dict[str, Type] = {}


def register_dataset(name: str):
    def decorator(cls: Type[Dataset]):
        dataset_registry[name] = cls
        return cls

    return decorator


class ImageEvaluationDataRaw:
    """
    image: torch.Tensor # C H W
    label: int # B
    annotation: object # str
    {
     "dist_anchor": bool
     "anchor_image: bool
    }
    """

    def __init__(self, image: torch.Tensor, label: int = 0, annotation: object = None):
        self.image = image
        self.label = label
        self.annotation = annotation


# collected object processed by dataloader
class ImageEvaluationData:
    """
    image: torch.Tensor # B C H W
    label: torch.Tensor # B
    annotations: List # e.g. List[str]
    """

    def __init__(self, image: torch.Tensor, label: torch.Tensor = None, annotations: List = None):
        self.image = image
        self.label = label
        self.annotations = annotations


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


@register_dataset("imagenet")
class ImageNetDataset(Dataset):
    def __init__(self, images_path, label_path, target_size=None, label_start=0, target_label_path = None, anchor_images_path=None,
                 dist_anchor_images_path=None, **kwargs):
        # attack algorithm shall save a One2One Image Path
        """
        Args:
            root_dir (string): 图像所在的根目录路径。
            annotation_file (string): 包含图像文件名和标签的文件路径。
            transform (callable, optional): 一个用于进行图像变换的可调用对象。
        """
        self.root_dir = images_path
        self.root_dir_anchor_images = anchor_images_path
        self.label_start = label_start
        self.image_files_anchor = []
        self.annotations = self._load_annotations(label_path)

        if target_label_path is not None:
            self.annotations_target_label = self._load_annotations(target_label_path)
        else:
            self.annotations_target_label = None

        if dist_anchor_images_path is not None:
            self.load_dist_anchors(dist_anchor_images_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ]) if target_size is None else transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def _load_annotations(self, annotation_file):
        # 加载标签文件，并存储为一个字典，键为图像文件名，值为类别编号
        annotations = {}
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = re.split(r"[,\s\r\t]+", line.strip())
                print(parts)
                if len(parts) == 2:
                    file_name, class_id = parts
                    annotations[file_name] = int(class_id) - self.label_start

        return annotations

    def load_dist_anchors(self, folder_path):
        for f in os.listdir(folder_path):
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                self.image_files_anchor.append(os.path.join(folder_path, f))

    def __len__(self):
        return len(self.annotations) + len(self.image_files_anchor)

    def __getitem__(self, idx):
        if idx >= len(self.annotations):
            idx = idx - len(self.annotations)
            file_name = self.image_files_anchor[idx]
            # class_id = # TODO
            img_path = file_name
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return ImageEvaluationDataRaw(image=image, label=0, annotation={"dist_anchor": True})

        # 根据 idx 获取文件名和类别编号
        file_name = list(self.annotations.keys())[idx]
        class_id = self.annotations[file_name]
        # 构建图像文件的完整路径
        img_path = os.path.join(self.root_dir, file_name)

        # 读取图像
        image = Image.open(img_path)
        # 应用变换
        if self.transform:
            image = self.transform(image)

        annonations = {"dist_anchor": False, "anchor_image": None, "target_label": None}
        if self.annotations_target_label is not None:
            annonations["target_label"] = self.annotations_target_label[file_name]

        if self.root_dir_anchor_images is not None:
            image1 = Image.open(os.path.join(self.root_dir_anchor_images, file_name))
            annonations["anchor_image"] = self.transform(image1) if self.transform else image1

        return ImageEvaluationDataRaw(image=image, label=class_id, annotation=annonations)

@register_dataset("imagenet_multilabel")
class ImageNetDatasetMultiLabel(ImageNetDataset):
    def __init__(self, images_path, label_path, target_size=None, label_start=0, target_label_path=None,
                 anchor_images_path=None,
                 dist_anchor_images_path=None, **kwargs):
        super().__init__(images_path, label_path, target_size, label_start, target_label_path, anchor_images_path,
                         dist_anchor_images_path, **kwargs)


    def _load_annotations(self, annotation_file):
        # 加载标签文件，并存储为一个字典，键为图像文件名，值为类别编号
        annotations = {}
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line[:1] + line[1:].replace("'s", "\\'s")
                filename, labels = eval(line)
                annotations[filename] = labels

        return annotations

    def __getitem__(self, idx):
        if idx >= len(self.annotations): # anchor distribution
            idx = idx - len(self.annotations)
            file_name = self.image_files_anchor[idx]
            # class_id = # TODO
            img_path = file_name
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return ImageEvaluationDataRaw(image=image, label=0, annotation={"dist_anchor": True})

        # 根据 idx 获取文件名和类别编号
        file_name = list(self.annotations.keys())[idx]
        class_ids = self.annotations[file_name]
        # 构建图像文件的完整路径
        img_path = os.path.join(self.root_dir, file_name)

        # 读取图像
        image = Image.open(img_path)
        # 应用变换
        if self.transform:
            image = self.transform(image)

        annonations = {"dist_anchor": False, "anchor_image": None, "target_label": None}
        annonations["class_id"] = torch.tensor(class_ids, dtype=torch.long)

        semantic_name = re.sub(r'_\d+$', '', file_name.split(".")[0])
        semantic_name = semantic_name.replace("_", " ")
        semantic_name = semantic_name.replace(",", ", ")
        annonations["semantic"] = semantic_name
        # remove file postfix
        if self.annotations_target_label is not None:
            annonations["target_label"] = self.annotations_target_label[file_name]

        if self.root_dir_anchor_images is not None:
            image1 = Image.open(os.path.join(self.root_dir_anchor_images, file_name))
            annonations["anchor_image"] = self.transform(image1) if self.transform else image1

        return ImageEvaluationDataRaw(image=image, label=class_ids[0], annotation=annonations)

    def __len__(self):
        return len(self.annotations) + len(self.image_files_anchor)

def get_dataloader(dataset_config):
    type1: str = dataset_config.type
    kw = dataset_config.copy()
    del kw.type
    dataset = dataset_registry[type1](**kw)

    def collate_fn(batch: List[ImageEvaluationDataRaw]):
        return ImageEvaluationData.__new__(ImageEvaluationData, batch)

    return DataLoader(dataset, batch_size=dataset_config.batch_size, shuffle=False,
                      pin_memory=False,  # False since no multi epoch is required.
                      num_workers=dataset_config.num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
    print(dataset_registry.keys())
