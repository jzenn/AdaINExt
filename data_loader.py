import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

import os


# the device being on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# try to load truncated images as well
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set max image pixel up
Image.MAX_IMAGE_PIXELS = None


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class MSCOCODataset(Dataset):
    """
    dataset class for MS-COCO dataset
    """
    def __init__(self, root_dir, loader):
        self.root_dir = root_dir
        self.loader = loader
        self.image_list = [x for x in os.listdir(root_dir) if is_image_file(x)]
        self.len = len(self.image_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = image_loader(img_name, self.loader, add_fake_batch_dimension=False)
        sample = {'image': image}

        return sample


class PainterByNumbersDataset(Dataset):
    """
    dataset class for kaggle painter-by-numbers dataset
    """
    def __init__(self, root_dir, loader):
        self.root_dir = root_dir
        self.loader = loader
        self.image_list = [x for x in os.listdir(root_dir) if is_image_file(x)]
        self.len = len(self.image_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = image_loader(img_name, self.loader, add_fake_batch_dimension=False)
        sample = {'image': image}

        return sample


class MergeCOCOPainterByNumbersDataset(Dataset):
    """
    dataset combining MS-COCO dataset and painter-by-numbers dataset
    """
    def __init__(self, coco_dataset, painter_by_numbers_dataset):
        self.coco_dataset = coco_dataset
        self.painter_by_numbers_dataset = painter_by_numbers_dataset
        self.len = min(self.painter_by_numbers_dataset.__len__(), self.coco_dataset.__len__())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        coco_data = self.coco_dataset.__getitem__(idx)
        painter_by_numbers_data = self.painter_by_numbers_dataset.__getitem__(idx)
        sample = {'coco': coco_data, 'painter_by_numbers': painter_by_numbers_data}
        return sample


def get_concat_dataloader(coco_data_path, painter_by_numbers_data_path, batch_size, loader):
    coco_dataset = MSCOCODataset(coco_data_path, loader=loader)
    painter_by_numbers_dataset = PainterByNumbersDataset(painter_by_numbers_data_path, loader=loader)
    merge_coco_painter_by_numbers_dataset = MergeCOCOPainterByNumbersDataset(coco_dataset, painter_by_numbers_dataset)

    concat_dataloader = DataLoader(merge_coco_painter_by_numbers_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4)

    return concat_dataloader


def image_loader(image_name, transformation, add_fake_batch_dimension=True):
    """
    loads an image
    :param image_name: the path of the image
    :param transformation: the transformation done on the image
    :param add_fake_batch_dimension: shoud add a 4th batch dimension
    :return: the image on the current device
    """
    image = Image.open(image_name).convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    if add_fake_batch_dimension:
        image = transformation(image).unsqueeze(0)
    else:
        image = transformation(image)
    # return image.to(device, torch.float)
    return image


def imnorm(tensor, transformation):
    """
    normalizes an image recived from EncoderDecoder
    :param tensor: the image as tensor
    :param transformation: a transformation applied to the image before saving
    :return:
    """
    # clone the tensor to not change the original one
    image = tensor.cpu().clone()
    # remove the batch dimension
    image = image.squeeze(0)
    if transformation is not None:
        image = transformation(image)

    return image