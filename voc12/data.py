import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import random
import pyarrow as pa


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def load_image_label_pair_list_from_npy(img_name_pair_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    return [(cls_labels_dict[img_name_pair[0]], cls_labels_dict[img_name_pair[1]]) for img_name_pair in img_name_pair_list]


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')
    # return os.path.join(voc12_root, img_name + '.jpg')


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][12:-4] for img_gt_name in img_gt_name_list]

    return img_name_list


def load_img_name_pair_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    # img_name_pair_list = [(img_gt_name.split(' ')[0][-15:-4], img_gt_name.split(' ')[1][-15:-4]) for img_gt_name in img_gt_name_list]
    # common_label_list = [int(img_gt_name.split(' ')[2]) for img_gt_name in img_gt_name_list]
    img_name_pair_list = [(img_gt_name.split(' ')[0][-15:-4], img_gt_name.split(' ')[1][-15:-4]) for img_gt_name in
                          img_gt_name_list]
    common_label_list = [int(img_gt_name.split(' ')[2]) for img_gt_name in img_gt_name_list]

    return img_name_pair_list, common_label_list


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label


class VOC12EDAMClsDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_pair_list,self.common_label_list = load_img_name_pair_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.label_pair_list = load_image_label_pair_list_from_npy(self.img_name_pair_list)

    def __len__(self):
        return len(self.img_name_pair_list)

    def __getitem__(self, idx):
        name_pair = self.img_name_pair_list[idx]

        img1 = PIL.Image.open(get_img_path(name_pair[0], self.voc12_root)).convert("RGB")
        img2 = PIL.Image.open(get_img_path(name_pair[1], self.voc12_root)).convert("RGB")

        if self.transform:
            img_pair = torch.stack((self.transform(img1), self.transform(img2)), dim=0)

        label1 = torch.from_numpy(self.label_pair_list[idx][0])
        label2 = torch.from_numpy(self.label_pair_list[idx][1])

        commen_label = self.common_label_list[idx]

        list1 = [i for i in range(20) if label1[i] == 0 and label2[i] == 0]
        random.shuffle(list1)
        list2 = [i for i in range(20) if label1[i] == 1 or label2[i] == 1]
        label_idx = list2
        sample_num = 20
        if len(label_idx) > sample_num:
            label_idx = label_idx[:sample_num]
        for i in range(0, -len(label_idx)+sample_num):
            label_idx.append(list1[i])

        assert label1[commen_label] == 1 and label2[commen_label] == 1
        assert len(label_idx) == len(set(label_idx))
        list3 = [0 for _ in range(2*sample_num)]
        for i in range(sample_num):
            if label1[label_idx[i]] == 1:
                list3[i] = 1.0
            if label2[label_idx[i]] == 1:
                list3[i+sample_num] = 1.0
        label = torch.tensor(list3)

        return name_pair, img_pair, label, label_idx


class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label




