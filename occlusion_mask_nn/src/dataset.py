import math
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import scipy.ndimage as ndimage
from skimage.measure import (
    label,
    regionprops,
)


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255


def try_path(path, exts=(".png", ".jpg", ".jpeg")):
    if path.is_file():
        return path

    for ext in exts:
        path = path.with_suffix(ext)
        if path.is_file():
            return path

    return path


def crop_img_by_mask(img, mask):
    label_mask = label(mask)
    prop, = regionprops(label_mask)
    y_min, x_min, y_max, x_max = prop.bbox
    r = 0.1
    d = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    d = r * d
    y_min = max(0, int(y_min - d))
    x_min = max(0, int(x_min - d))
    y_max = min(img.shape[0] - 1, int(y_max + d))
    x_max = min(img.shape[1] - 1, int(x_max + d))
    box = y_min, x_min, y_max, x_max
    img_crop = img[y_min:y_max, x_min:x_max].copy()

    return img_crop, box


def remove_small_objects(mask, thresh=100):
    contours, *_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)
    new_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= thresh:
            continue

        new_contours.append(c)

    new_mask = np.zeros_like(mask)
    cv2.drawContours(new_mask, new_contours, -1, (255, ), thickness=cv2.FILLED)

    return new_mask


def read_mask(path, low=(0, 0, 238), high=(2, 2, 255)):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = img[:, :, 1:]
    img = cv2.inRange(img, low, high)
    img = remove_small_objects(img)
    #img = cv2.medianBlur(img, 5)
    #img = cv2.medianBlur(img, 3)

    return img


def read_seg_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = img[:, :, 1:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    return img


def read_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = img[..., -1]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.bitwise_not(img)

    return img


def get_masks(mask, vmask):
    kernel = ndimage.generate_binary_structure(2, 2)
    iterations = int(0.006 * max(mask.shape[:2]))
    #print(iterations, mask.shape)
    dilate = ndimage.binary_dilation(
        mask,
        structure=kernel,
        iterations=iterations,
        mask=~vmask,
        output=None, border_value=0, origin=0, brute_force=False,
    )
    dilate = (dilate * 255).astype("uint8")
    dilate = cv2.bitwise_or(dilate, vmask)

    return dilate


def get_test_augs(img_size):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(img_size), interpolation=1),
            A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_REPLICATE),
            #A.Resize(*img_size),
            A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ],
        additional_targets={
            "smask": "mask",
        }
    )


def get_augs(img_size, p=0.5):
    augs = [
        A.LongestMaxSize(max_size=max(img_size), interpolation=1),
        A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_REPLICATE),
        #A.Resize(*img_size),
        A.HorizontalFlip(p=p),

        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=p),
        A.OneOf(
            [
                A.Blur(p=1),
                A.GlassBlur(p=1),
                A.GaussianBlur(p=1),
                A.MedianBlur(p=1),
                A.MotionBlur(p=1),
            ],
            p=p,
        ),
        A.RandomBrightnessContrast(p=p),
        A.OneOf(
            [
                A.RandomGamma(p=1),  # works only for uint
                A.ColorJitter(p=1),
                A.RandomToneCurve(p=1),  # works only for uint
            ],
            p=p,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
            ],
            p=p,
        ),
        A.OneOf(
            [
                A.PiecewiseAffine(),
                A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT),
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            ],
            p=0.2,
        ),
        #A.FancyPCA(p=0.2),
        #A.RandomFog(p=0.2),
        #A.RandomShadow(p=0.2),
        #A.RandomSunFlare(src_radius=150, p=0.2),
        A.ShiftScaleRotate(rotate_limit=15, p=p),#border_mode=cv2.BORDER_CONSTANT, p=0.2),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ]

    augs = A.Compose(
        augs,
        additional_targets={
            "smask": "mask",
        }
    )

    return augs


class DS(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, augs=None):
        self.df = df
        self.img_dir = img_dir
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            return self._getitem(index)
        except:
            index = (index + 1) % self.__len__()
            return self.__getitem__(index)

    def _getitem(self, index):
        item = self.df.iloc[index]

        img_fname = item.fname
        img = read_img(self.img_dir / img_fname)
        img_ori = img.copy()
        mask_path = self.img_dir / img_fname.replace("img", "mask")
        mask = read_mask(mask_path)
        smask_path = self.img_dir / img_fname.replace("img", "mask")
        smask = read_seg_mask(smask_path)

        if self.augs is not None:
            data = self.augs(
                image=img,
                mask=mask,
                smask=smask,
            )
            img = data["image"]
            mask = data["mask"]
            smask = data["smask"]

        img = np.transpose(img, (2, 0, 1))
        #img = img.astype("float32") / 255
        #img -= 0.5
        #img *= 2
        mask = (mask > 0).astype("float32")[None]
        smask = (smask > 0).astype("float32")[None]

        item = {
            "img": img,
            "mask": mask,
            "smask": smask,
            "label": 1.0,
        }

        return item #img, mask#, img_ori


class TestDS(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, augs=None):
        self.df = df
        self.img_dir = img_dir
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            return self._getitem(index)
        except:
            index = (index + 1) % self.__len__()
            return self.__getitem__(index)

    def _getitem(self, index):
        item = self.df.iloc[index]

        img_fname = item.fname
        #if random.random() < 0.5:
        #    img_path = self.img_dir / img_fname
        #else:
        #    img_path = self.img_dir / img_fname.replace("img", "img_s")

        img_path = self.img_dir / img_fname
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = self.img_dir / img_fname.replace("img", "mask")
        #mask_path = self.img_dir / img_fname.replace("img_s", "mask")
        mask_path = try_path(mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        vmask_path = self.img_dir / img_fname.replace("img", "vmask")
        #vmask_path = self.img_dir / img_fname.replace("img_s", "vmask")
        vmask_path = try_path(vmask_path)
        vmask = cv2.imread(str(vmask_path), cv2.IMREAD_GRAYSCALE)

        mask = get_masks(mask, vmask)

        if self.augs is not None:
            data = self.augs(image=img, mask=mask)
            img = data["image"]
            mask = data["mask"]

        img = np.transpose(img, (2, 0, 1))
        #img = img.astype("float32") / 255
        #img -= 0.5
        #img *= 2
        mask = (mask > 127).astype("float32")[None]
        smask = np.zeros_like(mask)

        item = {
            "img": img,
            "mask": mask,
            "smask": smask,
            "label": 0.0,
        }

        return item #img, mask#, img_ori


class TestDSUN(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, augs=None):
        self.df = df
        self.img_dir = img_dir
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            return self._getitem(index)
        except:
            index = (index + 1) % self.__len__()
            return self.__getitem__(index)

    def _getitem(self, index):
        item = self.df.iloc[index]

        img_fname = item.fname
        img_path = self.img_dir / img_fname
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        smask_path = self.img_dir / img_fname.replace("img", "m")
        #smask_path = self.img_dir / img_fname.replace("img_s", "m")
        smask_path = try_path(smask_path)
        smask = cv2.imread(str(smask_path), cv2.IMREAD_GRAYSCALE)
        _, smask = cv2.threshold(smask, 127, 255, cv2.THRESH_BINARY)
        img, box = crop_img_by_mask(img, smask)
        y_min, x_min, y_max, x_max = box

        smask = smask[y_min:y_max, x_min:x_max].copy()

        mask_path = self.img_dir / img_fname.replace("img", "mask")
        #mask_path = self.img_dir / img_fname.replace("img_s", "mask")
        mask_path = try_path(mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = mask[y_min:y_max, x_min:x_max].copy()
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if self.augs is not None:
            data = self.augs(
                image=img,
                mask=mask,
                smask=smask,
            )
            img = data["image"]
            mask = data["mask"]
            smask = data["smask"]

        img = np.transpose(img, (2, 0, 1))
        #img = img.astype("float32") / 255
        #img -= 0.5
        #img *= 2
        mask = (mask > 127).astype("float32")[None]
        smask = (smask > 127).astype("float32")[None]

        item = {
            "img": img,
            "mask": mask,
            "smask": smask,
            "label": 1.0,
        }

        return item #img, mask#, img_ori


def predict_tta(models, images, ntta=1):
    result = images.new_zeros((images.shape[0], 1, images.shape[-2], images.shape[-1]))
    n = 0
    for model in models:
        logits = model(images)
        result += logits
        n += 1

        if ntta == 2:
            # hflip
            logits = model(torch.flip(images, dims=[-1]))
            result += torch.flip(logits, dims=[-1])
            n += 1

        if ntta == 3:
            # vflip
            logits = model(torch.flip(images, dims=[-2]))
            result += torch.flip(logits, dims=[-2])
            n += 1

        if ntta == 4:
            # hvflip
            logits = model(torch.flip(images, dims=[-2, -1]))
            result += torch.flip(logits, dims=[-2, -1])
            n += 1

    result /= n *len(models)

    return result
