""" COCO transforms (quick and dirty)

Hacked together by Ross Wightman
"""
import random
import math
from copy import deepcopy

from PIL import Image
import numpy as np
import torch

import albumentations as A
from .cap_multi_aug import *
from .bboxCapAug import bboxCapAug

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class ImageToNumpy:

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        # np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img, annotations


class NumpyToImage:
    def __call__(self, arr, annotations: dict):
        return Image.fromarray(arr, mode='RGB'), annotations


class HWC2CHW:

    def __call__(self, np_img, annotations: dict):
        np_img = np.moveaxis(np_img, 2, 0)
        return np_img, annotations


class ImageToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype), annotations


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def clip_boxes_(boxes, img_size):
    height, width = img_size
    clip_upper = np.array([height, width] * 2, dtype=boxes.dtype)
    np.clip(boxes, 0, clip_upper, out=boxes)


def clip_boxes(boxes, img_size):
    clipped_boxes = boxes.copy()
    clip_boxes_(clipped_boxes, img_size)
    return clipped_boxes


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size


class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img, anno: dict):
        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        if 'bbox' in anno:
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            bbox_bound = (min(scaled_h, self.target_size[0]), min(scaled_w, self.target_size[1]))
            clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original

        return new_img, anno


class RandomResizePad:

    def __init__(self, target_size: int, scale: tuple = (0.1, 2.0), interpolation: str = 'random',
                 fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.scale = scale
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.fill_color = fill_color

    def _get_params(self, img):
        # Select a random scale factor.
        scale_factor = random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]

        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.size
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * random.uniform(0, 1))
        return scaled_h, scaled_w, offset_y, offset_x, img_scale

    def __call__(self, img, anno: dict):
        scaled_h, scaled_w, offset_y, offset_x, img_scale = self._get_params(img)

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        img = img.resize((scaled_w, scaled_h), interpolation)
        right, lower = min(scaled_w, offset_x + self.target_size[1]), min(scaled_h, offset_y + self.target_size[0])
        img = img.crop((offset_x, offset_y, right, lower))
        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        if 'bbox' in anno:
            bbox = anno['bbox']  # for convenience, modifies in-place
            bbox[:, :4] *= img_scale
            box_offset = np.stack([offset_y, offset_x] * 2)
            bbox -= box_offset
            bbox_bound = (min(scaled_h, self.target_size[0]), min(scaled_w, self.target_size[1]))
            clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original

        return new_img, anno


class RandomFlip:

    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def _get_params(self):
        do_horizontal = random.random() < self.prob if self.horizontal else False
        do_vertical = random.random() < self.prob if self.vertical else False
        return do_horizontal, do_vertical

    def __call__(self, img, annotations: dict):
        do_horizontal, do_vertical = self._get_params()
        width, height = img.size

        def _fliph(bbox):
            x_max = width - bbox[:, 1]
            x_min = width - bbox[:, 3]
            bbox[:, 1] = x_min
            bbox[:, 3] = x_max

        def _flipv(bbox):
            y_max = height - bbox[:, 0]
            y_min = height - bbox[:, 2]
            bbox[:, 0] = y_min
            bbox[:, 2] = y_max

        if do_horizontal and do_vertical:
            img = img.transpose(Image.ROTATE_180)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
                _flipv(annotations['bbox'])
        elif do_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
        elif do_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if 'bbox' in annotations:
                _flipv(annotations['bbox'])

        return img, annotations


def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == 'mean'
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color


class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


class Albu:

    def __init__(self, albu_transform):
        self.transform = albu_transform

    def __call__(self, img, annotations: dict):
        bboxes = annotations['bbox']
        bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
        augmented = self.transform(image=img, bboxes=bboxes, labels=annotations['cls'])
        image = augmented['image']
        bboxes = augmented['bboxes']
        if len(bboxes) > 0:
            annotations['bbox'] = np.array(bboxes)[:, [1, 0, 3, 2]]
        else:
            annotations['bbox'] = np.empty(shape=(0, 4))
        annotations['cls'] = np.array(augmented['labels'])
        return image, annotations


def transforms_coco_eval(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    fill_color = resolve_fill_color(fill_color, mean)

    image_tfl = [
        ResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color),
        ImageToNumpy(),
        HWC2CHW()
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def transforms_coco_train(
        img_size=224,
        interpolation='random',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        mode='noaug'):

    fill_color = resolve_fill_color(fill_color, mean)

    train_transforms = A.Compose([
        # A.RandomSizedBBoxSafeCrop(img_size[0], img_size[1], p=1.0),
        # A.ShiftScaleRotate(shift_limit=0, scale_limit=0.5, rotate_limit=0, p=0.5),
        # A.RandomScale(scale_limit=0.5, p=1.0),
        # A.Cutout(num_holes=8, max_h_size=int(0.1*img_size[0]), max_w_size=int(0.1*img_size[1]), fill_value=mean, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        ], p=0.5),
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=0.5),
        # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    ],
        p=1.0,
        bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0,  # drop bbox that smaller than 0 pixels
        min_visibility=0,
        label_fields=['labels']
    ))

    image_tfl = []
    if mode == 'noaug':
        image_tfl = [
            ResizePad(
                target_size=img_size, interpolation=interpolation, fill_color=fill_color),
            ImageToNumpy(),
            HWC2CHW()
        ]
    elif mode == 'aug':
        image_tfl = [
            RandomResizePad(
                target_size=img_size, scale=(0.5, 2.0), interpolation=interpolation, fill_color=fill_color),
            RandomFlip(horizontal=True, prob=0.5),
            ImageToNumpy(),
            HWC2CHW()
        ]
    elif mode == 'moreaug':
        image_tfl = [
            RandomResizePad(
                target_size=img_size, scale=(0.5, 2.0), interpolation=interpolation, fill_color=fill_color),
            ImageToNumpy(),
            Albu(train_transforms),
            HWC2CHW()
        ]
    # you can define your own augmentation mode, with implementation of bbox cap or cap using objects images.
    # elif mode == 'your own augmentation mode':
    #     image_tfl = [
    #         RandomResizePad(
    #             target_size=img_size, scale=(0.5, 2.0), interpolation=interpolation, fill_color=fill_color),
    #         # RandomFlip(horizontal=True, prob=0.5),
    #         ImageToNumpy(),
    #         Albu(train_transforms),
    #         # copy and paste augmentation using objects pictures
    #         cap_aug(n_objects_range=[3, 3], glob_split='_', p=0.5,
    #                 retry_iters=20, min_inter_area=100, glob_suffix='*.png'),
    #         # bbox copy and paste augmentation
    #         bboxCapAug(thresh=32*32, prob=0.8, copy_times=3, epochs=30, all_objects=False, one_object=False),
    #         HWC2CHW()
    #     ]
    else:  # source: like aug mode
        image_tfl = [
            RandomFlip(horizontal=True, prob=0.5),
            RandomResizePad(
                target_size=img_size, interpolation=interpolation, fill_color=fill_color),
            ImageToNumpy(),
            HWC2CHW()
        ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf
