# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

from shapely.geometry import box, Polygon
from shapely import affinity
from PIL import Image
import cv2
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size, for_classifier=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.for_classifier = for_classifier

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        if self.for_classifier:
            size = random.choice(self.min_size)
            return (size, size)
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is not None:
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomCropExpand(object):
    """Random crop with repeatedly expanding the range to included box borders."""

    def __init__(self, prob, init_crop_size=(0.5, 1.0)):

        if (not isinstance(init_crop_size, list)) and (not isinstance(init_crop_size, tuple)):
            raise ValueError('Paremeter init_crop_size should be a list or tuple!')
        elif len(init_crop_size) != 2:
            raise ValueError('Length of init_crop_size should be 2!')
        elif not (init_crop_size[0] <= 1 and init_crop_size[0] >= 0 and init_crop_size[1] <= 1 and init_crop_size[
            1] >= 0):
            raise ValueError('Elements of init_crop_size should be within [0, 1]!')
        self.prob = prob
        self.init_crop_size = init_crop_size

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        if isinstance(target, list):
            target0 = target[0]
        else:
            target0 = target
        while True:
            # Initial Crop Region
            crop_region = self.initial_crop_region(image)

            # Adjust Crop Region
            crop_region, keep_target = self.adjust_crop_region(crop_region, target0)
            if crop_region is None and keep_target is None:
                continue

            if isinstance(target, list):
                # check empty char
                new_t1 = target[1].crop(crop_region)
                if len(new_t1) < 1: return image, target

            image = image.crop(crop_region)
            if isinstance(target, list):
                target0 = keep_target.crop(crop_region)
                others = [t.crop(crop_region, remove_empty=True) for t in target[1:]]
                target = [target0] + others
            else:
                target = keep_target.crop(crop_region)
            return image, target

    def initial_crop_region(self, image):
        width, height = image.size
        ratio_w, ratio_h = torch.empty(2).uniform_(self.init_crop_size[0], self.init_crop_size[1])
        crop_width, crop_height = int(width * ratio_w), int(height * ratio_h)
        crop_xmin = torch.randint(width - crop_width, (1,))
        crop_ymin = torch.randint(height - crop_height, (1,))
        crop_xmax = crop_xmin + crop_width
        crop_ymax = crop_ymin + crop_height
        crop_region = torch.Tensor([crop_xmin, crop_ymin, crop_xmax, crop_ymax])
        return crop_region

    def intersect_area(self, bbox, bboxes):
        inter_xmin = torch.max(bbox[0], bboxes[:, 0])
        inter_ymin = torch.max(bbox[1], bboxes[:, 1])
        inter_xmax = torch.min(bbox[2], bboxes[:, 2])
        inter_ymax = torch.min(bbox[3], bboxes[:, 3])
        inter_width = torch.max(torch.Tensor([0]), inter_xmax - inter_xmin)
        inter_height = torch.max(torch.Tensor([0]), inter_ymax - inter_ymin)
        return inter_width * inter_height

    def adjust_crop_region(self, crop_region, target):
        keep_indies_ = torch.zeros((len(target)), dtype=torch.bool)
        while True:
            inter_area = self.intersect_area(crop_region, target.bbox)
            keep_indies = (inter_area > 0)
            if torch.sum(keep_indies) == 0:
                return None, None
            keep_target = target[keep_indies]
            if keep_indies.equal(keep_indies_):
                return crop_region.int().numpy().tolist(), keep_target
            keep_bbox = keep_target.bbox
            crop_xmin = torch.min(crop_region[0], torch.min(keep_bbox[:, 0]))
            crop_ymin = torch.min(crop_region[1], torch.min(keep_bbox[:, 1]))
            crop_xmax = torch.max(crop_region[2], torch.max(keep_bbox[:, 2]))
            crop_ymax = torch.max(crop_region[3], torch.max(keep_bbox[:, 3]))
            crop_region = torch.Tensor([crop_xmin, crop_ymin, crop_xmax, crop_ymax])
            keep_indies_ = keep_indies


class RandomRotate(object):
    def __init__(self, prob, max_theta=30):
        self.prob = prob
        self.max_theta = max_theta

    def __call__(self, image, target):
        if random.random() < self.prob and target is not None:
            # try:
            delta = random.uniform(-1 * self.max_theta, self.max_theta)
            width, height = image.size
            ## get the minimal rect to cover the rotated image
            img_box = [[[0, 0], [width, 0], [width, height], [0, height]]]
            rotated_img_box = _quad2minrect(_rotate_polygons(img_box, delta, (width / 2, height / 2)))
            r_height = int(
                max(rotated_img_box[0][3], rotated_img_box[0][1]) - min(rotated_img_box[0][3], rotated_img_box[0][1]))
            r_width = int(
                max(rotated_img_box[0][2], rotated_img_box[0][0]) - min(rotated_img_box[0][2], rotated_img_box[0][0]))
            r_height = max(r_height, height + 1)
            r_width = max(r_width, width + 1)

            ## padding im
            im_padding = np.zeros((r_height, r_width, 3))
            start_h, start_w = int((r_height - height) / 2.0), int((r_width - width) / 2.0)
            end_h, end_w = start_h + height, start_w + width
            im_padding[start_h:end_h, start_w:end_w, :] = image

            M = cv2.getRotationMatrix2D((r_width / 2, r_height / 2), delta, 1)
            im = cv2.warpAffine(im_padding, M, (r_width, r_height))
            im = Image.fromarray(im.astype(np.uint8))
            target = target.rotate(-delta, (r_width / 2, r_height / 2), start_h, start_w)
            return im, target
            # except:
            #    return image, target
        else:
            return image, target


def _quad2minrect(boxes):
    ## trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
    return np.hstack((boxes[:, ::2].min(axis=1).reshape((-1, 1)), boxes[:, 1::2].min(axis=1).reshape((-1, 1)),
                      boxes[:, ::2].max(axis=1).reshape((-1, 1)), boxes[:, 1::2].max(axis=1).reshape((-1, 1))))


def _boxlist2quads(boxlist):
    res = np.zeros((len(boxlist), 8))
    for i, box in enumerate(boxlist):
        # print(box)
        res[i] = np.array([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]])
    return res


def _rotate_polygons(polygons, angle, r_c):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15

    rotate_boxes_list = []
    for poly in polygons:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords)) < 5:
            print('img_box_ori:', poly)
            print('img_box_rotated:', rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = _boxlist2quads(rotate_boxes_list)
    return res
