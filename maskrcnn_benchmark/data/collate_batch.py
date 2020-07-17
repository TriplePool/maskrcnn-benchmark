# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids


class BatchCollatorMutiDomain(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        img_lists = list(zip(*(transposed_batch[0])))
        # img_lists = transposed_batch[0]
        target_lists = list(zip(*(transposed_batch[1])))
        # target_lists = transposed_batch[1]

        imgs = []
        tgts = []
        for img_l in img_lists:
            imgs += img_l
            # imgs.append(img_l[0])
        for tgt in target_lists:
            tgts += tgt
            # tgts.append(tgt[0])
        images = to_image_list(imgs, self.size_divisible)
        targets = tgts
        img_ids = transposed_batch[2]
        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
