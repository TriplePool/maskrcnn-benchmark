from PIL import Image, ImageDraw, ImageFont
import torch
import os
import random
from maskrcnn_benchmark.structures.target_list import TargetList


def read_img_list(list_file_path):
    with open(list_file_path) as f:
        res = []
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            file_path, class_id, domain_id = line.split(',')
            class_id = int(class_id)
            domain_id = int(domain_id)
            res.append((file_path, class_id, domain_id))
        return res


def get_all_faces_from_deep_fake_detection_dir(dir_path):
    file_lists = []
    videos = os.listdir(dir_path)
    img_num = 0
    for v in videos:
        v_path = os.path.join(dir_path, v)
        img_fns = os.listdir(v_path)
        img_paths = []
        for img_fn in img_fns:
            if img_fn.endswith('.png'):
                img_paths.append(os.path.join(v_path, img_fn))
        file_lists.append(img_paths)
        img_num += len(img_paths)
    return file_lists, img_num


def write_lists_to_file(file_lists, output_path, class_id=1, domain_id=0):
    out_str = ''
    for file_list in file_lists:
        for file in file_list:
            out_str += '{},{},{}\n'.format(file, class_id, domain_id)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(out_str)


def split_data_with_domain(datalist):
    res = {}
    for d in datalist:
        img_path, class_id, domain_id = d
        if domain_id not in res:
            res[domain_id] = [(img_path, class_id)]
        else:
            res[domain_id].append((img_path, class_id))
    return res


class FaceForensicsDataset(object):
    def __init__(self, list_file_path, transforms, split_with_domain=False):
        self.data_list = read_img_list(list_file_path)
        self.transforms = transforms
        self.split_with_domain = split_with_domain
        if self.split_with_domain:
            self.data_dict = split_data_with_domain(self.data_list)

    def __len__(self):
        if not self.split_with_domain:
            return len(self.data_list)
        else:
            return max([len(x) for x in self.data_dict.values()])

    def __getitem__(self, item):
        if not self.split_with_domain:
            img_path, class_id, domain_id = self.data_list[item]
            img = Image.open(img_path).convert("RGB")
            target = TargetList([class_id], [domain_id])
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target, item
        else:
            imgs = []
            targets = []
            for k in list(sorted(self.data_dict.keys())):
                img_path, class_id = self.data_dict[k][item % len(self.data_dict[k])]
                img = Image.open(img_path).convert("RGB")
                target = TargetList([class_id], [k])
                if self.transforms is not None:
                    img, target = self.transforms(img, target)

                imgs.append(img), targets.append(target), item
            return imgs, targets, item


if __name__ == '__main__':
    # videos_dir = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/original_sequences/actors/c23/faces/"
    # file_lists, img_num = get_all_faces_from_deep_fake_detection_dir(videos_dir)
    # print(img_num)
    # test_ratio = 0.1
    # train_num = (1 - test_ratio) * img_num
    # num_flag = 0
    # i = 0
    # while num_flag < train_num:
    #     num_flag += len(file_lists[i])
    #     i += 1
    #
    # train_index = i - 1
    # train_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/real_train.txt"
    # test_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/real_test.txt"
    #
    # write_lists_to_file(file_lists[:train_index + 1], train_path, 0, 0)
    # write_lists_to_file(file_lists[train_index + 1:], test_path, 0, 0)

    fake_train_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/fake_train.txt"
    fake_test_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/fake_test.txt"
    real_train_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/real_train.txt"
    real_test_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/real_test.txt"

    with open(fake_train_path) as f:
        fake_train_lines = f.readlines()
    with open(fake_test_path) as f:
        fake_test_lines = f.readlines()
    with open(real_train_path) as f:
        real_train_lines = f.readlines()
    with open(real_test_path) as f:
        real_test_lines = f.readlines()

    random.shuffle(fake_train_lines)
    random.shuffle(fake_test_lines)
    random.shuffle(real_train_lines)
    random.shuffle(real_test_lines)

    print('train-real: {}, train-fake: {}, test-real: {}, test-fake: {}'.format(len(real_train_lines),
                                                                                len(fake_train_lines),
                                                                                len(real_test_lines),
                                                                                len(fake_test_lines)))

    train_list = real_train_lines + fake_train_lines
    test_list = real_test_lines + fake_test_lines

    random.shuffle(train_list)
    random.shuffle(test_list)

    print('train: {}, test: {}'.format(len(train_list), len(test_list)))

    train_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/train.txt"
    test_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/test.txt"
    with open(train_path, 'w') as f:
        f.writelines(train_list)
    with open(test_path, 'w') as f:
        f.writelines(test_list)
