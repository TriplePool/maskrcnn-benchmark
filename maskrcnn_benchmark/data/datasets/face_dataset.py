from PIL import Image, ImageDraw, ImageFont
import torch
import os

from maskrcnn_benchmark.structures.target_list import TargetList


def read_img_list(list_file_path):
    with open(list_file_path) as f:
        res = []
        lines = f.readlines(list_file_path)
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


class FaceForensicsDataset(object):
    def __init__(self, list_file_path, transforms):
        self.data_list = read_img_list(list_file_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_path, class_id, domain_id = self.data_list[item]
        img = Image.open(img_path).convert("RGB")
        target = TargetList([class_id], [domain_id])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, item


if __name__ == '__main__':
    videos_dir = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/manipulated_sequences/DeepFakeDetection/c23/faces/"
    file_lists, img_num = get_all_faces_from_deep_fake_detection_dir(videos_dir)
    print(img_num)
    test_ratio = 0.1
    train_num = (1 - test_ratio) * img_num
    num_flag = 0
    i = 0
    while num_flag < train_num:
        num_flag += len(file_lists[i])
        i += 1

    train_index = i - 1
    train_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/fake_train.txt"
    test_path = "/home-hd-1/gpu/Workspace/Datas/dataset/ds/fake_test.txt"

    write_lists_to_file(file_lists[:train_index + 1], train_path, 1, 0)
    write_lists_to_file(file_lists[train_index + 1:], test_path, 1, 0)
