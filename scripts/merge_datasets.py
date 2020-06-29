import random


def merge_file_lists(list_paths, num_list, output_path, shuffle=True):
    assert len(list_paths) == len(num_list)
    res_lines = []
    for i in range(len(list_paths)):
        list_path = list_paths[i]
        with open(list_path, encoding='utf-8') as f:
            tmp_lines = f.readlines()
            tmp_num = num_list[i]
            if shuffle:
                random.shuffle(tmp_lines)
            res_lines.extend(tmp_lines[:tmp_num])

    if shuffle:
        random.shuffle(res_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(res_lines)


if __name__ == '__main__':
    paths = [
        "/home-hd-1/gpu/Workspace/Datas/dataset/ds/train.txt",
        "/home-hd-1/gpu/Workspace/Datas/dataset/Celeb-DF-v2/train.txt"
    ]

    nums = [500, 1500]
    opt_path = "/home-hd-1/gpu/Workspace/Datas/dataset/merged/train_d500_c1500.txt"

    merge_file_lists(paths, nums, opt_path)
