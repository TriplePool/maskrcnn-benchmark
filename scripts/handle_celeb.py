import random
import os

data_dir = "/home-hd-1/gpu/Workspace/Datas/dataset/Celeb-DF-v2/"
test_file_list_path = os.path.join(data_dir, 'List_of_testing_videos.txt')
domain_label = 1
with open(test_file_list_path) as f:
    lines = f.readlines()
    test_fn = {}
    for line in lines:
        label, v_path = line.split(' ')
        v_path = v_path[:-1]
        fn = os.path.basename(v_path).split('.')[0]
        dirn = os.path.dirname(v_path)
        if dirn not in test_fn:
            test_fn[dirn] = [fn]
        else:
            test_fn[dirn].append(fn)
img_lists = {}
for video_dir in test_fn.keys():
    test_list = test_fn[video_dir]
    train_img_path_list = []
    test_img_path_list = []
    faces_dir = os.path.join(data_dir, video_dir + '_faces')
    for img_dir in os.listdir(faces_dir):
        if img_dir in test_list:
            img_path = os.path.join(faces_dir, img_dir)
            for img_fn in os.listdir(img_path):
                test_img_path_list.append(os.path.join(img_path, img_fn))
        else:
            img_path = os.path.join(faces_dir, img_dir)
            for img_fn in os.listdir(img_path):
                train_img_path_list.append(os.path.join(img_path, img_fn))
    img_lists[video_dir] = {'train': train_img_path_list, 'test': test_img_path_list}

train_str_list = []
test_str_list = []

for k in img_lists:
    label = 1
    if 'real' in k:
        label = 0
    for ke in img_lists[k]:
        print('{}_{}: {}'.format(k, ke, len(img_lists[k][ke])))
        output_path = os.path.join(data_dir, '{}_{}.txt'.format(k, ke))
        output_str = ''
        for p in img_lists[k][ke]:
            tmp_str = '{},{},{}\n'.format(p, label, domain_label)
            output_str += tmp_str
            if ke == 'train':
                train_str_list.append(tmp_str)
            else:
                test_str_list.append(tmp_str)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_str)
train_file_path = os.path.join(data_dir, 'train.txt')
test_file_path = os.path.join(data_dir, 'test.txt')
print('train: {}'.format(len(train_str_list)))
print('test: {}'.format(len(test_str_list)))
with open(train_file_path, 'w', encoding='utf-8') as f:
    f.writelines(train_str_list)
with open(test_file_path, 'w', encoding='utf-8') as f:
    f.writelines(test_str_list)
