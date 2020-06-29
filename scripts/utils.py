import os


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