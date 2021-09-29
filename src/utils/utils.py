import os
import numpy as np
import torch


def process_txt(txt_file, datapath):
    
    with open(txt_file, "r") as f:
        data = [x.strip("\n") for x in f.readlines()]

    img_paths = []
    img_labels = []

    for x in data:
        img_path = x.split(" ")[0]
        img_paths.append(os.path.join(datapath, img_path))
        img_labels.append(int(x.split(" ")[1]))

    return {'img_paths':np.array(img_paths), 'labels':np.array(img_labels)}


def numeric_classes(tags_classes, dict_tags):
    num_classes = np.array([dict_tags.get(t) for t in tags_classes])
    return num_classes


def create_dict_texts(texts):
    d = {l: i for i, l in enumerate(texts)}
    return d


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, directory, save_name, last_chkpt):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    checkpoint_file = os.path.join(directory, save_name+'.pth')
    torch.save(state, checkpoint_file)
    last_chkpt_file = os.path.join(directory, last_chkpt+'.pth')
    
    if os.path.isfile(last_chkpt_file):
        os.remove(last_chkpt_file)
    else:
        print("Error: {} file not found".format(last_chkpt_file))