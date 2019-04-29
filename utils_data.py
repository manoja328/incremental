import os
import scipy.io
import sys
SPLIT_NUM=6
DIR_SPLIT_NUM=-2

def parse_devkit_meta_alphabetical(nb_cl, images_path):
    """This is used to load the data in the same way as pytorch's ImageFolder dataset"""
    def find_classes(dir):
        """Taken from torchvision.datasets.folder"""
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx



    """This loads the data in the same way done by pytorch's ImageFolder dataset i.e. by sorting the directory names"""
    label_names_sorted, full_labels_dic = find_classes(images_path)
    return label_names_sorted



label_names_sorted =parse_devkit_meta_alphabetical(nb_cl=1000, images_path='data/ILSVRC2012/train/')

with open('data/icarl_sorted.txt','w') as f:
    f.writelines('\n'.join(label_names_sorted))
