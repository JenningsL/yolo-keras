import os
import sys
from random import shuffle
import shutil

src = sys.argv[1]
dest = sys.argv[2]
ratio = 0.2

img_src = os.path.join(src, 'image_2')
label_src = os.path.join(src, 'label_2')
valid_img_dest = os.path.join(dest, 'valid', 'image')
valid_label_dest = os.path.join(dest, 'valid', 'label')
train_img_dest = os.path.join(dest, 'train', 'image')
train_label_dest = os.path.join(dest, 'train', 'label')
os.makedirs(valid_img_dest)
os.makedirs(valid_label_dest)
os.makedirs(train_img_dest)
os.makedirs(train_label_dest)
keys = os.listdir(os.path.join(src, 'image_2'))
keys = map(lambda f: f.split('.')[0], keys)
keys = filter(lambda k: k != '', keys)
shuffle(keys)

for i in range(len(keys)):
    if i < ratio * len(keys):
        shutil.copy(os.path.join(img_src, keys[i] + '.png'), valid_img_dest)
        shutil.copy(os.path.join(label_src, keys[i] + '.txt'), valid_label_dest)
    else:
        shutil.copy(os.path.join(img_src, keys[i] + '.png'), train_img_dest)
        shutil.copy(os.path.join(label_src, keys[i] + '.txt'), train_label_dest)
