from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
def get_annot_list(annotation_path):
    with open(annotation_path) as f:
        lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
    for i in range(len(lines)):
        lines[i] = lines[i].split(' ')
    return lines
train_annotation_path = './SegNet/CamVid/train.txt'
test_annotation_path = './SegNet/CamVid/test.txt'
valid_annotation_path = './SegNet/CamVid/val.txt'

train_annotation_list = get_annot_list(train_annotation_path)
test_annotation_list = get_annot_list(train_annotation_path)
valid_annotation_list = get_annot_list(train_annotation_path)

for line in train_annotation_list:
    img=mpimg.imread('.'+line[1][:-1])
    plt.imshow(img*255,cmap="hot")
    plt.figure()
    img=mpimg.imread('.'+line[0])
    plt.imshow(img,cmap="Greys_r")
    plt.show()

# img=mpimg.imread('/home/liuxinxin/Toolkit/seaRTNet/dataset/newtrain/2008000235.jpg')
# plt.imshow(img,cmap="Greys_r")
# plt.show()