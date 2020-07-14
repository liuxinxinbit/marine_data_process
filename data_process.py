import os 
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
import numpy as np
import random


def json2data(json_file):
    data = json.load(open(json_file))
    imageData = data.get('imageData')
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data['shapes'], label_name_to_value
    )
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    print(label_names)
    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
    )
    return img,lbl,lbl_viz
def find_target_file(find_dir,format_name):
    files= [find_dir+file for file in os.listdir(find_dir) if file.endswith(format_name)]
    return files
def read_traindata_names():
    trainset=[]
    for i in range(12):
        find_dir = 'marine_data/'+ str(i+1) + '/images/'
        files = find_target_file(find_dir,'.json')
        trainset+=files
    return trainset
def random_crop_or_pad( image, truth, size=(480, 640)):
    assert image.shape[:2] == truth.shape[:2]

    if image.shape[0] > size[0]:
        crop_random_y = random.randint(0, image.shape[0] - size[0])
        image = image[crop_random_y:crop_random_y + size[0],:,:]
        truth = truth[crop_random_y:crop_random_y + size[0],:]
    else:
        zeros = np.zeros((size[0], image.shape[1], image.shape[2]), dtype=np.float32)
        zeros[:image.shape[0], :image.shape[1], :] = image                                          
        image = np.copy(zeros)
        zeros = np.zeros((size[0], truth.shape[1]), dtype=np.float32)
        zeros[:truth.shape[0], :truth.shape[1]] = truth
        truth = np.copy(zeros)

    if image.shape[1] > size[1]:
        crop_random_x = random.randint(0, image.shape[1] - size[1])
        image = image[:,crop_random_x:crop_random_x + size[1],:]
        truth = truth[:,crop_random_x:crop_random_x + size[1]]
    else:
        zeros = np.zeros((image.shape[0], size[1], image.shape[2]))
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)
        zeros = np.zeros((truth.shape[0], size[1]))
        zeros[:truth.shape[0], :truth.shape[1]] = truth
        truth = np.copy(zeros)            

    return image, truth

# files_list=[]
# for i in range(12):
#     find_dir = 'marine_data/'+ str(i+1) + '/images/'
#     files = find_target_file(find_dir,'.json')
#     files_list+=files


# for json_file in files_list:
#     img,lbl,lbl_viz = json2data(json_file)
#     print(img.shape)
#     plt.imshow(lbl_viz)
#     plt.show()
batch_size=8
image_size=(448, 512, 3)
labels=3
trainset  = read_traindata_names()
while True:
    images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
    truths = np.zeros((batch_size, image_size[0], image_size[1], labels))
    for i in range(batch_size):
        random_line = random.choice(trainset)
        image,truth_mask,lbl_viz = json2data(random_line)
        truth_mask=truth_mask
        image, truth = random_crop_or_pad(image, truth_mask, image_size)
        images[i] = image/255
        truths[i] = (np.arange(labels) == truth[...,None]-1).astype(int) # encode to one-hot-vector
        print(image.shape,truth.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(truth)
        plt.pause(1)
        plt.clf()