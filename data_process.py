import os 
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz

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

    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
    )
    return img,lbl,lbl_viz
def find_target_file(find_dir,format_name):
    files= [find_dir+file for file in os.listdir(find_dir) if file.endswith(format_name)]
    return files



files_list=[]
for i in range(12):
    find_dir = '/home/liuxinxin/marine_data/'+ str(i+1) + '/images/'
    files = find_target_file(find_dir,'.json')
    files_list+=files


for json_file in files_list:
    img,lbl,lbl_viz = json2data(json_file)
    print(img.shape)
    plt.imshow(lbl_viz)
    plt.show()


