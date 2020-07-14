import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import LeakyReLU
import os
import random
import time
import wget
import tarfile
import numpy as np
import cv2
from PIL import Image
import os 
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz

class RTNet:
    def __init__(self, train_stage=1,use_cpu=False, print_summary=False):
        self.train_stage = 1
        self.parameter = [24,48,64,96,128,196]
        self.build(use_cpu=use_cpu, print_summary=print_summary)
        self.trainset  = self.read_traindata_names()
        self.num_train = len(self.trainset)

    def read_traindata_names(self,):
        trainset=[]
        for i in range(12):
            find_dir = 'marine_data/'+ str(i+1) + '/images/'
            files = self.find_target_file(find_dir,'.json')
            trainset+=files
        return trainset
    def json2data(self, json_file):
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
    def find_target_file(self,find_dir,format_name):
        files= [find_dir+file for file in os.listdir(find_dir) if file.endswith(format_name)]
        return files

    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='model.h5'):
        self.model.load_weights(file_path)
    
    #(0=Unlabeled, 1=sea, 2=sky, 3=boat)
    def random_crop_or_pad(self, image, truth, size=(448, 512)):
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
    def BatchGenerator(self,batch_size=8, image_size=(448, 512, 3), labels=3):#500, 375
        while True:
            images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
            truths = np.zeros((batch_size, image_size[0], image_size[1], labels))
            for i in range(batch_size):
                random_line = random.choice(self.trainset)
                image,truth_mask,lbl_viz = self.json2data(random_line)
                truth_mask=truth_mask+1
                image, truth = self.random_crop_or_pad(image, truth_mask, image_size)
                images[i] = image/255
                truths[i] = (np.arange(labels) == truth[...,None]-1).astype(int) # encode to one-hot-vector
            yield images, truths
            
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        batch_generator = self.BatchGenerator(batch_size=batch_size)
        self.model.fit_generator(batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def build_conv2D_block(self, inputs, filters, kernel_size, strides, block, i):
        conv2d = Conv2D(filters = filters, kernel_size=kernel_size,strides=strides, padding='same', \
        name='conv{}-{}'.format(block, i), use_bias=True,bias_initializer='zeros')(inputs)
        conv2d = BatchNormalization(name='batchnorm{}-{}'.format(block, i))(conv2d)
        conv2d_output = Activation(LeakyReLU(alpha=0.1), name='relu{}-{}'.format(block, i))(conv2d)
        return conv2d_output

    def build_conv2Dtranspose_block(self, inputs, filters, kernel_size, strides, block, i):
        conv2d = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=True,bias_initializer='zeros', padding='same', \
            name='deconv4{}-{}'.format(block, i))(inputs)
        conv2d = BatchNormalization(name='batchnorm_decon{}-{}'.format(block, i))(conv2d)
        conv2d_deconv = Activation(LeakyReLU(alpha=0.1), name='relu_decon{}-{}'.format(block, i))(conv2d)
        return conv2d_deconv

    def build(self, use_cpu=False, print_summary=False):
        inputs = Input(shape=(448, 512, 3))
            
        # initial layer
        conv2d_conv0_1 = self.build_conv2D_block(inputs,        filters = self.parameter[0],kernel_size=1,strides=1, block=0, i=0)
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0_1,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=1)
        ###########
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=2)
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=3)
        conv2d_conv0   = self.build_conv2D_block(conv2d_conv0,filters = self.parameter[0],kernel_size=3,strides=1, block=0, i=4)
        # first conv layer
        conv2d_conv1_1 = self.build_conv2D_block(conv2d_conv0,  filters = self.parameter[1],kernel_size=3,strides=2, block=1, i=0)
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1_1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=1)
        ###########
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=2)
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=3)
        conv2d_conv1   = self.build_conv2D_block(conv2d_conv1,filters = self.parameter[1],kernel_size=3,strides=1, block=1, i=4)
        # second conv layer
        conv2d_conv2_2 = self.build_conv2D_block(conv2d_conv1,  filters = self.parameter[2],kernel_size=3,strides=2, block=2, i=0)
        conv2d_conv2_1 = self.build_conv2D_block(conv2d_conv2_2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=1)
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2_1,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=2)
        ###########
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=3)
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=4)
        conv2d_conv2   = self.build_conv2D_block(conv2d_conv2,filters = self.parameter[2],kernel_size=3,strides=1, block=2, i=5)
        # third conv layer
        conv2d_conv3_2 = self.build_conv2D_block(conv2d_conv2,  filters = self.parameter[3],kernel_size=3,strides=2, block=3, i=0)
        conv2d_conv3_1 = self.build_conv2D_block(conv2d_conv3_2,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=1)
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3_1,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=2)
        ###########
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=3)
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=4)
        conv2d_conv3   = self.build_conv2D_block(conv2d_conv3,filters = self.parameter[3],kernel_size=3,strides=1, block=3, i=5)
        # fourth conv layer
        conv2d_conv4_2 = self.build_conv2D_block(conv2d_conv3,  filters = self.parameter[4],kernel_size=3,strides=2, block=4, i=0)
        conv2d_conv4_1 = self.build_conv2D_block(conv2d_conv4_2,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=1)
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4_1,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=2)
        ###########
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=3)
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=4)
        conv2d_conv4   = self.build_conv2D_block(conv2d_conv4,filters = self.parameter[4],kernel_size=3,strides=1, block=4, i=5)
        # fifth conv layer
        conv2d_conv5_1 = self.build_conv2D_block(conv2d_conv4,  filters = self.parameter[5],kernel_size=3,strides=2, block=5, i=0)
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5_1,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=1)
        ###########
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=2)
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=3)
        conv2d_conv5   = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=5, i=4)
        # fifth deconv layer
        conv2d_deconv5_1 = self.build_conv2D_block(conv2d_conv5,filters = self.parameter[5],kernel_size=3,strides=1, block=55, i=0)
        conv2d_deconv4   = self.build_conv2Dtranspose_block(conv2d_deconv5_1, filters=self.parameter[4], kernel_size=4, strides=2, block=55, i=1)
            
        #Concat4
        Concat_concat4 = concatenate([conv2d_conv4, conv2d_deconv4] , axis=-1)
            
        # fourth deconv layer
        conv2d_deconv4_1 = self.build_conv2D_block(Concat_concat4,filters = self.parameter[4],kernel_size=3,strides=1, block=44, i=0)
        conv2d_deconv3   = self.build_conv2Dtranspose_block(conv2d_deconv4_1, filters=self.parameter[3], kernel_size=4, strides=2, block=44, i=1)
            
        #Concat3
        Concat_concat3 = concatenate([conv2d_conv3 , conv2d_deconv3] , axis=-1)
            
        # third deconv layer
        conv2d_deconv3_1 = self.build_conv2D_block(Concat_concat3,filters = self.parameter[3],kernel_size=3,strides=1, block=33, i=0)
        conv2d_deconv2   = self.build_conv2Dtranspose_block(conv2d_deconv3_1, filters=self.parameter[2], kernel_size=4, strides=2, block=33, i=1)
            
        #Concat2
        Concat_concat2 = concatenate([conv2d_conv2 , conv2d_deconv2] , axis=-1)
            
        # sencod deconv layer
        conv2d_deconv2_1 = self.build_conv2D_block(Concat_concat2,filters = self.parameter[2],kernel_size=3,strides=1, block=22, i=0)
        conv2d_deconv1   = self.build_conv2Dtranspose_block(conv2d_deconv2_1, filters=self.parameter[1], kernel_size=4, strides=2, block=22, i=1)
            
        #Concat1
        Concat_concat1 = concatenate([conv2d_conv1 , conv2d_deconv1] , axis=-1)
            
        # first deconv layer
        conv2d_deconv1_1 = self.build_conv2D_block(Concat_concat1,filters = self.parameter[1],kernel_size=3,strides=1, block=11, i=0)
        conv2d_deconv0   = self.build_conv2Dtranspose_block(conv2d_deconv1_1, filters=self.parameter[0], kernel_size=4, strides=2, block=11, i=1)


        output = Conv2DTranspose(filters=3, kernel_size=1, strides=1, activation='softmax', padding='same', name='output')(conv2d_deconv0)
            
        self.model = Model(inputs=inputs, outputs=output)
 
        # ~ parallel_model = multi_gpu_model(self.model, gpus=1)
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy', 'mse'])
