#coding:utf8

import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import operator


def is_image_file(filename):  # 
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".tif",".tiff"])

def is_image_damaged(cvimage):
    white_pixel_count = 0
    height,weight,channel = cvimage.shape
    # for row in list(range(0,height)):
    #     for col in list(range(0,weight)):
    #         if cvimage[row][col][2] == 255:
    #             white_pixel_count += 1
    #             if white_pixel_count > 0.2*height*weight:
    #                 return True
    # return False
    one_channel = np.sum(cvimage, axis=2)
    white_pixel_count = len(one_channel[one_channel==255*3])   #Count the number of white pixels
    if white_pixel_count > 0.08*height*weight:
        return True
    return False

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    
# def rotate(xb,yb,angle):
#     M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
#     xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
#     yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
#     return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
def data_augment(xb,yb):
    # if np.random.random() < 0.25:
    #     xb,yb = rotate(xb,yb,90)
    # if np.random.random() < 0.25:
    #     xb,yb = rotate(xb,yb,180)
    # if np.random.random() < 0.25:
    #     xb,yb = rotate(xb,yb,270)
    # if np.random.random() < 0.25:
    #     xb = cv2.flip(xb, 1)  # flipcode > 0：Flip along the y axis
    #     yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

