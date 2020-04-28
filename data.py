import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import ndimage

def draw_circle(img,x,y,r):
    for i in range(2*r):
        for j in range(2*r):
            nx = x+i-r
            ny = y+j-r
            if np.sqrt((nx-x)**2+(ny-y)**2)<r:
                img[nx,ny]=1
    return img
def get_new_nosie_data(fram_num,p):
    
    image=[]
    for i in range(fram_num):
        img = np.zeros((512,512))
        target_x = random.randint(-200,200)+256
        target_y = random.randint(-200,200)+256
        noise_p = random.randint(0,100)
        if(noise_p<p):
            img = draw_circle(img,int(target_x),int(target_y),5)
        image.append(img)
    image = np.array(image)
    image = np.transpose(image,(1,2,0))
    return image
def distance_transform(image):
    for i in range(image.shape[2]):
        image[:,:,i] = ndimage.distance_transform_edt(1-image[:,:,i])-ndimage.distance_transform_edt(image[:,:,i])+5
    image[image>100]=100
    image=100-image
    return image
def get_new_true_data(fram_num):
    target_x = random.randint(-200,200)+256
    target_y = random.randint(-200,200)+256
    target_angle = random.randint(0,360)
    target_velocity = random.randint(0,20)*1852/3600
    image=[]
    for i in range(fram_num):
        img = np.zeros((512,512))
        target_x = target_x + np.sin(target_angle/(2*np.pi))*target_velocity
        target_y = target_y + np.cos(target_angle/(2*np.pi))*target_velocity
        img = draw_circle(img,int(target_x),int(target_y),5)
        image.append(img)
    label=np.zeros((512,512,1))
    # label[:,:,0]=image[0]
    # label[:,:,0]=image[0]*target_velocity/51
    label[:,:,0]=image[0]*target_angle/360
    image = np.array(image)
    image = np.transpose(image,(1,2,0))
    return image,label
def get_new_data(fram_num,noise_num,p):
    target_x = random.randint(-200,200)+256
    target_y = random.randint(-200,200)+256
    target_angle = random.randint(0,360)
    target_velocity = random.randint(0,20)*1852/3600
    image=[]
    for i in range(fram_num):
        img = np.zeros((512,512))
        target_x = target_x + np.sin(target_angle/(2*np.pi))*target_velocity
        target_y = target_y + np.cos(target_angle/(2*np.pi))*target_velocity
        img = draw_circle(img,int(target_x),int(target_y),5)
        image.append(img)
    label=np.zeros((512,512,1))
    # label[:,:,0]=image[0]
    # label[:,:,0]=image[0]*target_velocity/51
    label[:,:,0]=image[0]*target_angle/360
    image = np.array(image)
    image = np.transpose(image,(1,2,0))
    image = image + get_new_nosie_data(fram_num,p)
    
    return image,label



# image,label = get_new_data(6,1,20)
# image = distance_transform(image)/1
# plt.figure(figsize=(6,6),dpi=80)
# plt.figure(1)
# ax1 = plt.subplot(321)
# plt.imshow(np.array(image[:,:,0]),'hot')
# ax1 = plt.subplot(322)
# plt.imshow(np.array(image[:,:,1]),'hot')
# ax1 = plt.subplot(323)
# plt.imshow(np.array(image[:,:,2]),'hot')
# ax1 = plt.subplot(324)
# plt.imshow(np.array(image[:,:,3]),'hot')
# ax1 = plt.subplot(325)
# plt.imshow(np.array(image[:,:,4]),'hot')
# ax1 = plt.subplot(326)
# plt.imshow(np.array(image[:,:,5]),'hot')
# plt.show()
    


