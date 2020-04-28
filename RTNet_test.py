from RTNet import RTNet
import numpy as np
from tensorflow.keras.models import load_model, save_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import data

rtnet = RTNet()
rtnet.train(epochs=5, steps_per_epoch=50, batch_size=2)
rtnet.save()


image,label = data.get_new_data(4,1,20)
image = data.distance_transform(image)/100
rtnet.load()
# plt.figure()
# plt.imshow(image)
prediction = rtnet.predict(image)
print(prediction.shape,np.max(label)*360)
result = prediction[0,:,:,0]*(image[:,:,0]>0.95)
plt.figure()
plt.imshow(result*360)
plt.show()
