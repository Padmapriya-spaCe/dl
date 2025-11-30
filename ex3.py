# EX3- Different Types of Data Augmentation Techniques 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

img=load_img("bird.png",target_size=(224,224))
x=img_to_array(img)
samples=np.expand_dims(x,axis=0)

#brightness range
datagen=ImageDataGenerator(brightness_range=[0.2,0.4])
it=datagen.flow(samples,batch_size=1)
for i in range(9):
    plt.subplot(330+1+i)
    batch=next(it)
    image=batch[0].astype("uint8")
    plt.imshow(image)
plt.show()
print("-----------------------------------------------------")

#vertical shift
datagen=ImageDataGenerator(height_shift_range=0.5)
it=datagen.flow(samples,batch_size=1)
for i in range(9):
    plt.subplot(330+1+i)
    batch=next(it)
    image=batch[0].astype("uint8")
    plt.imshow(image)
plt.show()
print("-----------------------------------------------------")

#horizontal shift
datagen=ImageDataGenerator(width_shift_range=0.5)
it=datagen.flow(samples,batch_size=1)
for i in range(9):
    plt.subplot(330+1+i)
    batch=next(it)
    image=batch[0].astype("uint8")
    plt.imshow(image)
plt.show()
print("-----------------------------------------------------")

#horizontal flip
datagen=ImageDataGenerator(horizontal_flip=True)
it=datagen.flow(samples,batch_size=1)
for i in range(9):
    plt.subplot(330+1+i)
    batch=next(it)
    image=batch[0].astype("uint8")
    plt.imshow(image)
plt.show()
print("-----------------------------------------------------")

#rotation
datagen=ImageDataGenerator(rotation_range=60)
it=datagen.flow(samples,batch_size=1)
for i in range(9):
    plt.subplot(330+1+i)
    batch=next(it)
    image=batch[0].astype("uint8")
    plt.imshow(image)
plt.show()
print("-----------------------------------------------------")

#zoom
datagen=ImageDataGenerator(zoom_range=[0.2,0.1])
it=datagen.flow(samples,batch_size=1)
for i in range(9):
    plt.subplot(330+1+i)
    batch=next(it)
    image=batch[0].astype("uint8")
    plt.imshow(image)
plt.show()
