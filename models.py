'''
#EX1- Drawing Confusion Matrix and Computation of Different Metrics for Classification 
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

confusion_matrix={
    "Glioma":{"Glioma":120,"Meningioma":10,"Pituitary":12,"No Tumor":2},
    "Meningioma":{"Glioma":10,"Meningioma":130,"Pituitary":1,"No Tumor":0},
    "Pituitary":{"Glioma":1,"Meningioma":0,"Pituitary":140,"No Tumor":2},
    "No Tumor":{"Glioma":12,"Meningioma":10,"Pituitary":2,"No Tumor":200},
}
#display confusion matrix and other metrics
cm=ConfusionMatrix(matrix=confusion_matrix)
print(cm)

#plot heatmap of confusion matrix
# cm.plot(cmap="Blues")
# import matplotlib.pyplot as plt
# plt.show()

df = pd.DataFrame(confusion_matrix).T  # Transpose for correct orientation
plt.figure(figsize=(25, 25))
sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.ylabel("Predicted Disease")
plt.xlabel("Actual Disease")
plt.show()
'''
'''

#EX2- Layer Visualization and Feature maps in CNN

#install graphviz and pydot to visualize the model architecture
import tensorflow as tf 
from tensorflow import keras 
from keras.applications.vgg16 import VGG16 
from keras.utils import plot_model 

model = VGG16() 
plot_model(model, to_file='vgg16_plot.png', show_shapes=True, show_layer_names=True) 

# summarize convolutional filter shapes 
print(f"Total layers: {len(model.layers)}") 
for layer in model.layers: 
    if 'conv' not in layer.name: 
        continue 
    weights = layer.get_weights() 
    if len(weights) == 1: 
        filters = weights[0] 
        print(f"{layer.name}: filters={filters.shape}, no bias") 
    elif len(weights) == 2: 
        filters, biases = weights 
        print(f"{layer.name}: filters={filters.shape}, biases={biases.shape}") 
    else: 
        print(f"{layer.name}: unexpected weights format") 



from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

model = VGG16()
# model=Model(inputs=model.inputs,outputs=model.layers[1].output)
ixs=[1,3,5,6]
output=[model.layers[i].output for i in ixs]
model=Model(inputs=model.inputs,outputs=output)
img=load_img("bird.png",target_size=(224,224))
x=img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

features=model.predict(x)

# square=8
# ix=1
# for i in range(square):
#     for j in range(square):
#         ax=plt.subplot(square,square,ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(fm[0,:,:,ix-1],cmap="grey")
#         ix+=1
# plt.show()
square=8
t=1
for fm in features:
    print(f"Layers:{t}")
    ix=1
    for i in range(square):
        for j in range(square):
            ax=plt.subplot(square,square,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(fm[0,:,:,ix-1],cmap="grey")
            ix+=1
    t+=1
    plt.show()

'''
'''
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
'''


#EX4- Image Classification using Pre-trained CNN Models 
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions,VGG16
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np

model = VGG16(weights="imagenet")
img=load_img("bird.png",target_size=(224,224))
x=img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

preds=model.predict(x)
preds=decode_predictions(preds,top=3)[0]

for pred in preds:
    print(pred)
