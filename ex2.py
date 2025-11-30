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
#sir code and p
#Ex 1
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input

model=VGG16()

for layer in model.layers:
    if "conv" not in layer.name:
        continue 
    
    weights=layer.get_weights()
    if len(weights)==1:
        filters=weights[0]
        print(f"layer:{layer.name},filters:{filters.shape},no bias")
    if len(weights)==2:
        filters,bias=weights
        print(f"layer:{layer.name},filters:{filters.shape},bias:{bias.shape}")

#Ex 5
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(2,input_dim=1,activation="relu"))
model.add(Dense(1,activation="softmax"))


#Ex 2

import matplotlib.pyplot as plt
model=VGG16()
filters,weights=model.layers[1].get_weights()

f_min,f_max=filters.min(),filters.max()
filters=(filters-f_min)/(f_max-f_min)

n_filters,ix=6,1
for i in range(n_filters):
    f=filters[:,:,:,i]
    for j in range(3):
        ax=plt.subplot(n_filters,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:,:,j],cmap="grey")
        ix+=1

plt.show()

#Ex-3

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model=VGG16()
model=Model(inputs=model.inputs,outputs=model.layers[1].output)
img=load_img(f"C:\sem-7\deep learning\ModelExam\Testing\meningioma\Te-me_0010.jpg",target_size=(224,224))
img=img_to_array(img)
img=np.expand_dims(img,axis=0)
img=preprocess_input(img)

feature=model.predict(img)
ix=1
square=8
for i in range(square):
    for j in range(square):
        ax=plt.subplot(square,square,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature[0,:,:,ix-1],cmap="grey")
        ix+=1

# plt.show()

#Ex 4

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


model=VGG16()
ixs = [2, 5, 9, 13, 17]
outputs=[model.layers[i].output for i in ixs]

model=Model(inputs=model.inputs,outputs=model.layers[1].output)
img=load_img(f"C:\sem-7\deep learning\ModelExam\Testing\meningioma\Te-me_0010.jpg",target_size=(224,224))
img=img_to_array(img)
img=np.expand_dims(img,axis=0)
img=preprocess_input(img)

feature=model.predict(img)


square=8

for fmap in feature:
    ix=1
    for i in range(square):
        for j in range(square):
            ax=plt.subplot(square,square,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature[0,:,:,ix-1],cmap="grey")
            ix+=1

    plt.show()

'''
