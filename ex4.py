
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
