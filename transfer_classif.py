# test_no_gradcam.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# --- config ---
MODEL_PATH = "vgg19_brain_tumor.h5"   # change to provided model name
EVAL_DIR   = "brain_tumor_dataset/Testing/"
TARGET_SZ  = (224, 224)
BATCH_SIZE = 32

# --- load model ---
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()  # optional: see architecture

# Classes
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ---------------- Evaluation ---------------- #
eval_dir = 'brain_tumor_dataset/Testing/'
eval_datagen = ImageDataGenerator(rescale=1./255)

eval_generator = eval_datagen.flow_from_directory(
    eval_dir, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False
)

# Evaluate on full test set
loss, accuracy = model.evaluate(eval_generator, steps=len(eval_generator))
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# ---------------- Utility Functions ---------------- #
def preprocess_input(x):
    x = img_to_array(x)/255.
    return np.expand_dims(x, axis=0)

def predict_image(im):
    x = preprocess_input(im)
    pred_index = np.argmax(model.predict(x))
    return pred_index, classes[pred_index]

def predict_from_image_path(image_path):
    return predict_image(load_img(image_path, target_size=(224,224)))

# ---------------- Single Image Prediction Example ---------------- #
image_path = 'brain_tumor_dataset/Testing/glioma/Te-gl_0010.jpg'
pred_index, pred_class = predict_from_image_path(image_path)
print(f"Prediction -> {pred_index}: {pred_class}")

# ---------------- Confusion Matrix ---------------- #
Y_true = eval_generator.classes
Y_pred = np.argmax(model.predict(eval_generator, steps=len(eval_generator)), axis=1)

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------- Classification Report ---------------- #
report_dict = classification_report(Y_true, Y_pred, target_names=classes, digits=4, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
print(df_report)

# Save metrics to CSV
df_report.to_csv("classification_metrics_full_test.csv")
print("Saved full test classification metrics to 'classification_metrics_full_test.csv'")



'''
transfer learning- my code long version
import os
import numpy as np
import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

from keras.models import load_model
from io import BytesIO
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
# ---------------- Load Model ---------------- #
model = load_model("vgg19_brain_tumor.h5")

# Classes
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
# Dataset folder names (match your testing folder structure)
test_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---------------- Evaluation ---------------- #
eval_datagen = ImageDataGenerator(rescale=1./255)
eval_dir = 'brain_tumor_dataset/Testing/'

eval_generator = eval_datagen.flow_from_directory(
    eval_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# Evaluate on full test set
loss = model.evaluate(eval_generator, steps=len(eval_generator))
for index, name in enumerate(model.metrics_names):
    print(name, loss[index])

# ---------------- Utility Functions ---------------- #
def preprocess_input(x):
    x = img_to_array(x) / 255.
    return np.expand_dims(x, axis=0)

def predict_image(im):
    x = preprocess_input(im)
    pred = np.argmax(model.predict(x))
    return pred, classes[pred]

def predict_from_image_path(image_path):
    return predict_image(load_img(image_path, target_size=(224, 224)))

def predict_from_image_url(image_url):
    res = requests.get(image_url)
    im = Image.open(BytesIO(res.content))
    return predict_image(im)

# ---------------- Grad-CAM (TF2 version) ---------------- #
# ---------------- Grad-CAM (TF2 version) ---------------- #
def grad_CAM(image_path):
    im = load_img(image_path, target_size=(224,224))
    x = preprocess_input(im)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer('block5_conv4').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        index = tf.argmax(predictions[0])
        class_channel = predictions[:, index]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = heatmap.numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Overlay heatmap
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite('./tmp.jpg', superimposed_img)

    plt.figure(figsize=(12,6))
    plt.imshow(mpimg.imread('./tmp.jpg'))
    plt.axis("off")
    plt.show()


# ---------------- Example Run ---------------- #
print(predict_from_image_path('brain_tumor_dataset/Testing/glioma/Te-gl_0010.jpg'))
grad_CAM('brain_tumor_dataset/Testing/glioma/Te-gl_0010.jpg')

# ---------------- Confusion Matrix + Metrics ---------------- #
print("\n=== Confusion Matrix and Metrics on Full Test Set ===")

Y_true = eval_generator.classes
Y_pred = model.predict(eval_generator, steps=len(eval_generator), verbose=1)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report (overall + per-class metrics)
report_dict = classification_report(Y_true, Y_pred_classes, target_names=classes, digits=4, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
print("\nMetrics Table (Overall + Per-Class):")
print(df_report)
df_report.to_csv("classification_metrics_full_test.csv")
print("\nSaved full test classification metrics to classification_metrics_full_test.csv")

# ---------------- Metrics for Minimum 50 Images per Class ---------------- #
indices_50_per_class = []
for i, cls in enumerate(classes):
    cls_indices = np.where(Y_true == i)[0][:50]  # first 50 images per class
    indices_50_per_class.extend(cls_indices)

Y_true_50 = Y_true[indices_50_per_class]
Y_pred_50 = Y_pred_classes[indices_50_per_class]

report_50_dict = classification_report(Y_true_50, Y_pred_50, target_names=classes, digits=4, output_dict=True)
df_report_50 = pd.DataFrame(report_50_dict).transpose()
print("\nMetrics Table (50 Images per Class):")
print(df_report_50)
df_report_50.to_csv("classification_metrics_50perclass.csv")
print("\nSaved 50-per-class metrics to classification_metrics_50perclass.csv")
'''


'''
#transfer-sir
import os
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.models import load_model
from keras import backend as K

from io import BytesIO
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors

import requests

K.set_learning_phase(0) #set the learning phase to not training


model = load_model('model.03-0.94.hdf5')


# Set the image generator
eval_datagen = ImageDataGenerator(rescale=1./255)
eval_dir = '../OCT2017/eval'
eval_generator = eval_datagen.flow_from_directory(eval_dir, target_size=(299, 299), \
                                                    batch_size=32, class_mode='categorical')
# Evaluate the model for a small set of images
loss = model.evaluate_generator(eval_generator, steps=10)
out = {}
for index, name in enumerate(model.metrics_names):
    print(name, loss[index])  

# Utility functions
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# Preprocess the input
# Rescale the values to the same range that was used during training 
def preprocess_input(x):
    x = img_to_array(x) / 255.
    return np.expand_dims(x, axis=0) 

# Prediction for an image path in the local directory
def predict_from_image_path(image_path):
    return predict_image(load_img(image_path, target_size=(299, 299)))

# Prediction for an image URL path
def predict_from_image_url(image_url):
    res = requests.get(image_url)
    im = Image.open(BytesIO(res.content))
    return predict_from_image_path(im.fp)
    
# Predict an image
def predict_image(im):
    x = preprocess_input(im)
    pred = np.argmax(model.predict(x))
    return pred, classes[pred]


def grad_CAM(image_path):
    im = load_img(image_path, target_size=(299,299))
    x = preprocess_input(im)
    pred = model.predict(x)
    
    # Predicted class index
    index = np.argmax(pred)
    
    # Get the entry of the predicted class
    class_output = model.output[:, index]
    
    # The last convolution layer in Inception V3
    last_conv_layer = model.get_layer('conv2d_94')
    # Has 192 channels
    nmb_channels = last_conv_layer.output.shape[3]

    # Gradient of the predicted class with respect to the output feature map of the 
    # the convolution layer with 192 channels
    grads = K.gradients(class_output, last_conv_layer.output)[0]   
    
    # Vector of shape (192,), where each entry is the mean intensity of the gradient over 
    # a specific feature-map channel”
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Setup a function to extract the desired values
    iterate = K.function(model.inputs, [pooled_grads, last_conv_layer.output[0]])
    # Run the function to get the desired calues
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    # Multiply each channel in the feature-map array by “how important this channel is” with regard to the 
    # predicted class
 
    for i in range(nmb_channels):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    # The channel-wise mean of the resulting feature map is the heatmap of the class activation.
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    # Normalize the heatmap betwen 0 and 1 for visualization
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
       
    # Read the image again, now using cv2
    img = cv2.imread(image_path)
    # Size the heatmap to the size of the loaded image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert to RGB
    heatmap = np.uint8(255 * heatmap)
    # Pseudocolor/false color a grayscale image using OpenCV’s predefined colormaps
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
 
    # Superimpose the image with the required intensity
    superimposed_img = heatmap * 0.5 + img   
    
    # Write the image
    plt.figure(figsize=(24,12))
    cv2.imwrite('./tmp.jpg', superimposed_img)
    plt.imshow(mpimg.imread('./tmp.jpg'))
    plt.show() 


print(predict_from_image_path('../OCT2017/eval/DME/DME-15307-3.jpeg'))
grad_CAM('../OCT2017/eval/DME/DME-15307-3.jpeg')



for i, c in enumerate(classes):
    folder = './simple/test/' + c + '/'
    count = 1
    for file in os.listdir(folder):
        if file.endswith('.jpeg') == True:
            image_path = folder + file
            p, class_name = predict_from_image_path(image_path)
            if p == i:
                print(file, p, class_name)
            else:
                print(file, p, class_name, '**INCORRECT PREDICTION**')
                grad_CAM(image_path)
        count = count +1
        if count == 100:
            continue    

'''