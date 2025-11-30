import h5py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# -----------------------------
# Load model & class names
# -----------------------------
model = load_model('./output/vgg19_brain_model.h5')
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# -----------------------------
# Load test features from HDF5
# -----------------------------
def load_test_features(h5_path):
    with h5py.File(h5_path, 'r') as f:
        X_list, y_list = [], []
        for i in range(f['batches'][()]):
            X_list.append(np.array(f[f'features-{i}']))
            y_list.append(np.array(f[f'labels-{i}']))

    X = np.vstack(X_list)
    y = np.hstack([np.argmax(lbl, axis=1) for lbl in y_list])
    return X, y

X_test, y_test = load_test_features('./data/test_brain.h5')
print("Loaded test samples:", X_test.shape[0])

# -----------------------------
# Per-class metrics (50 per class)
# -----------------------------
per_class_rows = []

for i, cname in enumerate(classes):
    idx = np.where(y_test == i)[0][:50]     # first 50 samples per class
    y_true = y_test[idx]
    y_pred = np.argmax(model.predict(X_test[idx]), axis=1)

    per_class_rows.append([
        cname,
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall_score(y_true, y_pred, average='macro', zero_division=0),
        f1_score(y_true, y_pred, average='macro', zero_division=0)
    ])

per_class_df = pd.DataFrame(per_class_rows, 
                            columns=['Class','Accuracy','Precision','Recall','F1-score'])
print("\nPer-class metrics:\n", per_class_df)

# -----------------------------
# Overall metrics (all test samples)
# -----------------------------
y_pred_all = np.argmax(model.predict(X_test), axis=1)

overall_df = pd.DataFrame([[
    'Overall',
    accuracy_score(y_test, y_pred_all),
    precision_score(y_test, y_pred_all, average='macro', zero_division=0),
    recall_score(y_test, y_pred_all, average='macro', zero_division=0),
    f1_score(y_test, y_pred_all, average='macro', zero_division=0)
]], columns=['Class','Accuracy','Precision','Recall','F1-score'])

print("\nOverall metrics:\n", overall_df)

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_all)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=classes, yticklabels=classes)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# -----------------------------
# Save metrics
# -----------------------------
os.makedirs('./output', exist_ok=True)
per_class_df.to_csv('./output/per_class_metrics.csv', index=False)
overall_df.to_csv('./output/overall_metrics.csv', index=False)

print("\nSaved metrics to ./output/")



'''
#feature extraction-my code long version
import h5py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load trained model
# -----------------------------
model_path = './output/vgg19_brain_model.h5'
model = load_model(model_path)
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# -----------------------------
# Load test features
# -----------------------------
def load_test_features(path):
    h5f = h5py.File(path, 'r')
    batch_count = h5f['batches'][()]
    X_list, y_list = [], []
    for batch_id in range(batch_count):
        X_list.append(np.array(h5f[f'features-{batch_id}']))
        y_list.append(np.array(h5f[f'labels-{batch_id}']))
    X_all = np.vstack(X_list)
    y_all = np.hstack([np.argmax(y, axis=1) for y in y_list])
    h5f.close()
    return X_all, y_all

X_test, y_test = load_test_features('./data/test_brain.h5')
print(f"Loaded {X_test.shape[0]} test samples.")

# -----------------------------
# Per-class metrics (≥50 images per class)
# -----------------------------
per_class_metrics = []
for i, class_name in enumerate(classes):
    idx = np.where(y_test == i)[0]
    if len(idx) < 50:
        print(f"Warning: class '{class_name}' has less than 50 images ({len(idx)}). Using all available samples.")
    idx = idx[:50]
    X_class = X_test[idx]
    y_true_class = y_test[idx]
    
    y_pred_class = np.argmax(model.predict(X_class), axis=1)
    
    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    
    per_class_metrics.append([class_name, acc, prec, rec, f1])

per_class_df = pd.DataFrame(per_class_metrics, columns=['Class','Accuracy','Precision','Recall','F1-score'])
print("\nPer-class metrics (≥50 images per class):")
print(per_class_df)

# -----------------------------
# Overall metrics
# -----------------------------
y_pred_all = np.argmax(model.predict(X_test), axis=1)
overall_acc = accuracy_score(y_test, y_pred_all)
overall_prec = precision_score(y_test, y_pred_all, average='macro', zero_division=0)
overall_rec = recall_score(y_test, y_pred_all, average='macro', zero_division=0)
overall_f1 = f1_score(y_test, y_pred_all, average='macro', zero_division=0)

overall_df = pd.DataFrame([['Overall', overall_acc, overall_prec, overall_rec, overall_f1]],
                          columns=['Class','Accuracy','Precision','Recall','F1-score'])
print("\nOverall metrics (all test samples):")
print(overall_df)

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_all)
print("\nConfusion Matrix (overall):")
print(cm)

plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Overall)")
plt.colorbar()
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# -----------------------------
# Save metrics CSV
# -----------------------------
os.makedirs('./output', exist_ok=True)
per_class_df.to_csv('./output/per_class_metrics.csv', index=False)
overall_df.to_csv('./output/overall_metrics.csv', index=False)
print("\nMetrics saved to ./output/per_class_metrics.csv and ./output/overall_metrics.csv")
'''

'''
#feature extraction- sir code
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