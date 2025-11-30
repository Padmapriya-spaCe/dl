# Image Captioning - Simple Test File
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import os

# ============================
# Load tokenizer
# ============================
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
vocab_size = len(tokenizer.word_index) + 1
max_len = 34  # must match training

# =============================================
# Build the same model used during training
# =============================================
def build_caption_model(vocab_size, max_len):
    # --- Image feature branch ---
    img_input = Input(shape=(512,))        # VGG19 gives 512 features
    x1 = Dropout(0.5)(img_input)
    x1 = Dense(256, activation='relu')(x1)  # Reduce to 256

    # --- Text branch ---
    text_input = Input(shape=(max_len,))
    x2 = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    x2 = Dropout(0.5)(x2)
    x2 = LSTM(256)(x2)

    # --- Combine both ---
    merged = add([x1, x2])
    merged = Dense(256, activation='relu')(merged)
    output = Dense(vocab_size, activation='softmax')(merged)

    model = Model(inputs=[img_input, text_input], outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

# ============================
# Load model + weights
# ============================
model = build_caption_model(vocab_size, max_len)
model.load_weights("caption_vgg19_braintumor.h5")
print("Model loaded successfully!")

# =============================================
# VGG19 Feature Extractor and caption generator
# =============================================
vgg = VGG19(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg.predict(img, verbose=0)
    return features  # shape (1, 512)

def generate_caption(model, tokenizer, photo_features, max_len):
    caption = "startseq"
    for _ in range(max_len):
        # Convert text → sequence of numbers
        seq = tokenizer.texts_to_sequences([caption])[0]

        # Pad to fixed length
        seq = np.pad(seq, (0, max_len - len(seq)), 'constant')

        # Predict next word
        y_pred = model.predict([photo_features, seq.reshape(1, max_len)], verbose=0)
        next_word_id = np.argmax(y_pred)
        next_word = tokenizer.index_word.get(next_word_id)
        if next_word is None:
            break
        caption += " " + next_word
        if next_word == "endseq":
            break
    return caption

# ============================
# Run on test folder
# ============================
#test_dir = "./BrainTumorDataset/test"
test_dir = "./brain_tumor_dataset/Testing"

for cls in os.listdir(test_dir):
    class_path = os.path.join(test_dir, cls)
    images = os.listdir(class_path)[:10]  # first 10 images
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        features = extract_features(img_path)
        caption = generate_caption(model, tokenizer, features, max_len)
        print(f"{img_name}: {caption}")


'''
# test_caption_model.py - my code long version
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
vocab_size = len(tokenizer.word_index) + 1
max_len = 34  # same max_len used in training

# -----------------------------
# Rebuild the same model architecture
# -----------------------------
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(512,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# -----------------------------
# Rebuild and load weights
# -----------------------------
model = define_model(vocab_size, max_len)
model.load_weights("caption_vgg19_braintumor.h5")
print("✅ Model rebuilt and weights loaded successfully!")

# -----------------------------
# Save rebuilt model in modern format
# -----------------------------
rebuilt_keras_file = "caption_vgg19_braintumor_rebuilt2.h5"
model.save(rebuilt_keras_file)
print(f"✅ Rebuilt model saved as '{rebuilt_keras_file}' for direct future use!")

# -----------------------------
# Feature extractor (VGG19)
# -----------------------------
vgg_base = VGG19(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = vgg_base.predict(img, verbose=0)
    return feat

# -----------------------------
# Caption generation
# -----------------------------
def generate_caption(model, tokenizer, photo_feat, max_len):
    in_text = "startseq"
    for i in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = np.pad(sequence, (0, max_len - len(sequence)), 'constant')
        yhat = model.predict([photo_feat, sequence.reshape(1, max_len)], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# -----------------------------
# Loop over multiple test images
# -----------------------------
import os

test_images_dir = "./BrainTumorDataset/test"
for cls in os.listdir(test_images_dir):
    cls_path = os.path.join(test_images_dir, cls)
    images = os.listdir(cls_path)[:10]  # loop over first 10 images
    for img_file in images:
        img_path = os.path.join(cls_path, img_file)
        photo_feat = preprocess_image(img_path)
        caption = generate_caption(model, tokenizer, photo_feat, max_len)
        print(f"{img_file}: {caption}")

'''
'''
#image cap- sir code
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import cv2 
# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('model-ep004-loss3.572-val_loss3.833.h5')
# load and prepare the photograph
photo = extract_features('example.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
image = load_img('example.jpg')
image = img_to_array(image)
orig = cv2.imread('example.jpg')
cv2.putText(orig,description,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2,cv2.LINE_AA)
cv2.imshow("CAPTION GENERATION", orig)

cv2.waitKey(0)
print(description)

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen 

'''