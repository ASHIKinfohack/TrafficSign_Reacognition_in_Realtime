import pandas as pd
import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
# from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pickle
import os

# Load the dataset
with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)
with open('german-traffic-signs/valid.p', 'rb') as f:
    valid_data = pickle.load(f)

# Extract features and labels from the datasets
train_x, train_y = train_data['features'], train_data['labels']
val_x, val_y = valid_data['features'], valid_data['labels']
test_x, test_y = test_data['features'], test_data['labels']

# Preprocess the images
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

train_x = np.array([preprocessing(img) for img in train_x])
val_x = np.array([preprocessing(img) for img in val_x])
test_x = np.array([preprocessing(img) for img in test_x])

# Reshape the data
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], val_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# Convert labels to one-hot encoding
train_y = to_categorical(train_y, 43)
val_y = to_categorical(val_y, 43)
test_y = to_categorical(test_y, 43)

# Data Augmentation
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10.)
datagen.fit(train_x)

# Define the CNN model architecture
def build_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build the model
model = build_model()

# Train the model
history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=64),
                              steps_per_epoch=len(train_x) // 64,
                              epochs=28,
                              validation_data=(val_x, val_y),
                              shuffle=1)

# Save the trained model
model.save('model.h5')

# Evaluate model on test data
score = model.evaluate(test_x, test_y, verbose=0)
test_loss = score[0]
test_accuracy = score[1]

# Make predictions on test data
y_pred = model.predict(test_x)
y_pred_bool = np.argmax(y_pred, axis=1)

# Calculate precision, recall and f1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(np.argmax(test_y, axis=1), y_pred_bool,
                                                                 average='weighted')

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Test precision:", precision)
print("Test recall:", recall)
print("Test F1-score:", f1_score)

# Plot accuracy and loss over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
