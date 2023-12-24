import os
import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Function to load and preprocess images
def load_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=(64, 64))  # Resize images to (64, 64)
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return images, labels

# Load images and labels for pencils
pencil_folder = r"C:\Users\Divya M\OneDrive\Desktop\binary\pencil" 
pencil_images, pencil_labels = load_images(pencil_folder, 0)

# Load images and labels for pens
pen_folder = r"C:\Users\Divya M\OneDrive\Desktop\binary\pen"
pen_images, pen_labels = load_images(pen_folder, 1)

# Combine pencil and pen data
all_images = np.concatenate([pencil_images, pen_images])
all_labels = np.concatenate([pencil_labels, pen_labels])

# Convert lists to NumPy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with data augmentation and early stopping
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20,
          validation_data=(X_test, y_test), callbacks=[early_stopping])

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate CNN accuracy
accuracy_cnn = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy of CNN: {accuracy_cnn}")

# Build AdaBoost model using Decision Trees
base_classifier = DecisionTreeClassifier(max_depth=2)
boosted_model = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)

# Train the AdaBoost model
boosted_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Make predictions using AdaBoost
y_pred_boosted = boosted_model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate AdaBoost accuracy
accuracy_boosted = accuracy_score(y_test, y_pred_boosted)
print(f"Accuracy of AdaBoost: {accuracy_boosted}")

# Load the test image
test_image_path = r"C:\Users\Divya M\OneDrive\Desktop\binary\pen1.jpg"  # Replace with the path to your test image
test_img = image.load_img(test_image_path, target_size=(64, 64))
test_img_array = image.img_to_array(test_img)
test_img_array = np.expand_dims(test_img_array, axis=0)  # Add batch dimension

# Preprocess the test image
test_img_array = preprocess_input(test_img_array)

# Use the trained CNN model to predict the class
cnn_prediction = model.predict(test_img_array)
cnn_class_label = "Pen" if cnn_prediction > 0.4 else "Pencil"  # Adjust the threshold

# Use the trained AdaBoost model to predict the class
adaboost_prediction = boosted_model.predict(test_img_array.reshape(1, -1))
adaboost_class_label = "Pen" if adaboost_prediction == 1 else "Pencil"
print(f"The predicted class using CNN is: {cnn_class_label}")

