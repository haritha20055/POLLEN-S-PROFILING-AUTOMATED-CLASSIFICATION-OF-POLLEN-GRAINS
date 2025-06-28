# pollen_profiling.py
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ====================================
# 1. Load dataset & do EDA
# ====================================
path = "data/"
names = [name.replace(' ', '_').split('_')[0] for name in os.listdir(path)]
classes = Counter(names)

print("Class counts:", classes)
print("Total images:", len(names))

# Bar chart
plt.figure(figsize=(10,4))
plt.bar(*zip(*classes.items()))
plt.title("Class Counts in Dataset")
plt.xticks(rotation=90)
plt.show()

# Scatter plot of image sizes
sizes = [cv2.imread(os.path.join(path, name)).shape for name in os.listdir(path)]
x, y, _ = zip(*sizes)
plt.figure(figsize=(5,5))
plt.scatter(x, y)
plt.plot([0,800],[0,800],'r')
plt.title("Image size scatterplot")
plt.show()

# ====================================
# 2. Preprocess images (resize + normalize)
# ====================================
def process_img(img, size=(128,128)):
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

X = []
Y = []

for name in os.listdir(path):
    img = cv2.imread(os.path.join(path, name))
    X.append(process_img(img))
    Y.append(name.replace(" ", "_").split("_")[0])

X = np.array(X)

# ====================================
# 3. Encode labels & split data
# ====================================
le = LabelEncoder()
Y_enc = le.fit_transform(Y)
Y_cat = to_categorical(Y_enc)  # ✅ corrected line

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.25, stratify=Y_cat
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ====================================
# 4. Build CNN model
# ====================================
input_shape = (128,128,3)
num_classes = Y_cat.shape[1]

model = Sequential([
    Conv2D(16, 3, activation='relu', padding='same', input_shape=input_shape),
    MaxPooling2D(3),
    Conv2D(32, 2, activation='relu', padding='same'),
    MaxPooling2D(2),
    Conv2D(64, 2, activation='relu', padding='same'),
    MaxPooling2D(2),
    Conv2D(128, 2, activation='relu', padding='same'),
    MaxPooling2D(3),
    Flatten(),
    Dropout(0.2),
    Dense(500, activation='relu'),
    Dropout(0.2),
    Dense(150, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()

# ====================================
# 5. Compile & train
# ====================================
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled.")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(X_train)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20),
    ModelCheckpoint('cnn_best.h5', save_best_only=True)
]

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=4),
    steps_per_epoch=len(X_train)//4,
    epochs=500,
    validation_data=(X_test, Y_test),
    callbacks=callbacks,
    verbose=1
)

# ====================================
# 6. Evaluate on test data
# ====================================
model.load_weights('cnn_best.h5')
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test accuracy: {:.2f}%".format(score[1]*100))

# ====================================
# 7. Plot accuracy & loss curves
# ====================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss")

plt.show()

# ====================================
# 8. Predict a single new image
# ====================================
img_path = "data/anadenanthera_16.jpg"  # change this to any image you want to predict
img = load_img(img_path, target_size=(128,128))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

y_pred = model.predict(x)
class_idx = np.argmax(y_pred, axis=1)[0]
class_name = le.inverse_transform([class_idx])[0]

print(f"Predicted class for {img_path}: {class_name}")
