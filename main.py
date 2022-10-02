import numpy as np
from keras import models
from keras import layers
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

validation_images = train_images[50000:]
train_images = train_images[:50000]

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

validation_labels = train_labels[50000:]
train_labels = train_labels[:50000]


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer = RMSprop(),
              loss = "categorical_crossentropy",
              metrics = ["acc"])
model.summary()

epochs = 5
history = model.fit(train_images,
                    train_labels, 
                    batch_size = 128,
                    epochs = epochs, 
                    validation_data=(validation_images, validation_labels))
model.save('digit_recognition.h5')

train_loss = history.history['loss']
validation_loss = history.history['val_loss']
train_acc = history.history['acc']
validation_acc = history.history['val_acc']

xs = list(range(1, epochs+1))

plt.plot(xs, train_loss, 'r', label='Training loss')
plt.plot(xs, validation_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xticks(range(1,epochs+1))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()

plt.plot(xs, train_acc, 'r', label='Training acc')
plt.plot(xs, validation_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xticks(range(1,epochs+1))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

model.evaluate(test_images, test_labels)

import random
data = mnist.load_data()[1][0][random.randint(0,9999)]
plt.imshow(data, cmap=plt.cm.gray_r)
plt.show()

result = model.predict(data.reshape((1, 28, 28, 1)).astype("float32") / 255)
probability = max(result[0]) / sum(result[0]) * 100
print("Result:", np.argmax(result), end = "  ")
print(f"Probability: {probability:.3f}%")