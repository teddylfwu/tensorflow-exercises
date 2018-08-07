# Tensorflow and tf.keras

import tensorflow as tf
from tensorflow import keras

# common libraries 
import numpy as np
import matplotlib.pyplot as plt

# check the version of tensorflow
print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labs), (test_imgs, test_labs) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#print(train_imgs.shape)
#print(train_imgs[0])
#plt.figure()
#plt.imshow(train_imgs[0])
#plt.colorbar()
#plt.gca().grid(False)
#plt.show()

train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

#matplotlib inline
plt.figure(figsize=(8,8))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid('off')
  plt.imshow(train_imgs[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labs[i]])
plt.show()

# define keras model architecture
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Start training the model
model.fit(train_imgs, train_labs, epochs=5)

# Start testing on test dataset
test_loss, test_acc = model.evaluate(test_imgs, test_labs)
print('Test accuracy:', test_acc)


# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
predictions = model.predict(test_imgs)
plt.figure(figsize=(8,8))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid('off')
  plt.imshow(test_imgs[i], cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions[i])
  true_label = test_labs[i]
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  plt.xlabel("{} ({})".format(class_names[predicted_label], 
                              class_names[true_label]),
                              color=color)
plt.show()
