import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import random

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
model.compile(
    optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics = ["accuracy"])
model.fit(train_images, train_labels, epochs = 5) 

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Loss = %.3f" % test_loss)
print("Accuracy = %.3f" % test_acc)

n = random.randint(0, 9999)
plt.imshow(test_images[n])
#plt.imshow(train_images[n], cmap=plt.cm.binary)
plt.show()
prediction = model.predict(test_images)
print("Prediction %d" % np.argmax(prediction[n]))

