import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from PIL import Image

(training_images, training_labels), \
    (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
model = models.load_model('image_classifier.model')
img = cv.imread('frog.jpeg')
img = cv.resize(img, (32, 32), interpolation=cv.INTER_LANCZOS4)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.expand_dims(img, axis=0) / 255)  # Add the extra dimension
index = np.argmax(prediction)
print(prediction)
print(f'prediction is {class_names[index]}')

plt.imshow(img, cmap=plt.cm.binary)
plt.show()
