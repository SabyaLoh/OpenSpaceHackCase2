import os.path,os 
import cv2 
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

folder = 'digits\\'

print("ИНС загружается")
# Загружаем данные об архитектуре сети
json_file = open("neuro\mnist_model_cnn.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель
loaded_model = model_from_json(loaded_model_json)
# Загружаем сохраненные веса в модель
loaded_model.load_weights("neuro\mnist_model_cnn.h5")
print("ИНС загружена")

# Компилируем загруженную модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#Загружаем изображение
for fileNameImg in os.listdir(folder,):
    if fileNameImg.endswith(".jpg"):
        img_path=folder+str(fileNameImg)
        img = image.load_img(img_path,target_size=(28,28), grayscale = True)
    
#Преобразуем изображение в массив и нормализуем
        x=image.img_to_array(img)
        x=255-x
        x/=255
        x=np.expand_dims(x, axis=0)

#Запуск распознавания
        prediction = loaded_model.predict(x)
        prediction = np.argmax(prediction, axis=1)
        print(prediction)
plt.imshow(img, cmap='gray')
