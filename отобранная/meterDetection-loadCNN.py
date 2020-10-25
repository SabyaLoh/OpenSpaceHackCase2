import cv2 
#import sys
import os.path,os 
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


#imagePath = sys.argv[1]
cascPath = "cascade.xml"

# Подключаем каскад Хаара
meterCascade = cv2.CascadeClassifier(cascPath)

img = cv2.imread('meter_test2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0)
#Изменяем разрешение полученного изображения
final_len = 600
# проверяем вид изображения: альбомный или портретный
if(gray.shape[1] > gray.shape[0]):
    # альбомное изображение
    r = float(final_len)/gray.shape[1]
    dim = (final_len, int(gray.shape[0]*r))
else:
    # портрентое изображение
    r = float(final_len)/gray.shape[0]
    dim = (int(gray.shape[1]*r),final_len )
    
#уменьшаем изображение до подготовленных размеров
res = cv2.resize(gray,dim,interpolation = cv2.INTER_AREA)
# Определяем область счетчика
meters = meterCascade.detectMultiScale(
        res,
        scaleFactor=1.1,
        minNeighbors=50,
        minSize=(50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE)
print("Найдено областей: {0} ".format(len(meters)))

resizer=25
# Рисуем прямоугольник вокруг области и вырезаем
for (x, y, w, h) in meters:
    #cv2.rectangle(res, (x, y), (x+w, y+h), (255, 255, 255), 2)
    crop_img=res[y:y+h, x:x+w]
    r = float(resizer)/crop_img.shape[0]
    dim = (int(crop_img.shape[1]*r),resizer )
    crop_res = cv2.resize(crop_img,dim,interpolation = cv2.INTER_AREA)
    #plt.imshow(crop_res)

list=''
index = 0
dh,dw=crop_res.shape
#Нарезаем фрагменты с цифрами
for i in range(1,6):
    digit = crop_res[0:dh, (i-1)*int(dw/8):i*int(dw/8)]
    index+=1
    list = cv2.threshold(digit, 95,255,cv2.THRESH_BINARY)[1]
    fileNew=folder+str(index)+".jpg"
    cv2.imwrite(fileNew, list)  
    print(str(list))

#cv2.imshow("digit", res)
#cv2.imshow("crop", crop_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
folder = 'temp\\'

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



#for the_file in os.listdir(folder):
#    file_path = os.path.join(folder, the_file)
#    if os.path.isfile(file_path):
#        os.unlink(file_path)
#print(folder +"очищен")
    
