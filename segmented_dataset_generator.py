from tensorflow.keras.models import Model , load_model
from tensorflow.keras.preprocessing import image 
import numpy as np 
import os
from PIL import Image
import cv2
import numpy as np
import time, random, os
import pandas as pd

model = load_model('redline_segm.h5', compile = False)

dir = 'D:\SAJAT\\00_ML\TEST\humans_1_pre_test'
dir_res = 'D:\SAJAT\\00_ML\TEST\model_results'

#The base parameters for preparing the data
white = (255,255,255)
black = (0,0,0)
class_labels = (white, black)

#The size of the opened pictures
img_w = 320
img_h = 256

#list of filenames which are in the dir folder
file_names = sorted(os.listdir(dir))

##The functions to prepare and modify pictures
def load_imageset(folder): #opening all pictures from -dir- folder
    image_list = []
    cur_time = time.time()
    for filename in sorted(os.listdir(f'{folder}')):
        image_list.append(image.load_img(os.path.join(f'{folder}', filename), target_size=(img_w, img_h)))               
 
    print('All imageset is loaded. Time of loading: {:.2f} с'.format( time.time() - cur_time))
    print('Number of pictures:', len(image_list)) 

    return image_list

def rgb_to_labels(image_list): #turning rgb data to binary version (white - 0, black - 1)
    result = []

    for d in image_list:
        sample = np.array(d)
        y = np.zeros((img_w, img_h, 1), dtype='uint8')
        
        for i, cl in enumerate(class_labels): 
            y[np.where(np.all(sample == class_labels[i], axis=-1))] = i 

        result.append(y)
  
    return np.array(result)

def labels_to_rgb(image_list): #turning binary data to rgb (0 - white, 1 - black)
    result = []

    for y in image_list:
        temp = np.zeros((img_w, img_h, 3), dtype='uint8')

        for i, cl in enumerate(class_labels):
            temp[np.where(np.all(y==i, axis=-1))] = class_labels[i]

        result.append(temp)
  
    return np.array(result)



all_images = load_imageset(dir) #get all images from folder from which we want to create segmented ones

x_test = []   #creating numpy array from images to feed the model with it
for img in all_images:
    x = image.img_to_array(img)
    x_test.append(x)

x_test = np.array(x_test)
print(f"The pictures which will be processed have the next shape: {x_test.shape}")

x_bw = np.copy(x_test) #black and white version of original picture
line = [0, 0, 0]
other = [255,255,255]

for img in x_bw: #turning the picture to black and white (black where is red line, white other parts of the picture)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] >= 250 and img[i][j][1] <= 5 and img[i][j][2] <= 5 : #check if the pixel is red
                img[i][j] = line
            else:
                img[i][j] = other 

x_tester = rgb_to_labels(x_bw)

predict = np.argmax(model.predict(x_tester), axis=-1) #get the result from the model
test_segments = labels_to_rgb(predict[..., None]) #change the labeled array to rgb 

#saving the results to file to the result folder --> dir_res
for i in range(len(test_segments)):
    cv2.imwrite(f'{dir_res}\{file_names[i]}', test_segments[i])