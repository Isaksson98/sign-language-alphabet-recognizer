import cv2
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import random
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator



CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

folder_path = 'C:/Users/Filip/Code/DeepLearning/Datasets/Signlanguage/test/'

#model = keras.models.load_model('C:/Users/Filip/Code/DeepLearning/sign-language-alphabet-recognizer/signLanguage.h5')
model = keras.models.load_model('C:/Users/Filip/Code/DeepLearning/sign-language-alphabet-recognizer/model_weights.h5')

def prepare(filepath):

    train_datagen = ImageDataGenerator(rescale = 1./255)

    IMG_SIZE = 40
    norm_image = cv2.normalize(filepath, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new_array = cv2.resize(norm_image, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

def predicting():
    
    train_datagen = ImageDataGenerator(rescale = 1./255)

    images = []
    imageFileNames = []

    for img in os.listdir(folder_path):
        imageFileNames.append(img)
        img = os.path.join(folder_path, img)
        img = image.load_img(img, color_mode='grayscale', target_size=(40, 40))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = train_datagen.standardize(img)
        
        images.append(img)


    z = list(zip(images, imageFileNames))
    random.shuffle(z)
    images, imageFileNames = zip(*z)

    # stack up images list to pass for prediction
    images = np.vstack(images)
    classes = model.predict_classes(images, batch_size=32)
    class_probs = model.predict_proba(images, batch_size=32)

    fig, ax = plt.subplots(3,3)
    for i in range(9):

        final = ''
        result_args = (-class_probs[i]).argsort()[:3]

        for m in range(3):
            final += (CATEGORIES[int(result_args[m])]) + ' '
            final += format(np.round(class_probs[i][int(result_args[m])],3),'.1%') + '\n'

        ax[int(i/3), i%3].grid('off')

        ax[int(i/3), i%3].set_xticks([])
        ax[int(i/3), i%3].set_yticks([])

        ax[int(i/3), i%3].set_ylabel(final, rotation=0, labelpad=20)
        if imageFileNames[i][0] == final[0]:
            ax[int(i/3), i%3].set_xlabel(imageFileNames[i][0])
        else:
            ax[int(i/3), i%3].set_xlabel(imageFileNames[i])
        ax[int(i/3), i%3].imshow(images[i].reshape(40,40), cmap='gray')
    #plt.show()
    
    fig, ax = plt.subplots(3,3)
    for i in range(9):
        j=i+9
        final = ''
        result_args = (-class_probs[j]).argsort()[:3]

        for m in range(3):
            final += (CATEGORIES[int(result_args[m])]) + ' '
            final += format(np.round(class_probs[j][int(result_args[m])],3),'.1%') + '\n'

        ax[int(i/3), i%3].grid('off')

        ax[int(i/3), i%3].set_xticks([])
        ax[int(i/3), i%3].set_yticks([])

        ax[int(i/3), i%3].set_ylabel(final, rotation=0, labelpad=20)
        if imageFileNames[j][0] == final[0]:
            ax[int(i/3), i%3].set_xlabel(imageFileNames[j][0])
        else:
            ax[int(i/3), i%3].set_xlabel(imageFileNames[j])
        ax[int(i/3), i%3].imshow(images[j].reshape(40,40), cmap='gray')
    #plt.show()

    fig, ax = plt.subplots(3,3)
    for i in range(9):
        k=i+18
        
        final = ''
        result_args = (-class_probs[k]).argsort()[:3]

        for m in range(3):
            final += (CATEGORIES[int(result_args[m])]) + ' '
            final += format(np.round(class_probs[k][int(result_args[m])],3),'.1%') + '\n'

        ax[int(i/3), i%3].grid('off')

        ax[int(i/3), i%3].set_xticks([])
        ax[int(i/3), i%3].set_yticks([])

        ax[int(i/3), i%3].set_ylabel(final, rotation=0, labelpad=20)
        if imageFileNames[k][0] == final[0]:
            ax[int(i/3), i%3].set_xlabel(imageFileNames[k][0])
        else:
            ax[int(i/3), i%3].set_xlabel(imageFileNames[k])
        ax[int(i/3), i%3].imshow(images[k].reshape(40,40), cmap='gray')
    plt.show()


#predicting()




cap = cv2.VideoCapture(0)#use 0 if using inbuilt webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #frame1 = cv2.resize(frame, (200, 200))
    frame = cv2.flip(frame, 1);
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    x=400
    y=150
    h=200
    w=200

    frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    crop_img = gray[y : (y + h), x :( x + w)]
    crop_img=cv2.flip(crop_img, 1);
    cv2.imshow('Video', crop_img)

    


    class_probs = model.predict_proba([prepare(crop_img)])
    prob_text = ''
    result_args = (-class_probs).argsort()[:3]


    for m in range(3):
        prob_text += (CATEGORIES[int(result_args[0][m])]) + ' '
        prob_text += format(np.round(class_probs[0][int(result_args[0][m])],2),'.1%') + '\n'

    
    prediction = model.predict([prepare(crop_img)])

    font = cv2.FONT_HERSHEY_SIMPLEX
    final = (CATEGORIES[int(np.argmax(prediction[0]))])
    cv2.putText(frame,final,(x+20,135), font, 1, (0,0,255), 2, cv2.LINE_AA)
    
    dy = 35
    for i, line in enumerate(prob_text.split('\n')):
        y2 = y + i*dy
        cv2.putText(frame, line,(x-250,y2+dy), font, 1/2, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow('Input', frame)
    

    c = cv2.waitKey(1)
    if c == 27: # hit esc key to stop
        break

cap.release()
cv2.destroyAllWindows()
