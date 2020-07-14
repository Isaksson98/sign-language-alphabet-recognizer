from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import os
import random

def visualize_dataset(image_list):

    if image_list.size == 0:
        _,dir, _ = next(os.walk('C:/Users/Filip/Code/DeepLearning/Datasets/SignLanguage/train/'))

        fig, ax = plt.subplots(3,3)
        for idy, letter in enumerate(random.sample(dir,3)):
            _,_, images = next(os.walk('C:/Users/Filip/Code/DeepLearning/Datasets/SignLanguage/train/'+letter))

            for idx, img in enumerate(random.sample(images, 3)):
                img_read = plt.imread('C:/Users/Filip/Code/DeepLearning/Datasets/SignLanguage/train/'+letter+'/'+img)
                ax[idy, idx].imshow(img_read)
                if idx == 0:
                    ax[idy, idx].set_ylabel(letter, rotation=0, fontsize=18, labelpad=20)
                ax[idy, idx].grid('off')
                #ax[idy, idx].set_title(img)
                ax[idy, idx].set_xticks([])
                ax[idy, idx].set_yticks([])
        plt.show()
    else:
        fig, ax = plt.subplots(3,3)
        for i in range(9):
            #print((int(i/3)), end=', ')
            #print(i%3)
            ax[int(i/3), i%3].axis('off')
            ax[int(i/3), i%3].imshow(image_list[i].reshape(40,40), cmap='gray')
        plt.show()

#visualize_dataset(np.array([]))

def evaluate(history):
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')

    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.savefig('C:/Users/Filip/Code/DeepLearning/sign-language-alphabet-recognizer/plots/accuracy14.png')
    plt.show()

    plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.savefig('C:/Users/Filip/Code/DeepLearning/sign-language-alphabet-recognizer/plots/loss14.png')

from sklearn.model_selection import train_test_split

BATCH_SIZE = 32 #128 Effect of batch size
EPOCHS = 20

def create_network():
    num_of_classes = 29
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(40,40,1), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(40,40,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    #model.add(Dense(units=128, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(units=num_of_classes, activation='softmax'))

    adamOpti = Adam(lr = 0.0001) # Default lr = 0.0001
    #Effect of learning rate?
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model

model = create_network()

training_set_generator = ImageDataGenerator(rescale = 1./255,
                                            validation_split=0.2,
                                            zoom_range = 0.3,
                                            rotation_range=10,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            fill_mode='nearest'
                                            )

training_set = training_set_generator.flow_from_directory('C:/Users/Filip/Code/DeepLearning/Datasets/SignLanguage/train',
                                    class_mode='categorical',
                                    batch_size=BATCH_SIZE,
                                    color_mode='grayscale',
                                    target_size=(40,40),
                                    subset='training',
                                    shuffle=True)

validation_set = training_set_generator.flow_from_directory('C:/Users/Filip/Code/DeepLearning/Datasets/SignLanguage/train',
                                    class_mode='categorical',
                                    batch_size=BATCH_SIZE,
                                    color_mode='grayscale',
                                    target_size=(40,40),
                                    subset='validation',
                                    shuffle=True)

x,y = training_set.next()
#visualize_dataset(x)

checkpoint = ModelCheckpoint("C:/Users/Filip/Code/DeepLearning/sign-language-alphabet-recognizer/model_weights.h5",
                            monitor='val_accuracy', verbose=1,
                            save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator = training_set,
                              steps_per_epoch = training_set.samples // BATCH_SIZE,
                              validation_data = validation_set, 
                              validation_steps = validation_set.samples // BATCH_SIZE,
                              epochs = EPOCHS,
                              verbose = 2,
                              callbacks = callbacks_list)

model.save('C:/Users/Filip/Code/DeepLearning/sign-language-alphabet-recognizer/signLanguage.h5')

evaluate(history)