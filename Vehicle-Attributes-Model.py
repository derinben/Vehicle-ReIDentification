from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/My\ Drive/Project_VeRI/VeRi.zip

"""## Imports"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
import datetime
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tqdm import tqdm

"""## Pre-Processing Data"""

class data():

    def __init__(self, train_path , test_path):

        self.train_path = train_path
        self.test_path = test_path
        
    def load_data(self):

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_df.columns = ['ImageName' , 'VehicleID' ,'CameraID' , 'ColorID' ,'TypeID']
        test_df.columns = ['ImageName' , 'VehicleID' ,'CameraID' , 'ColorID' ,'TypeID']        

        train_df.drop(columns = ['CameraID'], axis = 1,inplace = True)
        train_df.drop(columns = ['VehicleID'], axis = 1,inplace = True)
        test_df.drop(columns = ['VehicleID'], axis = 1,inplace = True)
        test_df.drop(columns = ['CameraID'], axis = 1,inplace = True)

        index_df = train_df[train_df['TypeID'] == 5].index
        train_df.drop(index_df, inplace=True)
        
        return train_df, test_df
    
    def Enumerate(self, col):

        dict_train = {}
        dict_test  = {}
        self.train_df, self.test_df = self.load_data()

        arr_train = np.sort(self.train_df[col].unique())
        arr_test = np.sort(self.test_df[col].unique())

        for count, item in enumerate(arr_train):
            dict_train[item] = count
        for count, item in enumerate(arr_test):
            dict_test[item] = count          

        return dict_train ,dict_test
    
    def change(self):

        #VehicleID_dict_train, VehicleID_dict_test = self.Enumerate('VehicleID')
        ColorID_dict_train, ColorID_dict_test = self.Enumerate('ColorID')
        TypeID_dict_train, TypeID_dict_test = self.Enumerate('TypeID')
        
        #self.train_df = self.train_df.replace({"VehicleID": VehicleID_dict_train})
        self.train_df = self.train_df.replace({"ColorID": ColorID_dict_train})
        self.train_df = self.train_df.replace({"TypeID": TypeID_dict_train})

        #self.test_df = self.test_df.replace({"VehicleID": VehicleID_dict_test})
        self.test_df = self.test_df.replace({"ColorID": ColorID_dict_test})
        self.test_df = self.test_df.replace({"TypeID": TypeID_dict_test})

        return self.train_df, self.test_df

train_path = '/content/drive/MyDrive/Project_VeRI/train_label.csv'
test_path = '/content/drive/MyDrive/Project_VeRI/test_label.csv'

m = data(train_path, test_path)
train_df , test_df = m.change()

"""## Custom Generator"""

#Custom DataGenerator for Veri Dataset  
class VeriDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, X_col, y_col, batch_size=32,input_size = (100,100,3), shuffle=True):

        self.batch_size = batch_size
        self.df = df.copy()
        self.input_size = input_size
        self.shuffle = shuffle
        self.X_col = X_col
        self.y_col = y_col

        self.n = len(self.df)
        #self.n_vehicleID = df[y_col['VehicleID']].nunique()
        self.n_colorID = df[y_col['ColorID']].nunique()      
        self.n_typeID =  df[y_col['TypeID']].nunique() 


    def __get_input(self, path, target_size):
        
        image = tf.keras.preprocessing.image.load_img("/content/VeRi/image_train/" + path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        return image_arr/255.


    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes, dtype='float32')


    def __get_data(self, batches):
    
        path_batch = batches[self.X_col['ImageName']]  
        #vehicleID_batch = batches[self.y_col['VehicleID']] 
        colorID_batch = batches[self.y_col['ColorID']]
        TypeID_batch = batches[self.y_col['TypeID']]

        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])

        #y0_batch = np.asarray([self.__get_output(y, self.n_vehicleID) for y in vehicleID_batch])
        y0_batch = np.asarray([self.__get_output(y, self.n_colorID) for y in colorID_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_typeID) for y in TypeID_batch])

        return X_batch, tuple([y0_batch, y1_batch])

    
    def __len__(self):
        return self.n // self.batch_size

    
    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y

    def on_epoch_end(self):
        pass

    def data_show(self):
        print(self.df)

    def total(self):
        print("Total Images:", len(self.df))

#Custom DataGenerator for Veri Dataset  
class VeriDataGenerator_Valid(tf.keras.utils.Sequence):

    def __init__(self, df, X_col, y_col, batch_size=32,input_size = (100,100,3), shuffle=True):

        self.batch_size = batch_size
        self.df = df.copy()
        self.input_size = input_size
        self.shuffle = shuffle
        self.X_col = X_col
        self.y_col = y_col

        self.n = len(self.df)
        #self.n_vehicleID = df[y_col['VehicleID']].nunique()
        self.n_colorID = df[y_col['ColorID']].nunique()      
        self.n_typeID =  df[y_col['TypeID']].nunique() 


    def __get_input(self, path, target_size):
        
        image = tf.keras.preprocessing.image.load_img("/content/VeRi/image_test/" + path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        return image_arr/255.


    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes, dtype='float32')


    def __get_data(self, batches):
    
        path_batch = batches[self.X_col['ImageName']]  
        #vehicleID_batch = batches[self.y_col['VehicleID']] 
        colorID_batch = batches[self.y_col['ColorID']]
        TypeID_batch = batches[self.y_col['TypeID']]

        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])

        #y0_batch = np.asarray([self.__get_output(y, self.n_vehicleID) for y in vehicleID_batch])
        y0_batch = np.asarray([self.__get_output(y, self.n_colorID) for y in colorID_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_typeID) for y in TypeID_batch])

        return X_batch, tuple([y0_batch, y1_batch])

    
    def __len__(self):
        return self.n // self.batch_size

    
    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y

    def on_epoch_end(self):
        pass

    def data_show(self):
        print(self.df)

    def total(self):
        print("Total Images:", len(self.df))

train_data = train_df.sample(frac=1, random_state=42).reset_index(drop=True)


valid_data = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

traingen = VeriDataGenerator(train_data,
                            X_col={'ImageName':'ImageName'},
                            y_col={'ColorID': 'ColorID' , 'TypeID':'TypeID'},
                            batch_size=32, input_size=(224,224,3))

validgen = VeriDataGenerator_Valid(valid_data,
                            X_col={'ImageName':'ImageName'},
                            y_col={'ColorID': 'ColorID' , 'TypeID':'TypeID'},
                            batch_size=32, input_size=(224,224,3))

"""## Model

### Inception v3
"""

pre_trained_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
#pre_trained_model.summary()

tf.keras.backend.clear_session()

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
x = Flatten()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
output_1 = tf.keras.layers.Dense(10, activation='softmax', name='ColorID_Output')(x)
output_2 = tf.keras.layers.Dense(8, activation='softmax', name='TypeID_Output')(x)

model = Model(pre_trained_model.input, [output_1, output_2])
Adam = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['acc'])

model.summary()

"""#### Callbacks"""

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.0009, s=5)
lr_scheduler_ed = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
early_stopping_m = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

csv_file = 'appearance_attr_baseline.csv'

"""### Training"""

STEP_SIZE_TRAIN=traingen.n//traingen.batch_size

STEP_SIZE_VALID=validgen.n//validgen.batch_size

history = model.fit(traingen,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    epochs=5,
                    validation_data = validgen,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[CSVLogger(csv_file), tensorboard_callback],
                    verbose=1)

pd.read_csv(csv_file).head()

plt.figure(figsize=(12,12)) 
pd.DataFrame(history.history).plot()
plt.figure()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

cam_acc = history.history['ColorID_Output_acc']
val_cam_acc = history.history['val_ColorID_Output_acc']

type_acc = history.history['TypeID_Output_acc']
val_type_acc = history.history['val_TypeID_Output_loss']

epochs = range(0, 5)
plt.plot(epochs, loss, 'g')
plt.plot(epochs, val_loss, 'b')

plt.plot(epochs, cam_acc, 'g')
plt.plot(epochs, val_cam_acc, 'b')

plt.plot(epochs, type_acc, 'g')
plt.plot(epochs, val_type_acc, 'b')
