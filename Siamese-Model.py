import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFont, ImageDraw


def initialize_base_network():  #to build a base network with VGG16 architecture
    VGGpart = tf.keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3,))
    for layer in VGGpart.layers:
        layer.trainable = True
    
    last_layer = VGGpart.get_layer('block5_pool')
    print('last layer output shape ', last_layer.output_shape)
    last_output = last_layer.output                

    x = Flatten(name="flatten_input")(last_output)
    x = Dense(512, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(256, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    output = Dense(128, activation='relu', name="third_base_dense")(x)
    model1 = tf.keras.models.Model(inputs=VGGpart.input, outputs=output)

    return model1

  
  
base_network = initialize_base_network()

# create the left input and point to the base network
input_a = Input(shape=(224,224,3,), name="left_input")
vect_output_a = base_network(input_a)

# create the right input and point to the base network
input_b = Input(shape=(224,224,3,), name="right_input")
vect_output_b = base_network(input_b)

# measure the similarity of the two vector outputs
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([vect_output_a,vect_output_b])
    
# Add a dense layer with a sigmoid unit to generate the similarity score
output = tf.keras.layers.Dense(1,activation='sigmoid')(L1_distance)

model = Model([input_a, input_b], output)
#model.summary()

#to visualize the model
# from tensorflow.python.keras.utils.vis_utils import plot_model
# plot_model(model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')


#Data Generator for the model 
siamese_df = pd.read_csv('data/siamese_df.csv')
train_siamese = siamese_df[0: int(len(siamese_df) * 0.80)]
val_siamese = siamese_df[int(len(siamese_df) * 0.80) : ]

generator = ImageDataGenerator(rescale = 1/255.0)


def get_flow_from_dataframe(generator, dataframe,directory="/tmp/Veri/VeRi/image_train",
                            image_shape=(224, 224),
                            color_mode='rgb', batch_size=16):

    train_generator_1 = generator.flow_from_dataframe(dataframe, directory = directory,
                                                        target_size=image_shape,
                                                        color_mode=color_mode,
                                                        x_col='Image1',
                                                        y_col='label',
                                                        shuffle = False,
                                                        class_mode='binary',                                                
                                                        batch_size=batch_size,                                                        
                                                        drop_duplicates=False)

    train_generator_2 = generator.flow_from_dataframe(dataframe, directory = directory,     
                                                        target_size=image_shape,
                                                        color_mode=color_mode,
                                                        x_col='Image2',
                                                        y_col='label',
                                                        shuffle = False,
                                                        class_mode='binary',                                          
                                                        batch_size=batch_size,                                                      
                                                        drop_duplicates=False)
    while True:
        x_1 = train_generator_1.next()
        x_2 = train_generator_2.next()
        yield [ x_1[0] , x_2[0]  ] , x_1[1]
       
      
train_gen = get_flow_from_dataframe(generator, train_siamese, image_shape=(224,224),
                                        color_mode='rgb',
                                        batch_size=16)

val_gen = get_flow_from_dataframe(generator, val_siamese, image_shape=(224,224),
                                        color_mode='rgb',
                                        batch_size=16)

#test the generator 
'''
a = next(train_gen)[0:]
a1 = a[0][0][0]
a2 = a[0][1][0]
c = a[1][0]
rows, cols = 1, 2
plt.subplot(rows, cols, 1)
plt.imshow(a1)
plt.subplot(rows, cols, 2)
plt.imshow(a2)
print("LABEL :", c)
'''

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.001, s=10)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights = True)
mc =tf.keras.callbacks.ModelCheckpoint('BaselineSiamese1.h5', save_best_only=True)

class CustomCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99.0% accuracy so cancelling training!")
                self.model.stop_training = True
mycallback = CustomCallBack()

adam = tf.keras.optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=adam, metrics =['accuracy' , tf.keras.metrics.Precision() , tf.keras.metrics.Recall()])
history = model.fit(train_gen,  
                    steps_per_epoch = len( train_siamese)//16, 
                    epochs=20,
                    validation_data= val_gen, 
                    validation_steps= len(val_siamese)//16,
                    callbacks= [early_stopping_cb,mc,mycallback])

fig = plt.figure(figsize=(10,10))

# Plot accuracy
plt.subplot(221)
plt.plot(history.history['accuracy'],'bo-', label = "acc")
plt.plot(history.history['val_accuracy'], 'ro-', label = "val_acc")
plt.title("train_accuracy vs val_accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()

# Plot loss function
plt.subplot(222)
plt.plot(history.history['loss'][1:],'bo-', label = "loss")
plt.plot(history.history['val_loss'][1:], 'ro-', label = "val_loss")
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()


plt.subplot(223)
plt.plot(history.epoch,history.history['lr'],'o-')
plt.title("train_loss vs val_loss")
plt.ylabel("learning rate")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()
