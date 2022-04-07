#-*- conding:utf-8 -*-
import os
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
import keras 
import h5py 
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,TensorBoard
from CGNet import Net

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
KTF.set_session(sess)
epochs = 4 
NUM_Classes = 2
imsize = (224,224,3)        #裁减为20 个224*224*3 的固定分辨率图像块
batch_size = 32
subdir = './4850/check/'
model_name = '/4850_'+str(epochs)

  
train_datagen = ImageDataGenerator(1./255.)     
train_generator = train_datagen.flow_from_directory(
    './4850/cut_224/train/',
    target_size=(imsize[0], imsize[1]),
    batch_size=batch_size,
    shuffle=True,
    class_mode = 'binary'
)

val_datagen = ImageDataGenerator(1./255.)       
val_generator = val_datagen.flow_from_directory(
    './4850/cut_224/valid/',
    target_size=(imsize[0], imsize[1]),
    batch_size=batch_size,
    class_mode = 'binary'
)

model = Net(imsize,NUM_Classes)
model.summary()
history = LossHistory()
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=0,mode = 'auto')      #在训练过程中缩小学习率
checkpoint = ModelCheckpoint(subdir+model_name+"_"+"{epoch:02d}-{loss:.2f}.h5", monitor='loss', 
                             verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)
sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['acc'])

model.fit_generator(
	train_generator,
	epochs=epochs,
	verbose =1,
    validation_data = val_generator,
    validation_steps=val_generator.n/batch_size,
    steps_per_epoch=train_generator.n//batch_size,
	callbacks=[reduce_lr,checkpoint,history],
    shuffle= True
)

#with open(subdir+'loss.txt','a',encoding='utf-8') as f:
#    f.write(str(history.losses['epoch']))

#with open(subdir+'acc.txt','a',encoding='utf-8') as f:
#    f.write(str(history.losses['epoch']))   

#model.save('./'+model_name+'.h5')
model.save('cgnet4.h5')
print('train done')
