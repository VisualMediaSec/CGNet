from cbam_keras import attach_attention_module
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D,BatchNormalization,Flatten,Activation,Dense,concatenate,Concatenate
from keras.models import Model
from keras.regularizers import l1
from keras.layers import MaxPooling2D ,AveragePooling2D
import keras.backend as K 

def Net(imsize,NUM_Classes=2):
    base_model = VGG16(include_top=False,weights='imagenet',input_shape=imsize)
    feature_1 = base_model.get_layer('block1_pool').output #64*112*112     
    feature_2 = base_model.get_layer('block2_pool').output #128*56*56   
    feature_3 = base_model.get_layer('block3_pool').output #256*28*28
    #特征迁移
    feature_1 = Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_regularizer=l1(0.0015))(feature_1)
    feature_1 = BatchNormalization()(feature_1)
    feature_1 = Activation('relu')(feature_1)
    feature_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(feature_1) 
    feature_1 = Conv2D(256,(3,3),strides=(1,1),padding='same',kernel_regularizer=l1(0.0015))(feature_1)
    feature_1 = BatchNormalization()(feature_1)
    feature_1 = Activation('relu')(feature_1)
    feature_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(feature_1) 

    feature_1 = attach_attention_module(feature_1)

    feature_2 = Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_regularizer=l1(0.0015))(feature_2)
    feature_2 = BatchNormalization()(feature_2)
    feature_2 = Activation('relu')(feature_2)
    feature_2 = Conv2D(256,(3,3),strides=(1,1),padding='same',kernel_regularizer=l1(0.0015))(feature_2)
    feature_2 = BatchNormalization()(feature_2)
    feature_2 = Activation('relu')(feature_2)
    feature_2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(feature_2) # 256 * 28 * 28
    feature_2 = attach_attention_module(feature_2)

    feature_3 = Conv2D(256,(3,3),strides=(1,1),padding='same',kernel_regularizer=l1(0.0015))(feature_3)
    feature_3 = BatchNormalization()(feature_3)
    feature_3 = Activation('relu')(feature_3)
    feature_3 = Conv2D(256,(3,3),strides=(1,1),padding='same',kernel_regularizer=l1(0.0015))(feature_3)               
    feature_3 = BatchNormalization()(feature_3)
    feature_3 = Activation('relu')(feature_3)                                         # 256 * 28 * 28
    feature_3 = attach_attention_module(feature_3)

    feature_all = Concatenate(axis=-1)([feature_1,feature_2,feature_3]) #(256*3)*28*28
    feature_all = Conv2D(256,(1,1),strides=(1,1),padding='same',kernel_regularizer=l1(0.0015))(feature_all)
    feature_all = BatchNormalization()(feature_all)
    feature_all = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(feature_all) #
    feature_all = Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_regularizer=l1(0.0015))(feature_all) 
    feature_all = BatchNormalization()(feature_all)
    bt,h,w,c = K.int_shape(feature_all)
    feature_all = AveragePooling2D(pool_size=(h,w),padding='same')(feature_all)    

    feature_all = Flatten()(feature_all)
    feature_all = Dense(512,activation='relu')(feature_all)
    feature_all = Dense(256,activation='relu')(feature_all)
    feature_all = Dense(NUM_Classes,activation='softmax')(feature_all)
    model = Model(inputs =base_model.input,output = feature_all)
    return model
