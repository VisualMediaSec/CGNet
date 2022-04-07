import keras
from keras import backend as K
from keras.preprocessing.image import *
from keras.models import Sequential, load_model

model = load_model('cgnet4.h5')
batch_size = 32
test_datagen = ImageDataGenerator(1./255.)
test_generator = test_datagen.flow_from_directory(
        './4850/cut_224/test/',
        shuffle=False,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

score = model.evaluate_generator(test_generator, verbose=1)
print('acc: ',score[1])
