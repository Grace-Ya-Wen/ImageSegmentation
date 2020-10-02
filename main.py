from model_1 import *
from data import *
from random import seed
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History
from matplotlib import pyplot as plt
from hyperopt import fmin, tpe, Trials

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='reflect')
                    
myGene = trainGenerator(16,'data/membrane/train/','test-raw','test-mask',data_gen_args)

input_img = Input((128, 128, 3), name='img')
model = get_unet(input_img, n_filters=32, dropout=0.1, batchnorm=True)

#reduceLR = ReduceLROnPlateau(factor=0.05, patience=3, min_lr=0.00001, verbose=1)
model_checkpoint = ModelCheckpoint('test_17.hdf5', monitor='loss',verbose=1, save_best_only=True)
earlystop = EarlyStopping(patience=8, verbose=1)

model.compile(optimizer=Adam(1e-6), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(myGene,steps_per_epoch=300,epochs=100,callbacks=[earlystop,model_checkpoint])


#model = load_model('13images_padding_0.01dropout.hdf5')
#result = model.evaluate(image_arr,mask_arr, batch_size = 1)
#print(result)
testGene = testGenerator("data/membrane/test/train-raw",num_image=3)
results = model.predict(testGene,3,verbose=1)
saveResult("data/membrane/test/0.01",results)


#model_fit = load_model('test.hdf5')
# Plot the loss function
#fig, ax = plt.subplots(1, 1, figsize=(10,6))


# plt.plot(model_fit.history['loss'], 'r', label='train')
# plt.plot(model_fit.history['val_loss'], 'b' ,label='val')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

#plt.plot(model_fit.history['acc'], 'r', label='train')
#plt.plot(model_fit.history['val_acc'], 'b' ,label='val')
#plt.ylabel('Loss')
#plt.xlabel('Accura#cy')
#plt.legend()
#plt.show()

#model = load_model('13images_padding_0.01dropout.hdf5')


#result = model.evaluate(image_arr,mask_arr, batch_size = 1)
#print(result)

