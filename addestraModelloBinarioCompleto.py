import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import modelliScript.new_nn_traffic_binary_randomized as nn
import modelliScript.single_lut_nn
from keras.utils import to_categorical
import sklearn.feature_selection as fs
import hickle as hkl
from sklearn.utils import class_weight
from tensorflow import keras
from keras import backend as K
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder


print("Loading data")
data = hkl.load('dataNotNormalized_pcap.hkl')
print("Module loaded")








def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val





X_train = data['xtrain']
X_test = data['xtest']
y_train = data['ytrain']
y_test = data['ytest']

#adeguiamo il tutto a una rappresentazione binaria
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print("Tipo di X_train: ", type(X_train))
print("Tipo di y_train: ", type(y_train))




FEATURE_NUMBERS = 4
BITWIDTH = 8
CLASS_NUMBER = 2


if FEATURE_NUMBERS == -1:
    model = nn.nn(bitwidth=BITWIDTH,LUT=1,class_number=CLASS_NUMBER)

if FEATURE_NUMBERS==8:
    # instantiate full model
    print("Instantiating model with 7 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=7,class_number=CLASS_NUMBER)
if FEATURE_NUMBERS==6:
    # instantiate middle model
    print("Instantiating model with 5 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=5,class_number=CLASS_NUMBER)
if FEATURE_NUMBERS==4:
    # instantiate middle model
    print("Instantiating model with 3 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=3,class_number=CLASS_NUMBER)
if FEATURE_NUMBERS==2:
    # instantiate small model
    print("Instantiating model with 1 LUT")
    model = nn.nn(bitwidth=BITWIDTH,LUT=1,class_number=CLASS_NUMBER)
 
#adeguiamo il tutto a una rappresentazione binaria
num_ones = np.sum(y_train)
total = y_train.size
print(f"Number of ones in y_train: {num_ones}, Total size of y_train: {total}, Ratio: {num_ones / total}")

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


print("Forma di y_train:", y_train.shape)
print("Forma di y_test:", y_test.shape)

opt = keras.optimizers.Adam(0.00001)
loss_object = tf.keras.losses.CategoricalCrossentropy()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_metric])

checkpoint_filepath = './tmp/model'+str(BITWIDTH)+'Complete'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_f1_metric',
                            verbose=1, save_best_only=True,save_weights_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)



if FEATURE_NUMBERS==8:
    print("FEATURE NUMBERS 8")
    # instantiate full model
    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5],X_train[:,6],X_train[:,7]], y_train,
                    batch_size=512,epochs=100,shuffle=True,
                    #class_weight= dic_weights,
                    callbacks=[checkpoint, early_stopping, reduce_lr],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test))
    model.load_weights(checkpoint_filepath)
    model.save('modelloTCPBinEdge-IIoT8.h5')
    
if FEATURE_NUMBERS==6:
    print("FEATURE NUMBERS 6")
    # instantiate full model
    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5]], y_train,
                    batch_size=512,epochs=100,shuffle=True,
                    #class_weight= dic_weights,
                    callbacks=[checkpoint, early_stopping, reduce_lr],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5]],y_test))
    model.load_weights(checkpoint_filepath)
    model.save('modelloTCPBinEdge-IIoT6ChiSquare.h5')
    
if FEATURE_NUMBERS==4:
    print("FEATURE NUMBERS 4")
    # instantiate full model
    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3]], y_train,
                    batch_size=512,epochs=100,shuffle=True,
                    #class_weight= dic_weights,
                    callbacks=[checkpoint, early_stopping, reduce_lr],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3]],y_test))
    model.load_weights(checkpoint_filepath)
    model.save('DDoSDetectorFullPrecisionBit2.h5')
    
if FEATURE_NUMBERS==2:
    # modifico in modo da considerare len e ttl del pacchetto
    print("FEATURE NUMBERS 2")
    # instantiate full model
    history = model.fit([X_train[:,0],X_train[:,1]], y_train,
                    batch_size=512,epochs=100,shuffle=True,
                    #class_weight= dic_weights,
                    callbacks=[checkpoint, early_stopping, reduce_lr],
                    validation_data = ([X_test[:,0],X_test[:,1]],y_test))
    model.load_weights(checkpoint_filepath)
    model.save('modelloTCPBinEdge-IIoT4f.h5')