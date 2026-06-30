import larq as lq
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import regularizers
    
def nn(bitwidth, LUT,class_number):

    if LUT == 1:
        # MODEL 1
        model1_in_1 = Input(shape=(8,),name='input1')
        model1_in_1_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='model1_in_1_quant')(model1_in_1)
        
        model1_layer_1 = Dense(256, activation='relu', name='model1_layer_1')(model1_in_1_quant)
        model1_bn_1 = BatchNormalization()(model1_layer_1)
        
        
        model1_layer_2 = Dense(128, activation='relu', name='model1_layer_2')(model1_bn_1)
        model1_bn_2 = BatchNormalization()(model1_layer_2)
        
        model1_layer_3 = Dense(32, activation='relu', name='model1_layer_3')(model1_bn_2)
        model1_bn_3 = BatchNormalization()(model1_layer_3)
        
        model1_out = Dense(class_number, activation='softmax', name='model1_out')(model1_bn_3)

        model1 = Model([model1_in_1], model1_out)

     
        plot_model(model1, to_file='model.png', show_shapes=True, show_layer_names=True)
        return model1
      