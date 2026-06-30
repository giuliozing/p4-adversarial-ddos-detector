from tensorflow.keras.layers import Input, Dense, Concatenate, Add, BatchNormalization , Dropout
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model



def nn(bitwidth, LUT,class_number):

    first_layer_neurons = 512

    layer_1_5_neurons = 256

    second_layer_neurons = 128

    layer_2_5_neurons = 64

    third_layer_neurons = 32

    if LUT == 1:
        model1_in_1 = Input(shape=(1,),name='input1')
        model1_in_2 = Input(shape=(1,),name='input2')
        model_1_concatenated = Concatenate()([model1_in_1, model1_in_2])

        
        model1_layer_1 = Dense(first_layer_neurons, activation='relu', name='model1_layer_1')(model_1_concatenated)
        model1_dropout_1 = Dropout(0.05)(model1_layer_1)
        model1_bn_1 = BatchNormalization()(model1_dropout_1)
        
        model1_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model1_layer_1_5')(model1_bn_1)
        model1_dropout_2 = Dropout(0.05)(model1_layer_1_5)
        model1_bn_1_5 = BatchNormalization()(model1_dropout_2)
        
        model1_layer_2 = Dense(second_layer_neurons, activation='relu', name='model1_layer_2')(model1_bn_1_5)
        model1_dropout_3 = Dropout(0.05)(model1_layer_2)
        model1_bn_2 = BatchNormalization()(model1_dropout_3)

        model1_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model1_layer_2_5')(model1_bn_2)
        model1_dropout_4 = Dropout(0.05)(model1_layer_2_5)
        model1_bn_2_5 = BatchNormalization()(model1_dropout_4)
        
        model1_layer_3 = Dense(third_layer_neurons, activation='relu', name='model1_layer_3')(model1_bn_2_5)
        model1_dropout_5 = Dropout(0.05)(model1_layer_3)
        model1_bn_3 = BatchNormalization()(model1_dropout_5)

        
        model1_out = Dense(class_number, activation='softmax', name='model1_out')(model1_bn_3)


        model1 = Model([model1_in_1,model1_in_2], model1_out)

     
        plot_model(model1, to_file='model.png', show_shapes=True, show_layer_names=True)
        return model1

    if LUT == 3:

        # MODEL 1
        model1_in_1 = Input(shape=(1,),name='input1')
        model1_in_2 = Input(shape=(1,),name='input2')
        model_1_concatenated = Concatenate()([model1_in_1, model1_in_2])

        
        model1_layer_1 = Dense(first_layer_neurons, activation='relu', name='model1_layer_1')(model_1_concatenated)
        model1_dropout_1 = Dropout(0.05)(model1_layer_1)
        model1_bn_1 = BatchNormalization()(model1_dropout_1)
        
        model1_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model1_layer_1_5')(model1_bn_1)
        model1_dropout_2 = Dropout(0.05)(model1_layer_1_5)
        model1_bn_1_5 = BatchNormalization()(model1_dropout_2)
        
        model1_layer_2 = Dense(second_layer_neurons, activation='relu', name='model1_layer_2')(model1_bn_1_5)
        model1_dropout_3 = Dropout(0.05)(model1_layer_2)
        model1_bn_2 = BatchNormalization()(model1_dropout_3)

        model1_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model1_layer_2_5')(model1_bn_2)
        model1_dropout_4 = Dropout(0.05)(model1_layer_2_5)
        model1_bn_2_5 = BatchNormalization()(model1_dropout_4)
        
        model1_layer_3 = Dense(third_layer_neurons, activation='relu', name='model1_layer_3')(model1_bn_2_5)
        model1_dropout_5 = Dropout(0.05)(model1_layer_3)
        model1_bn_3 = BatchNormalization()(model1_dropout_5)

        
        model1_out = Dense(bitwidth, activation='relu', name='model1_out')(model1_bn_3)


        model1 = Model([model1_in_1,model1_in_2], model1_out)

        
        
        # MODEL 2
        model2_in_1 = Input(shape=(1,),name='input3')
        model2_in_2 = Input(shape=(1,),name='input4')
        model_2_concatenated = Concatenate()([model2_in_1, model2_in_2])

        
        model2_layer_1 = Dense(first_layer_neurons, activation='relu', name='model2_layer_1')(model_2_concatenated)
        model2_dropout_1 = Dropout(0.05)(model2_layer_1)
        model2_bn_1 = BatchNormalization()(model2_dropout_1)

        model2_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model2_layer_1_5')(model2_bn_1)
        model2_dropout_2 = Dropout(0.05)(model2_layer_1_5)
        model2_bn_1_5 = BatchNormalization()(model2_dropout_2)


        model2_layer_2 = Dense(second_layer_neurons, activation='relu', name='model2_layer_2')(model2_bn_1_5)
        model2_dropout_3 = Dropout(0.05)(model2_layer_2)
        model2_bn_2 = BatchNormalization()(model2_dropout_3)
        
        model2_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model2_layer_2_5')(model2_bn_2)
        model2_dropout_4 = Dropout(0.05)(model2_layer_2_5)
        model2_bn_2_5 = BatchNormalization()(model2_dropout_4)

        model2_layer_3 = Dense(third_layer_neurons, activation='relu', name='model2_layer_3')(model2_bn_2_5)
        model2_dropout_5 = Dropout(0.05)(model2_layer_3)
        model2_bn_3 = BatchNormalization()(model2_dropout_5)

        model2_out = Dense(bitwidth, activation='relu', name='model2_out')(model2_bn_3)

        model2 = Model([model2_in_1,model2_in_2], model2_out)

        
        # SECOND LAYER MODELS
        # MODEL 5]
        model5_in_1 = model1.output
        model5_in_2 = model2.output

        model_5_concatenated = Concatenate()([model5_in_1, model5_in_2])
        
        model5_layer_1 = Dense(first_layer_neurons, activation='relu', name='model5_layer_1')(model_5_concatenated)
        model5_dropout_1 = Dropout(0.05)(model5_layer_1)
        model5_bn_1  = BatchNormalization()(model5_dropout_1)


        model5_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model5_layer_1_5')(model5_bn_1)
        model5_dropout_2 = Dropout(0.05)(model5_layer_1_5)
        model5_bn_1_5 = BatchNormalization()(model5_dropout_2)
        
        model5_layer_2 = Dense(second_layer_neurons, activation='relu', name='model5_layer_2')(model5_bn_1_5)
        model5_dropout_3 = Dropout(0.05)(model5_layer_2)
        model5_bn_2 = BatchNormalization()(model5_dropout_3)
        
        model5_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model5_layer_2_5')(model5_bn_2)
        model5_dropout_4 = Dropout(0.05)(model5_layer_2_5)
        model5_bn_2_5 = BatchNormalization()(model5_dropout_4)

        model5_layer_3 = Dense(third_layer_neurons, activation='relu', name='model5_layer_3')(model5_bn_2_5)
        model5_dropout_5 = Dropout(0.05)(model5_layer_3)
        model5_bn_3 = BatchNormalization()(model5_dropout_5)
        
        model5_out = Dense(class_number, activation='softmax', name='model5_out')(model5_bn_3)

        model5 = Model([model1.input, model2.input], model5_out)


        
        plot_model(model5, to_file='model.png', show_shapes=True, show_layer_names=True)
        return model5


    if LUT == 5:

        # MODEL 1
        model1_in_1 = Input(shape=(1,),name='input1')
        model1_in_2 = Input(shape=(1,),name='input2')
        model_1_concatenated = Concatenate()([model1_in_1, model1_in_2])

        
        model1_layer_1 = Dense(first_layer_neurons, activation='relu', name='model1_layer_1')(model_1_concatenated)
        model1_dropout_1 = Dropout(0.05)(model1_layer_1)
        model1_bn_1 = BatchNormalization()(model1_dropout_1)
        
        model1_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model1_layer_1_5')(model1_bn_1)
        model1_dropout_2 = Dropout(0.05)(model1_layer_1_5)
        model1_bn_1_5 = BatchNormalization()(model1_dropout_2)
        
        model1_layer_2 = Dense(second_layer_neurons, activation='relu', name='model1_layer_2')(model1_bn_1_5)
        model1_dropout_3 = Dropout(0.05)(model1_layer_2)
        model1_bn_2 = BatchNormalization()(model1_dropout_3)

        model1_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model1_layer_2_5')(model1_bn_2)
        model1_dropout_4 = Dropout(0.05)(model1_layer_2_5)
        model1_bn_2_5 = BatchNormalization()(model1_dropout_4)
        
        model1_layer_3 = Dense(third_layer_neurons, activation='relu', name='model1_layer_3')(model1_bn_2_5)
        model1_dropout_5 = Dropout(0.05)(model1_layer_3)
        model1_bn_3 = BatchNormalization()(model1_dropout_5)

        
        model1_out = Dense(bitwidth, activation='relu', name='model1_out')(model1_bn_3)


        model1 = Model([model1_in_1,model1_in_2], model1_out)

        
        
        # MODEL 2
        model2_in_1 = Input(shape=(1,),name='input3')
        model2_in_2 = Input(shape=(1,),name='input4')
        model_2_concatenated = Concatenate()([model2_in_1, model2_in_2])

        
        model2_layer_1 = Dense(first_layer_neurons, activation='relu', name='model2_layer_1')(model_2_concatenated)
        model2_dropout_1 = Dropout(0.05)(model2_layer_1)
        model2_bn_1 = BatchNormalization()(model2_dropout_1)

        model2_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model2_layer_1_5')(model2_bn_1)
        model2_dropout_2 = Dropout(0.05)(model2_layer_1_5)
        model2_bn_1_5 = BatchNormalization()(model2_dropout_2)


        model2_layer_2 = Dense(second_layer_neurons, activation='relu', name='model2_layer_2')(model2_bn_1_5)
        model2_dropout_3 = Dropout(0.05)(model2_layer_2)
        model2_bn_2 = BatchNormalization()(model2_dropout_3)
        
        model2_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model2_layer_2_5')(model2_bn_2)
        model2_dropout_4 = Dropout(0.05)(model2_layer_2_5)
        model2_bn_2_5 = BatchNormalization()(model2_dropout_4)

        model2_layer_3 = Dense(third_layer_neurons, activation='relu', name='model2_layer_3')(model2_bn_2_5)
        model2_dropout_5 = Dropout(0.05)(model2_layer_3)
        model2_bn_3 = BatchNormalization()(model2_dropout_5)

        model2_out = Dense(bitwidth, activation='relu', name='model2_out')(model2_bn_3)

        model2 = Model([model2_in_1,model2_in_2], model2_out)

        
        
         # MODEL 3
        model3_in_1 = Input(shape=(1,),name='input5')
        model3_in_2 = Input(shape=(1,),name='input6')
        model_3_concatenated = Concatenate()([model3_in_1, model3_in_2])

        
        model3_layer_1 = Dense(first_layer_neurons, activation='relu', name='model3_layer_1')(model_3_concatenated)
        model3_dropout_1 = Dropout(0.05)(model3_layer_1)
        model3_bn_1 = BatchNormalization()(model3_dropout_1)


        model3_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model3_layer_1_5')(model3_bn_1)
        model3_dropout_2 = Dropout(0.05)(model3_layer_1_5)
        model3_bn_1_5 = BatchNormalization()(model3_dropout_2)


        model3_layer_2 = Dense(second_layer_neurons, activation='relu', name='model3_layer_2')(model3_bn_1_5)
        model3_dropout_3 = Dropout(0.05)(model3_layer_2)
        model3_bn_2 = BatchNormalization()(model3_dropout_3)
        
        model3_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model3_layer_2_5')(model3_bn_2)
        model3_dropout_4 = Dropout(0.05)(model3_layer_2_5)
        model3_bn_2_5 = BatchNormalization()(model3_dropout_4)


        model3_layer_3 = Dense(third_layer_neurons, activation='relu', name='model3_layer_3')(model3_bn_2_5)
        model3_dropout_5 = Dropout(0.05)(model3_layer_3)
        model3_bn_3 = BatchNormalization()(model3_dropout_5)

        model3_out = Dense(bitwidth, activation='relu', name='model3_out')(model3_bn_3)

        model3 = Model([model3_in_1,model3_in_2], model3_out)


        
        # SECOND LAYER MODELS
        # MODEL 5]
        model5_in_1 = model1.output
        model5_in_2 = model2.output

        model_5_concatenated = Concatenate()([model5_in_1, model5_in_2])
        
        model5_layer_1 = Dense(first_layer_neurons, activation='relu', name='model5_layer_1')(model_5_concatenated)
        model5_dropout_1 = Dropout(0.05)(model5_layer_1)
        model5_bn_1  = BatchNormalization()(model5_dropout_1)


        model5_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model5_layer_1_5')(model5_bn_1)
        model5_dropout_2 = Dropout(0.05)(model5_layer_1_5)
        model5_bn_1_5 = BatchNormalization()(model5_dropout_2)
        
        model5_layer_2 = Dense(second_layer_neurons, activation='relu', name='model5_layer_2')(model5_bn_1_5)
        model5_dropout_3 = Dropout(0.05)(model5_layer_2)
        model5_bn_2 = BatchNormalization()(model5_dropout_3)
        
        model5_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model5_layer_2_5')(model5_bn_2)
        model5_dropout_4 = Dropout(0.05)(model5_layer_2_5)
        model5_bn_2_5 = BatchNormalization()(model5_dropout_4)

        model5_layer_3 = Dense(third_layer_neurons, activation='relu', name='model5_layer_3')(model5_bn_2_5)
        model5_dropout_5 = Dropout(0.05)(model5_layer_3)
        model5_bn_3 = BatchNormalization()(model5_dropout_5)
        
        model5_out = Dense(bitwidth, activation='relu', name='model5_out')(model5_bn_3)

        model5 = Model([model1.input, model2.input], model5_out)


        model7_in_1 = model3.output
        model7_in_2 = model5.output
        model_7_concatenated = Concatenate()([model7_in_1, model7_in_2])
        

        
        model7_layer_1 = Dense(first_layer_neurons, activation='relu', name='model7_layer_1')(model_7_concatenated)
        model7_dropout_1 = Dropout(0.05)(model7_layer_1)
        model7_bn_1  = BatchNormalization()(model7_dropout_1)

        model7_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model7_layer_1_5')(model7_bn_1)
        model7_dropout_2 = Dropout(0.05)(model7_layer_1_5)
        model7_bn_1_5 = BatchNormalization()(model7_dropout_2)

        model7_layer_2 = Dense(second_layer_neurons, activation='relu', name='model7_layer_2')(model7_bn_1_5)
        model7_dropout_3 = Dropout(0.05)(model7_layer_2)
        model7_bn_2 = BatchNormalization()(model7_dropout_3)
        
        model7_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model7_layer_2_5')(model7_bn_2)
        model7_dropout_4 = Dropout(0.05)(model7_layer_2_5)
        model7_bn_2_5 = BatchNormalization()(model7_dropout_4)

        model7_layer_3 = Dense(third_layer_neurons, activation='relu', name='model7_layer_3')(model7_bn_2_5)
        model7_dropout_5 = Dropout(0.05)(model7_layer_3)
        model7_bn_3 = BatchNormalization()(model7_dropout_5)
        
        model7_out = Dense(class_number, activation='softmax', name='model7_out')(model7_bn_3)

        model7 = Model([model3.input, model5.input], model7_out)
        
        plot_model(model7, to_file='model.png', show_shapes=True, show_layer_names=True)
        return model7

    if LUT == 7:

        # MODEL 1
        model1_in_1 = Input(shape=(1,),name='input1')
        model1_in_2 = Input(shape=(1,),name='input2')
        model_1_concatenated = Concatenate()([model1_in_1, model1_in_2])

        
        model1_layer_1 = Dense(first_layer_neurons, activation='relu', name='model1_layer_1')(model_1_concatenated)
        model1_dropout_1 = Dropout(0.05)(model1_layer_1)
        model1_bn_1 = BatchNormalization()(model1_dropout_1)
        
        model1_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model1_layer_1_5')(model1_bn_1)
        model1_dropout_2 = Dropout(0.05)(model1_layer_1_5)
        model1_bn_1_5 = BatchNormalization()(model1_dropout_2)
        
        model1_layer_2 = Dense(second_layer_neurons, activation='relu', name='model1_layer_2')(model1_bn_1_5)
        model1_dropout_3 = Dropout(0.05)(model1_layer_2)
        model1_bn_2 = BatchNormalization()(model1_dropout_3)

        model1_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model1_layer_2_5')(model1_bn_2)
        model1_dropout_4 = Dropout(0.05)(model1_layer_2_5)
        model1_bn_2_5 = BatchNormalization()(model1_dropout_4)
        
        model1_layer_3 = Dense(third_layer_neurons, activation='relu', name='model1_layer_3')(model1_bn_2_5)
        model1_dropout_5 = Dropout(0.05)(model1_layer_3)
        model1_bn_3 = BatchNormalization()(model1_dropout_5)

        
        model1_out = Dense(bitwidth, activation='relu', name='model1_out')(model1_bn_3)


        model1 = Model([model1_in_1,model1_in_2], model1_out)

        
        
        # MODEL 2
        model2_in_1 = Input(shape=(1,),name='input3')
        model2_in_2 = Input(shape=(1,),name='input4')
        model_2_concatenated = Concatenate()([model2_in_1, model2_in_2])

        
        model2_layer_1 = Dense(first_layer_neurons, activation='relu', name='model2_layer_1')(model_2_concatenated)
        model2_dropout_1 = Dropout(0.05)(model2_layer_1)
        model2_bn_1 = BatchNormalization()(model2_dropout_1)

        model2_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model2_layer_1_5')(model2_bn_1)
        model2_dropout_2 = Dropout(0.05)(model2_layer_1_5)
        model2_bn_1_5 = BatchNormalization()(model2_dropout_2)


        model2_layer_2 = Dense(second_layer_neurons, activation='relu', name='model2_layer_2')(model2_bn_1_5)
        model2_dropout_3 = Dropout(0.05)(model2_layer_2)
        model2_bn_2 = BatchNormalization()(model2_dropout_3)
        
        model2_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model2_layer_2_5')(model2_bn_2)
        model2_dropout_4 = Dropout(0.05)(model2_layer_2_5)
        model2_bn_2_5 = BatchNormalization()(model2_dropout_4)

        model2_layer_3 = Dense(third_layer_neurons, activation='relu', name='model2_layer_3')(model2_bn_2_5)
        model2_dropout_5 = Dropout(0.05)(model2_layer_3)
        model2_bn_3 = BatchNormalization()(model2_dropout_5)

        model2_out = Dense(bitwidth, activation='relu', name='model2_out')(model2_bn_3)

        model2 = Model([model2_in_1,model2_in_2], model2_out)

        
        
         # MODEL 3
        model3_in_1 = Input(shape=(1,),name='input5')
        model3_in_2 = Input(shape=(1,),name='input6')
        model_3_concatenated = Concatenate()([model3_in_1, model3_in_2])

        
        model3_layer_1 = Dense(first_layer_neurons, activation='relu', name='model3_layer_1')(model_3_concatenated)
        model3_dropout_1 = Dropout(0.05)(model3_layer_1)
        model3_bn_1 = BatchNormalization()(model3_dropout_1)


        model3_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model3_layer_1_5')(model3_bn_1)
        model3_dropout_2 = Dropout(0.05)(model3_layer_1_5)
        model3_bn_1_5 = BatchNormalization()(model3_dropout_2)


        model3_layer_2 = Dense(second_layer_neurons, activation='relu', name='model3_layer_2')(model3_bn_1_5)
        model3_dropout_3 = Dropout(0.05)(model3_layer_2)
        model3_bn_2 = BatchNormalization()(model3_dropout_3)
        
        model3_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model3_layer_2_5')(model3_bn_2)
        model3_dropout_4 = Dropout(0.05)(model3_layer_2_5)
        model3_bn_2_5 = BatchNormalization()(model3_dropout_4)


        model3_layer_3 = Dense(third_layer_neurons, activation='relu', name='model3_layer_3')(model3_bn_2_5)
        model3_dropout_5 = Dropout(0.05)(model3_layer_3)
        model3_bn_3 = BatchNormalization()(model3_dropout_5)

        model3_out = Dense(bitwidth, activation='relu', name='model3_out')(model3_bn_3)

        model3 = Model([model3_in_1,model3_in_2], model3_out)

        
        
        # MODEL 4
        model4_in_1 = Input(shape=(1,),name='input7')
        model4_in_2 = Input(shape=(1,),name='input8')
        model_4_concatenated = Concatenate()([model4_in_1, model4_in_2])

        
        model4_layer_1 = Dense(first_layer_neurons, activation='relu', name='model4_layer_1')(model_4_concatenated)
        model4_dropout_1 = Dropout(0.05)(model4_layer_1)
        model4_bn_1 = BatchNormalization()(model4_dropout_1)

        model4_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model4_layer_1_5')(model4_bn_1)
        model4_dropout_2 = Dropout(0.05)(model4_layer_1_5)
        model4_bn_1_5 = BatchNormalization()(model4_dropout_2)


        model4_layer_2 = Dense(second_layer_neurons, activation='relu', name='model4_layer_2')(model4_bn_1_5)
        model4_dropout_3 = Dropout(0.05)(model4_layer_2)
        model4_bn_2 = BatchNormalization()(model4_dropout_3)
        

        model4_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model4_layer_2_5')(model4_bn_2)
        model4_dropout_4 = Dropout(0.05)(model4_layer_2_5)
        model4_bn_2_5 = BatchNormalization()(model4_dropout_4)


        model4_layer_3 = Dense(third_layer_neurons, activation='relu', name='model4_layer_3')(model4_bn_2_5)
        model4_dropout_5 = Dropout(0.05)(model4_layer_3)
        model4_bn_3 = BatchNormalization()(model4_dropout_5)

        model4_out = Dense(bitwidth, activation='relu', name='model4_out')(model4_bn_3)

        model4 = Model([model4_in_1,model4_in_2], model4_out)


        
        # SECOND LAYER MODELS
        # MODEL 5
        model5_in_1 = model1.output
        model5_in_2 = model2.output
        model_5_concatenated = Concatenate()([model5_in_1, model5_in_2])
        
        model5_layer_1 = Dense(first_layer_neurons, activation='relu', name='model5_layer_1')(model_5_concatenated)
        model5_dropout_1 = Dropout(0.05)(model5_layer_1)
        model5_bn_1  = BatchNormalization()(model5_dropout_1)


        model5_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model5_layer_1_5')(model5_bn_1)
        model5_dropout_2 = Dropout(0.05)(model5_layer_1_5)
        model5_bn_1_5 = BatchNormalization()(model5_dropout_2)
        
        model5_layer_2 = Dense(second_layer_neurons, activation='relu', name='model5_layer_2')(model5_bn_1_5)
        model5_dropout_3 = Dropout(0.05)(model5_layer_2)
        model5_bn_2 = BatchNormalization()(model5_dropout_3)
        
        model5_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model5_layer_2_5')(model5_bn_2)
        model5_dropout_4 = Dropout(0.05)(model5_layer_2_5)
        model5_bn_2_5 = BatchNormalization()(model5_dropout_4)

        model5_layer_3 = Dense(third_layer_neurons, activation='relu', name='model5_layer_3')(model5_bn_2_5)
        model5_dropout_5 = Dropout(0.05)(model5_layer_3)
        model5_bn_3 = BatchNormalization()(model5_dropout_5)
        
        model5_out = Dense(bitwidth, activation='relu', name='model5_out')(model5_bn_3)

        model5 = Model([model1.input, model2.input], model5_out)

        # MODEL 6
        model6_in_1 = model3.output
        model6_in_2 = model4.output
        model_6_concatenated = Concatenate()([model6_in_1, model6_in_2])
        

        model6_layer_1 = Dense(first_layer_neurons, activation='relu', name='model6_layer_1')(model_6_concatenated)
        model6_dropout_1 = Dropout(0.05)(model6_layer_1)
        model6_bn_1  = BatchNormalization()(model6_dropout_1)

        
        model6_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model6_layer_1_5')(model6_bn_1)
        model6_dropout_2 = Dropout(0.05)(model6_layer_1_5)
        model6_bn_1_5 = BatchNormalization()(model6_dropout_2)

        
        model6_layer_2 = Dense(second_layer_neurons, activation='relu', name='model6_layer_2')(model6_bn_1_5)
        model6_dropout_3 = Dropout(0.05)(model6_layer_2)
        model6_bn_2 = BatchNormalization()(model6_dropout_3)

        model6_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model6_layer_2_5')(model6_bn_2)
        model6_dropout_4 = Dropout(0.05)(model6_layer_2_5)
        model6_bn_2_5 = BatchNormalization()(model6_dropout_4)

        model6_layer_3 = Dense(third_layer_neurons, activation='relu', name='model6_layer_3')(model6_bn_2_5)
        model6_dropout_5 = Dropout(0.05)(model6_layer_3)
        model6_bn_3 = BatchNormalization()(model6_dropout_5)
        
        model6_out = Dense(bitwidth, activation='relu', name='model6_out')(model6_bn_3)

        model6 = Model([model3.input, model4.input], model6_out)


        # THIRD LAYER

        

        # FOURTH LAYER MODELS

        # MODEL 7

        model7_in_1 = model5.output
        model7_in_2 = model6.output
        model_7_concatenated = Concatenate()([model7_in_1, model7_in_2])
        

        
        model7_layer_1 = Dense(first_layer_neurons, activation='relu', name='model7_layer_1')(model_7_concatenated)
        model7_dropout_1 = Dropout(0.05)(model7_layer_1)
        model7_bn_1  = BatchNormalization()(model7_dropout_1)

        model7_layer_1_5 = Dense(layer_1_5_neurons, activation='relu', name='model7_layer_1_5')(model7_bn_1)
        model7_dropout_2 = Dropout(0.05)(model7_layer_1_5)
        model7_bn_1_5 = BatchNormalization()(model7_dropout_2)

        model7_layer_2 = Dense(second_layer_neurons, activation='relu', name='model7_layer_2')(model7_bn_1_5)
        model7_dropout_3 = Dropout(0.05)(model7_layer_2)
        model7_bn_2 = BatchNormalization()(model7_dropout_3)
        
        model7_layer_2_5 = Dense(layer_2_5_neurons, activation='relu', name='model7_layer_2_5')(model7_bn_2)
        model7_dropout_4 = Dropout(0.05)(model7_layer_2_5)
        model7_bn_2_5 = BatchNormalization()(model7_dropout_4)

        model7_layer_3 = Dense(third_layer_neurons, activation='relu', name='model7_layer_3')(model7_bn_2_5)
        model7_dropout_5 = Dropout(0.05)(model7_layer_3)
        model7_bn_3 = BatchNormalization()(model7_dropout_5)
        
        model7_out = Dense(class_number, activation='softmax', name='model7_out')(model7_bn_3)

        #model7_out_quant = lq.quantizers.DoReFa(k_bit=bitwidth, mode="activations",name='model7_out_quant')(model7_out)

        model7 = Model([model5.input, model6.input], model7_out)
        
        plot_model(model7, to_file='model.png', show_shapes=True, show_layer_names=True)
        return model7