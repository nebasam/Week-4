
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers  import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

from ctc_utils import ctc_lambda_func, add_ctc_loss, cnn_output_length

from tensorflow.keras.layers import (Input, Lambda)
from tensorflow.keras.optimizers import SGD




def simple_rnn(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    # plot_model(model, to_file='models/model_1.png')
    return model



def model_2(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2, 
    cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate)(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    # plot_model(model, to_file='models/model_2.png', show_shapes=True)
    return model



def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True,  name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    
   
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    # model.output_length = lambda x: x
    print(model.summary())
    
    return model



def bidirectional_rnn_model_gpu(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units,
                                  return_sequences=True,
                                  implementation=2,
                                  name='bi_rnn',
                                  activation = 'tanh',
                                recurrent_activation = 'sigmoid',
                                recurrent_dropout = 0,
                                unroll = False,
                                use_bias = True,
                                 ))(input_data)

                                 
    bn_bidir_rnn = BatchNormalization( name='norm_bidir_rnn')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units,
                                  activation='softmax',
                                  return_sequences=True,
                                  implementation=2,
                                  name='bi_rnn'
                                 ))(input_data)
    bn_bidir_rnn = BatchNormalization( name='norm_bidir_rnn')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# def model_3():


def model_3(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2, 
    cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = Bidirectional(cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate))(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = Bidirectional(cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate))(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = Bidirectional(cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate))(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = Bidirectional(cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn'))(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    # plot_model(model, to_file='models/model_2.png', show_shapes=True)
    return model



def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
   
    # TODO: Add recurrent layers, each with batch normalization
    if recur_layers>1:
            #first input layer
        rnn=GRU(output_dim, return_sequences=True, name='rnn')(input_data)
        bn_layer = BatchNormalization(name='bt_rnn')(rnn)
        
        for i in range(recur_layers - 1):
            rnn=GRU(output_dim, return_sequences=True, name='rnn_{}'.format(i))(bn_layer)
            bn_layer = BatchNormalization(name='bt_rnn_{}'.format(i))(rnn)
    else:
        rnn=GRU(output_dim, return_sequences=True)(input_data)
        bn_layer = BatchNormalization(name='bt_rnn')(rnn)
           
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model