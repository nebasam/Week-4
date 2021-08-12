from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers  import  Input,Lambda
    


def ctc_lambda_func(args):
    if len(args) != 4:
        raise ValueError("The elements in args are not up to the required number. Must be 4 in number")
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [  input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if type(input_length) not in [int]:
        raise TypeError("input_length is not in the required format. Must be an integer")
    if type(filter_size) not in [int]:
        raise TypeError("filter_size is not in the required format. Must be an integer")
    if type(border_mode) not in [str]:
        raise TypeError("border_mode is not in the required format. Must be a string")
    if type(stride) not in [int]:
        raise TypeError("stride is not in the required format. Must be an integer")
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length 
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1

    return (output_length + stride -1 ) // stride


