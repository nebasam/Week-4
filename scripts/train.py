"""
Defines a functions for training a NN.
"""

from data_generator import AudioGenerator
from data_generator import make_audio_gen

import _pickle as pickle

from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda)
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   
import os