from AudioManipulator import AudioManipulator
import librosa
import numpy as np


def pad_audio_files(paths,size=98400):
    if type(size) not in [int]:
        raise TypeError("size is not in the required format. Must be an integer")
    audios=[]
    counter = 0
    for i in paths:
        if type(i) not in [str]:
            raise TypeError("An element in position {} in path not in the required format. Must be a string".format(counter+1))
        audio_data, sr=librosa.load(i,sr=None,mono=False)
        manipulator=AudioManipulator(audio_data,sr=sr)
        audio_data=manipulator.pad(size)
        audios.append(audio_data)
        counter += 1
    return audios

def featurize(audios,sr):
    if type(audios) not in [np.array]:
        raise TypeError("Audios is not in the required format. Must be a numpy array")
    if type(sr) not in [int]:
        raise TypeError("Sample rate (sr) is not in the required format. Must be an integer")
    feautures=[]
    for i in audios:
        manipulator=AudioManipulator(i,sr=sr)
        mfcc=manipulator.generate_MFCC()
        feautures.append(mfcc)
    return feautures

def pitch_audio(audios,pitch,sr):
    if type(audios) not in [np.array]:
        raise TypeError("Audios is not in the required format. Must be a numpy array")
    if type(pitch) not in [int, float]:
        raise TypeError("Pitch_factor is not in the required format. Must be an integer or a float")
    if type(sr) not in [int]:
        raise TypeError("Sample rate (sr) is not in the required format. Must be an integer")
    feautures=[]
    for i in audios:
        manipulator=AudioManipulator(i,sr=sr)
        mfcc=manipulator.pitch(pitch)
        feautures.append(mfcc)
    return feautures
def shift(audios,direction,amount,sr):
    if type(amount) not in [int]:
        raise TypeError("amount is not in the required format. Must be an integer")
    if type(direction) not in [str]:
        raise TypeError("direction is not in the required format. Must be a string")
    if type(audios) not in [np.array]:
        raise TypeError("Audios is not in the required format. Must be a numpy array")
    if type(sr) not in [int]:
        raise TypeError("Sample rate (sr) is not in the required format. Must be an integer")
    feautures=[]
    for i in audios:
        manipulator=AudioManipulator(i,sr=sr)
        mfcc=manipulator.shift_to(amount,direction)
        feautures.append(mfcc)
    return feautures