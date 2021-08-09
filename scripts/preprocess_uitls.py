from AudioManipulator import AudioManipulator
import librosa


def pad_audio_files(paths,size=98400):
    audios=[]
    for i in paths:
        audio_data, sr=librosa.load(i,sr=None,mono=False)
        manipulator=AudioManipulator(audio_data,sr=sr)
        audio_data=manipulator.pad(size)
        audios.append(audio_data)
    return audios

def featurize(audios,sr):
    feautures=[]
    for i in audios:
        manipulator=AudioManipulator(i,sr=sr)
        mfcc=manipulator.generate_MFCC()
        feautures.append(mfcc)
    return feautures