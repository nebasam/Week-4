import librosa
import numpy as np

class AudioManipulator():
    def __init__(self,audio,sr):
        if type(audio) not in [np.array]:
            raise TypeError("Audio is not in the required format")
        if type(sr) not in [int]:
            raise TypeError("Sample rate (sr) is not in the required format")
        self.audio=audio
        self.sr=sr
        self.stereo=bool(len(self.audio.shape)-1)
    
    def cut(self,audio,length):
        if type(audio) not in [np.array]:
            raise TypeError("Audio is not in the required format")
        if type(length) not in [int]:
            raise TypeError("Length is not in the required format")
        trunc=[]
        if self.stereo:
            trunc.append(audio[0][:length])
            trunc.append(audio[1][:length])
        else:
            trunc=audio[:length]
        self.audio=np.array(trunc)
        return self.audio
    
    def pad(self,length):
        if type(length) not in [int]:
            raise TypeError("Length is not in the required format")
        if self.stereo:
            axis=1
        else:
            axis=0
        pad_size = length - self.audio.shape[axis]

        if pad_size <= 0:
            return self.audio

        npad = [(0, 0)] * self.audio.ndim
        npad[axis] = (0, pad_size)
    
        self.audio=np.pad(self.audio, pad_width=npad, mode='constant', constant_values=0)
        return self.audio

    def pitch(self,pitch_factor):
        if type(pitch_factor) not in [int, float]:
            raise TypeError("Pitch_factor is not in the required format")
        self.audio= librosa.effects.pitch_shift(self.audio, self.sr, n_steps=pitch_factor)
        return self.audio

    def shift_to(self, shift_max, shift_direction):
        if type(shift_max) not in [int]:
            raise TypeError("Shift_max is not in the required format")
        if type(shift_direction) not in [str]:
            raise TypeError("Shift_direction is not in the required format")
        shift = np.random.randint(self.sr * shift_max)
        if shift_direction == 'right':
            shift = shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        elif shift_direction == 'left':
            shift=-shift
        augmented_data = np.roll(self.audio, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        self.audio=augmented_data
        return self.audio

    def generate_MFCC(self,n_mfcc=26):
        if type(n_mfcc) not in [int]:
            raise TypeError("n_mfcc is not in the required format")
        mfccs = librosa.feature.mfcc(self.audio, self.sr,n_mfcc=n_mfcc)
        return mfccs