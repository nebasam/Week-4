import random
import torch
import torchaudio
from torchaudio import transforms
#from IPython.display import Audio

class PrepSound():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  def __init__(self, audio_file, path):
    self.audio = audio_file
    self.aud_path = path + '/' + audio_file
  

  def open(self, output=False):
    samples, sr = torchaudio.load(self.aud_path)
    self.samples = samples
    self.sample_rate = sr
    if output:
      return (samples, sr)

  # ----------------------------
  # Convert the given audio to the desired number of channels
  # ----------------------------
  def rechannel(self, new_channel, output=False):
    samples = self.samples
    sr = self.sample_rate

    if (samples.shape[0] == new_channel):
      pass

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      samples = samples[:1, :]
      self.samples = samples
    else:
      # Convert from mono to stereo by duplicating the first channel
      self.samples = torch.cat([samples, samples])
      self.samples = samples
    if output:
      return ((samples, sr))

  # ----------------------------
  # Since Resample applies to a single channel, we resample one channel at a time
  # ----------------------------
  
  def resample(self,newsr, output=False):
    samples = self.samples
    sr = self.sample_rate

    if (sr == newsr):
      pass

    num_channels = samples.shape[0]
    # Resample first channel
    resamp = torchaudio.transforms.Resample(sr, newsr)(samples[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      resamp_dup = torchaudio.transforms.Resample(sr, newsr)(samples[1:,:])
      resamp = torch.cat([resamp, resamp_dup])
    self.samples = resamp
    self.sample_rate = newsr
    if output:
      return ((resamp, newsr))
    

  # ----------------------------
  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  # ----------------------------
  def pad_trunc(self, max_ms, output=False):
    samples = self.samples
    sr = self.sample_rate
    num_rows, samples_len = samples.shape
    max_len = sr//1000 * max_ms

    if (samples_len > max_len):
      # Truncate the signal to the given length
      samples = samples[:,:max_len]

    elif (samples_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - samples_len)
      pad_end_len = max_len - samples_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      samples = torch.cat((pad_begin, samples, pad_end), 1)
      
    self.samples = samples
    self.sample_rate = sr
    if output:
      return ((samples, sr))

  # ----------------------------
  # Generate a Spectrogram
  # ----------------------------
  def spectro_gram(self, n_mels=64, n_fft=1024, hop_len=None, output=False):
    samples = self.samples
    sr = self.sample_rate
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(samples)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    if output:
      return (spec)