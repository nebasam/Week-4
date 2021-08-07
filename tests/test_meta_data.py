import unittest
import pandas as pd
import sys, os 
sys.path.append(os.path.abspath(os.path.join('scripts')))
from MetaDataGeneration import Create_meta_data

sys.path.append(os.path.abspath(os.path.join('../data/test')))

trans_path = 'test.txt'
audio_path = 'wav5'

meta = Create_meta_data(transcription_path=trans_path, audio_path=audio_path,
                        audio_extension='.wav')

class TestMeta_Data(unittest.TestCase):
    """
    A class for unit-testing functiosns in the MetaDataGeneration.py file.
    """
    def setUp(self):
        self.audio_extension = '.wav'
        return self.audio_extension
         
    def test_get_filenameAndTranscription(self):
        self.assertIsInstance( meta.get_filenameAndTranscription(), dict)



if __name__ == '__main__':
	unittest.main()

