import unittest
from scripts.MetaCreate import MetaCreate
import pickle

meta_load_transcription_output = pickle.load(open('test_data/meta_load_transcription_output','rb'))
meta_load_audio_file_paths_output = pickle.load(open('test_data/meta_load_audio_file_paths_output','rb'))
meta_data_output = pickle.load(open('test_data/meta_data_output','rb'))

transcription_path = r'test_data/text.txt'
audio_path = 'test_data/'
audio_extension='wav'
separater='\t'


class TestMeta(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = MetaCreate(transcription_path=transcription_path, audio_path=audio_path, audio_extension=audio_extension, separater=separater)        

    def test_load_transcription(self):
        self.assertEqual(self.meta.load_transcription(), meta_load_transcription_output)

    '''def test_load_audio_file_paths(self):
        self.assertEqual(self.meta.load_audio_file_paths()['SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part100'], meta_load_audio_file_paths_output)'''

    def test_meta_data(self):
        print(self.meta.meta_data())
        self.assertEqual(self.meta.meta_data(), meta_data_output)'''

    def test_meta(self):
        self.assertRaises(TypeError, MetaCreate())

    '''def test_meta_to_json(self):
        self.assertRaises(NameError, self.meta.meta_to_json(),{'path':2})'''

if __name__ == '__main__':
	unittest.main()
