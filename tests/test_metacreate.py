import unittest
from scripts.MetaCreate import MetaCreate
import pickle

meta_load_transcription_output = pickle.load(open('../test_data/meta_load_transcription_output','rb'))
meta_load_audio_file_paths_output = pickle.load(open('../test_data/meta_load_audio_file_paths_output','rb'))
meta_data_output = pickle.load(open('../test_data/meta_data_output','rb'))

transcription_path = r'../test_data/text.txt'
audio_path = '../test_data/'
audio_extension='wav'
separater='\t'


class TestMeta(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = MetaCreate(transcription_path=transcription_path, audio_path=audio_path, audio_extension=audio_extension, separater=separater)        

    def test_load_transcription(self):
        self.assertEqual(self.meta.load_transcription(), meta_load_transcription_output)

    def test_load_audio_file_paths(self):
        self.assertEqual(self.meta.load_audio_file_paths(), meta_load_audio_file_paths_output)

    def test_meta_data(self):
        self.assertEqual(self.meta.meta_data(), meta_data_output)

    def test_meta_to_json(self):
        self.assertRaises(TypeError, self.meta.meta_to_json(), 1)

    def test_meta(self):
        self.assertRaises(TypeError, MetaCreate(), 2)

if __name__ == '__main__':
	unittest.main()