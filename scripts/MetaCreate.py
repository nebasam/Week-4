import librosa
import pandas as pd
import traceback
import os

class MetaCreate():
    def __init__(self, transcription_path:str=r'../data/train/text', audio_path:str='../data/train',audio_extension='wav',separater='\t') -> None:
        self.meta=None
        self.transcription_path=transcription_path
        self.audio_path=audio_path
        self.audio_extension=audio_extension
        self.file_to_trancscript={}
        self.file_to_path={}
        self.separater=separater
    def load_transcription(self):
        name_to_text = {}
        try:
            with open (self.transcription_path, encoding="utf-8")as f:
                f.readline()
                for line in f:
                    line=line.strip()
                    ls=line.split(self.separater,1)
                    name_to_text[ls[0]]=ls[1]
            self.file_to_trancscript=name_to_text
        except FileNotFoundError:
            print(f"File {self.transcription_path} couldn't be found")
        except Exception:
            print("Error Occured")
            traceback.print_exc()
        return name_to_text
    def get_file_to_transcription(self):
        return self.file_to_trancscript

    def load_audio_file_paths(self):
        dict={}
        try:
            files = librosa.util.find_files(self.audio_path, ext=self.audio_extension, recurse=True)
            names = [os.path.splitext(os.path.basename(x))[0] for x in files]
            for i in range(0,len(files)):
                dict[names[i]]=os.path.relpath(files[i])
            self.file_to_path=dict
        except Exception:
            print("Error Occured")
            traceback.print_exc()
        return dict
    def get_file_to_path(self):
        return self.file_to_path
    def meta_data(self): 
        target=[]
        filenames=[]
        paths=[]
        duration_of_recordings=[]
        channels=[]
        sampleRates=[]
        for i in self.file_to_trancscript:
            try:
                target.append(self.file_to_trancscript[i])
                filenames.append(i)
                paths.append(self.file_to_path[i])

                audio, sampleRate = librosa.load(self.file_to_path[i],sr=None,mono=False)
                sampleRates.append(sampleRate)
                duration_of_recordings.append(float(audio.shape[0]/sampleRate))
                channels.append(len(audio.shape))
            except KeyError:
                print(f"Error occured couldn't find path for {i}")
                continue
            except Exception:
                print(f"Error Occured for file {i}")
                traceback.print_exc()
                continue
                
                
        data=pd.DataFrame({'file': filenames,'text': target,'path':paths,'sample_rate':sampleRates,"channels":channels, 'duration':duration_of_recordings})
        self.meta=data
        return data
    def generate_meta_data(self):
        self.load_audio_file_paths()
        self.load_transcription()
        self.meta_data()
        return self.meta
    def get_meta(self):
        return self.meta
    def meta_to_json(self,path='meta.json'):
        self.meta.to_json(path)