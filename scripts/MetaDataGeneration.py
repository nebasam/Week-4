# Meta Data Generation
import os
import glob
import traceback
import pandas as pd
import librosa

class Create_meta_data():
    def __init__(self, transcription_path:str, audio_path: str, audio_extension: str = 'wav') -> None:
        self.meta_data = pd.DataFrame()
        self.transcription_path = transcription_path
        self.audio_path = audio_path
        self.audio_extension = audio_extension
        self.filenameAndTranscription = {}
        self.filenameAndPath = {}
        
    def load_transcription(self):
        """
        This function loads the transcription file, processes it to obtain
        the audio file name and corresponding transcription. Values are stored in 
        a dict.

        Args:
            self.transcription_path (str): Path to the transcription file
            self.filenameAndTranscription (dict): An Empty Dict where the audio file name: audio transcription will be stored

        """
        filename_to_transcription = {}
        try:
            with open (self.transcription_path, encoding="utf-8")as f:
                f.readline()
                for line in f:
                    file_name=line.split("	")[0]
                    transcription=line.split("	")[1].replace('\n','')
                    filename_to_transcription[file_name]=transcription
            self.filenameAndTranscription = filename_to_transcription
        except Exception:
            print("Error Occured")
            traceback.print_exc()
            
    def get_filenameAndTranscription(self) -> dict:
        """This function returns a dictionary of file name ,audio transcription pairs

        Returns:
            dict: key=file name : value=audio transcription
        """
        try:
            self.load_transcription()
            return self.filenameAndTranscription
        except Exception:
            print("Error Occured")
            traceback.print_exc()
    
    def load_audio_filepaths(self) -> dict:
        """
        This function loads the abs file path of each audio file and 
        stores them in a dict with format; audio file name: path to file.
        
        Returns:
            [dict]: Dict that contains audio file name : absolute file path
        """
        audiofileAndPath = {}
        try:
            for filename in glob.iglob(os.path.join(self.audio_path,'**/',f'*{self.audio_extension}'),recursive = True):
                path = filename
                file_name = os.path.splitext(os.path.split(filename)[1])[0]
                audiofileAndPath[file_name] = path
            self.filenameAndPath = audiofileAndPath
        except Exception:
            print("Error Occured")
            traceback.print_exc()  
        return audiofileAndPath
    
    def get_filenameAndPath(self) -> dict:
        """This function returns a dictionary of file name ,file path pairs

        Returns:
            dict: key=file name : value=absolute file path
        """
        try:
            self.load_audio_filepaths()
            return self.filenameAndPath
        except Exception:
            print("Error Occured")
            traceback.print_exc()
    
    def create_meta_data(self):
        """
            This functions creates meta data from the audio and transcription files.
            target :  audio transcription
            filenames:  audio filename
            paths: audio file path
            duration_of_recording:audio file duration
            channels: audio channels
            sampleRates: audio sampling rate            
        """
        target=[]
        filenames=[]
        paths=[]
        duration_of_recordings=[]
        channels=[]
        sampleRates=[]
        
        for filename,transcription in self.filenameAndTranscription.items():
            try:
                target.append(transcription)
                filenames.append(filename)
                paths.append(self.filenameAndPath[filename])
                audio , samplerate = librosa.load(self.filenameAndPath[filename],sr=None,mono=False)
                sampleRates.append(samplerate)
                duration_of_recordings.append(float(audio.shape[0]/samplerate))
                channels.append(len(audio.shape))
            except KeyError:
                print(f"Error occured couldn't find path for {filename}")
                continue
            except Exception:
                print(f"Error Occured for file {filename}")
                traceback.print_exc()
                continue  
        
        data=pd.DataFrame({'file': filenames,'text': target,'path':paths,
                           'sample_rate':sampleRates,"channels":channels, 'duration':duration_of_recordings})
        
        self.meta_data = data
    
    def generate_meta_data(self) -> pd.DataFrame:
        """
            This functions generate meta data from the audio and transcription files

        Returns:
            [DataFrame]: The dataframe contains columns: files = audio filename, text = audio transcription,
                        path = audio file path, sample_rate = audio sampling rate, channels = audio channels,
                        duration: audio file duration 
            
        """
        try:
            self.load_transcription()
            self.load_audio_filepaths()
            self.create_meta_data()
            return self.meta_data
        except Exception:
            print("Error Occured while generating meta data")
            traceback.print_exc()   
    
    def meta_data_to_json(self, path: str = 'metaData.json') -> None:
        """This function saves the meta data as a json file

        Args:
            path (str, optional): File path to save meta data. Defaults to 'metaData.json'.
        """
        try:
            self.generate_meta_data()
            self.meta.to_json(path)
            print('Meta data json file save successful! ')
        except Exception:
            print("Error Occured while saving file")
            traceback.print_exc()
        
        
        
                
            
        
            
            
        
        
    
    

