import numpy as np
from utils import text_to_int_sequence


class Batch():
    def __init__(self,features,feature_valid,texts,texts_valid,minibatch_size):
        self.minibatch_size=minibatch_size
        self.features=features
        self.current_index=0
        self.current_valid_index=0
        self.texts=texts
        self.features_valid=feature_valid
        self.texts_valid=texts_valid
    def getBatch(self,partition):
        if partition == 'train':
            feautre_list = self.features
            cur_index = self.current_index
            texts = self.texts
        elif partition == 'valid':
            feautre_list = self.features_valid
            cur_index = self.current_valid_index
            texts = self.texts_valid
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")
                
        features=feautre_list[cur_index:cur_index+self.minibatch_size]
        texts=texts[cur_index:cur_index+self.minibatch_size]


        max_length = max([features[i].shape[1] 
            for i in range(0, self.minibatch_size)])

        # print(self.minibatch_size,cur_index,len(texts),len(self.texts_valid),self.current_valid_index)
        

        max_string_length = max([len(texts[ i]) 
            for i in range(0, self.minibatch_size)])

        # initialize the arrays
        # X_data = np.zeros([self.minibatch_size, 26, max_length])
        X_data = np.zeros([self.minibatch_size,  max_length,26])
        
        labels = np.ones([self.minibatch_size, max_string_length]) * 28
        input_length = np.zeros([self.minibatch_size, 1])
        label_length = np.zeros([self.minibatch_size, 1])
        
        # print(labels.shape)
        # print(max_string_length)

        for i in range(0, self.minibatch_size):
            # calculate X_data & input_length
            feat = features[i]
           
            # j = feat.shape[1]
            
            # l = np.array(text_to_int_sequence(texts[i])).shape[0]
            # if j < l:
            #     print(True)
            # else:
            #     print(j,l)
            
            input_length[i] = feat.shape[1]
            # X_data[i,:,:feat.shape[1]] = feat
            # X_data[i, :feat.shape[1], :] = feat.T
            X_data[i, :feat.shape[1], :] = feat.T

            # print(feat.T.shape)
            # calculate labels & label_length)
            label = np.array(text_to_int_sequence(texts[ i])) 
            
            # label=np.pad(label,max_string_length-len(label),'constant')

            labels[i, :len(label)] = label
            # print(label.shape)
            #label_length[i] = len(label)
            label_length[i] = len(label)
            # print(label_length[i],len(label))
            
        outputs = {'ctc': np.zeros([self.minibatch_size])}
        inputs = {'the_input': X_data, 
                'the_labels': labels, 
                'input_length': input_length, 
                'label_length': label_length 
                }
        
        

        return (inputs, outputs)

    def next_train(self):
        while True:
            ret = self.getBatch('train')
            if self.current_index+ self.minibatch_size >= len(self.texts) - self.minibatch_size:
                self.current_index = 0
                # self.suffle()
            
            self.current_index += self.minibatch_size
            yield ret 

    def next_valid(self):
        """ Obtain a batch of validation data
        """
        while True:
            ret = self.getBatch('valid')
            
            if self.current_valid_index+ self.minibatch_size >= len(self.texts_valid) - self.minibatch_size:
                self.current_valid_index = 0
                # self.shuffle_data_by_partition('valid')
                
            self.current_valid_index += self.minibatch_size
            yield ret
        
    def shuffle_data(self,features, durations, texts):
        """ Shuffle the data (called after making a complete pass through 
            training or validation data during the training process)
        Params:
            audio_paths (list): Paths to audio clips
            durations (list): Durations of utterances for each audio clip
            texts (list): Sentences uttered in each audio clip
        """
        p = np.random.permutation(len(self.features))
        self.features = [self.features[i] for i in p] 
        self.texts = [self.texts[i] for i in p]
        # return audio_paths, durations, texts