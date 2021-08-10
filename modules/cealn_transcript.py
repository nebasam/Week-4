import re
import string

class CleanTrans():
    '''Clas for cleaning transcrits'''
    def  __init__(self, df, trans_col):
       self.df = df
       self.tran_col = trans_col
    
    def rm_punct_digit_mthd(self, text):
        regex = re.compile('[%s]' % re.escape(string.punctuation+string.digits))
        processed = regex.sub(' ', text)
        return processed.lower()

    def fill_unk(self, unk_val='<UNK>', fill_val=None, output=False):
        df = self.df
        trans_col = self.tran_col
        df[trans_col] = df[trans_col].apply(lambda x: fill_val if x==unk_val else x)
        self.df = df
        if output:
            return df

    def rm_punct_digit(self, output=False):
        df = self.df
        trans_col = self.tran_col
        df[trans_col] = [self.rm_punct_digit_mthd(t) for t in df[trans_col]]
        self.df = df
        if output:
            return df

    def run_all(self):
        self.fill_unk()
        df = self.rm_punct_digit(output=True)
        df.dropna(axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df