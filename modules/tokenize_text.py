from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TokenizerWrap(Tokenizer):
    def __init__(self, texts, padding, len_sent, filters, reverse=False):
        Tokenizer.__init__(self, filters=filters, char_level=True)

        self.len_sent = len_sent
         
        # https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do
        # creates the vocabulary index based on word frequency
        self.fit_on_texts(texts)

        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys()))
        
        # Transforms each text in texts to a sequence of integers
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
            truncating = 'post'

        # https://stackoverflow.com/questions/42943291/what-does-keras-io-preprocessing-sequence-pad-sequences-do
        # Ensure that all sequences in a list have the same length
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=len_sent,
                                           padding=padding,
                                           truncating=truncating
                                           )

    def token_to_word(self, token):
        '''Converts a given token into word'''
        # If token is 0 word is empty space since 0 is used for padding
        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        '''Converts  given tokens into words'''
        # If token is 0 word is not taken since 0 is used for padding
        words = [self.index_to_word[token] for token in tokens if token != 0]
        # Join the words to construct sentence
        text = "".join(words)
        return text

    def text_to_tokens(self, text, reverse=False, padding=False):
        '''Converts  given text into tokens'''
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            tokens = np.flip(tokens, axis=1)
            truncating = 'pre'
        else:
            truncating = 'post'

        if padding:
            tokens = pad_sequences(tokens,
                                   maxlen=self.len_sent,
                                   padding=truncating,
                                   truncating=truncating
                                   )
        return tokens
