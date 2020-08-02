import os
import numpy as np
import tensorflow as tf

#useful dicts and stuff
pos_dict = {"a": "adj", "r": "adv", "n": "noun", "v": "verb"}
punc_list = [",", ".", ";", ":", "?", "!", "\n", "\""]

#parsing crappy csv
def parse_line(string):
    string = string.strip().strip("\"")
    word = string[:string.find("(")]
    pos = string[string.find("("):string.find("(")+1]
    definition = string[string.find(")")+1:]
    return word.strip(), pos.strip(), definition.strip()

#parsing WordNet
def get_info(string):
    '''
    takes in a line from wordnet, and then gives
    (word, definitions(list), pos)
    '''
    broken = string.split(" ")
    word = broken[4]
    pos = broken[2]
    definitions = string[string.find("|") + 1:]
    defs = []
    for definition in definitions.split(";"):
        definition = definition.strip()
        if not definition:
            continue
        if definition[0] == "\"":
            continue
        while "(" in definition and ")" in definition:
            definition = definition[:definition.find("(")].strip() + " " + definition[definition.find(")") + 1:].strip()
        if not definition:
            continue
        defs.append(definition)
    return word, defs, pos

def read_dir(directory):
    '''takes in 1 layer deep directory, and walks through it to get all files and parses each file, 
    returning a list of tuples
    (word, definitions(list), pos)
    '''
    filenames = []
    data = []
    for root, _, files in os.walk(directory):
        files = files
    for filey in files:
        filenames.append(root+"/"+filey)
    for filename in filenames:
        with open(filename) as fileboi:
            for line in fileboi:
                data.append(get_info(line))
    return data

def get_definitions(data):
    ''' takes in a list in the format
    [(word, definitions(list), pos)]
    
    and gives back a list of definitions vectors (comprised of words), along with the longest word wise definition
    ([defs], max_length, [wordmap])
    '''
    definitions = []
    wordmap = []
    max_length = 0
    for words in data:
        for definition in words[1]:
            def_list = definition.split()
            definition_vector = []
            for word in def_list:
                word = process_word(word)
                if word:
                    definition_vector.append(word)
            if len(def_list) > max_length:
                max_length = len(def_list)
            if definition_vector:
                definitions.append(definition_vector)
                wordmap.append(words[0])
    return definitions, max_length, wordmap

def process_word(word):
    for punc in punc_list:
        word = word.replace(punc, "")
    word = word.strip()
    return word

#get dictionaries with word -> int mapping
def get_word_dicts(definitions, idx=1):
    '''takes in a list of definition lists, and returns the integer mapping of words (INDEX STARTS AT 1)
    returns word2num (dict), num2word (dict)
    '''
    word2num = {}
    for words in definitions:
        for word in words:
            if word in word2num:
                continue
            else:
                word2num[word] = idx
                idx += 1
    num2word = {num: word for word, num in word2num.items()}

    return word2num, num2word

def convert_word2int(definitions, word2num):
    return [[word2num[word] for word in definition] for definition in definitions]

def defs_to_np(definitions, max_length, padding_num = 0):
    '''takes in definitions, and pads them to max length with (default) 0s
    returns numpy array of padded definitions
    '''
    output = []
    for definition in definitions:
        length = len(definition)
        diff = max_length - length
        output.append(definition + [padding_num for i in range(diff)])
    output = np.array(output, dtype="int32")
    #return np.reshape(output, (output.shape + (1,))
    return output

def defs2str(definitions):
    '''converts a list of list of words to a list of strings'''
    return [" ".join(each) for each in definitions]

def to_one_hot(x):
    vocab_size = 46948 + 1
    #output = tf.zeros((x.shape)+(vocab_size,))
    #mask = np.array(x) > 0
    label = tf.one_hot(x, vocab_size)
    return x, label[:, 1:]

# creating embeddings








'''
class LSTMAutoencoder(Model):
    def __init__(self, vocab_size, max_length):
        super(LSTMAutoencoder, self).__init__()
        #self.inputy = Input(shape=(None,), dtype="int32")
        self.embedding = Embedding(input_dim=50000, output_dim=64, input_length=max_length, mask_zero=True)
        self.encodingLSTM1 = LSTM(32, return_sequences=True)
        self.encodingLSTM2 = LSTM(16)
        self.repeatlayer = RepeatVector(max_length)
        self.decodingLSTM1 = LSTM(16, return_sequences=True)
        self.decodingLSTM2 = LSTM(32, return_sequences=True)
        self.denseboi = TimeDistributed(Dense(100, activation="relu"))
        self.finalDense = TimeDistributed(Dense(vocab_size, activation="softmax"))

    def call(self, inputs):
        #x = self.inputy(inputs)
        
        x = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        x = self.encodingLSTM1(x, mask=mask)
        x = self.encodingLSTM2(x, mask=mask)
        x = self.repeatlayer(x)
        x = self.decodingLSTM1(x, mask=mask)
        x = self.decodingLSTM2(x, mask=mask)
        x = self.denseboi(x)
        x = self.finalDense(x)
        x = tf.math.argmax(x, axis=2, output_type=tf.dtypes.int32) + 1
        x = self.pad_output(x, max_length)
        x = tf.cast(x, dtype=tf.dtypes.float32)
        print(x)
        return x
    
    def pad_output(self, x, max_length):
        pad = [[0, 0], [0, max_length-x.shape[1]]]
        return tf.pad(x, pad, mode="CONSTANT")
'''