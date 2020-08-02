import os
import numpy as np 
import utils2
import utils
import structs
import pickle
import tensorflow as tf

data_dir = "C:/Users/DS/Desktop/zhan8425_project/data_wordnet"
data = utils.read_dir(data_dir)
word_net = structs.Dictionary()
word_net.word2vec = pickle.load(open("normedword2vec512.p", "rb"))
word_net.char_list = pickle.load(open("charlist.p", "rb"))
word_net.punc_list = [",", ".", ";", ":", "?", "!", "\n", "\""]
word_net.generating_model = tf.keras.models.load_model("512generatormodel.h5")
for word, definition, _ in data:
    #process and encode vector here
    if definition:
        encoded_vec = definition
        word_net.insert(structs.Node(structs.Word(word.lower(), definition, encoded_vec)))
        
word_net.run()