# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:09:05 2021

@author: peter
"""

from nltk import WordNetLemmatizer as WNL
from nltk import word_tokenize

class Lemmatizer():
    def __init__(self):
        self.wnl = WNL()
        
    def __call__(self, text):
        word_list = word_tokenize(text)
        tokens = [self.wnl.lemmatize(word) for word in word_list]
        
        return tokens
 