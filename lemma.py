# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:49:37 2020

@author: anant
"""


import nltk
import os
import re
import math
import operator
from banglakit import lemmatizer as lem
from banglakit.lemmatizer import BengaliLemmatizer
from banglakit.lemmatizer.consts import *
import io
from PyRouge.pyrouge import Rouge
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np

######### data


stop_words = set(stopwords.words("stopwords-bn"))
lemmatizer = BengaliLemmatizer()

def sentence_tokenize(text):
    sentences = re.compile('[।!?] ').split(text)
    sentences
    return sentences

def build_vocabulary(sentences):
    word_to_ix = {}
    ix_to_word = {}

    for sent in sentences:
        for word in sent:
            if word not in word_to_ix:                      #word_to_ix is index
                word_to_ix[word] = len(word_to_ix)
                ix_to_word[len(ix_to_word)] = word
    return word_to_ix, ix_to_word

def filter_sentences(sentences):
    norm_sents = [normalize_sentence(s) for s in sentences]
    filtered_sents = [filter_words(sent) for sent in norm_sents]
    fs = [lemmatize_words(sent) for sent in filtered_sents]       ####
    print(norm_sents)
    return fs

def filter_words(sentence):
    filtered_sentence = []
    for word in word_tokenize(sentence):
        if word in stop_words:
            continue

        filtered_sentence.append(word)
    
    return filtered_sentence


def load_data(file_name):
    file = open(file_name , 'r', encoding="utf8")
    text = file.read()
    tokenized_sentence = sentence_tokenize(text)
    sentences = tokenized_sentence
    return sentences

def normalize_sentence(sentence):   
    
    return sentence.replace(u"\u2013", u"-").replace(
        u"\u2019", u"'").replace(u"\u201c", u"\"").replace(
        u"\u201d", u"\"")

def lemmatize_words(words):             
    lemmatized_words = []
    for word in words:
          p = re.sub(r'\d+', '', lemmatizer.lemmatize(word, pos='verb'))
          p = lemmatizer.lemmatize(p, pos='noun')
          lemmatized_words.append(p)
    return lemmatized_words

#////////////////////////////////////////////////////// model

def build_coo_matrix(sentences, word_to_ix):                #### co occurance matrix
    S = np.zeros((len(word_to_ix), len(word_to_ix)))

    for sent in sentences:
        for src, target in zip(sent[:-1], sent[1:]):
            if src == target:
                continue
            
            S[word_to_ix[src]][word_to_ix[target]] = 1
            S[word_to_ix[target]][word_to_ix[src]] = 1
    
    return normalize_matrix(S)


def build_similarity_matrix(sentences):                     #### calculate how simillar
    S = np.zeros((len(sentences), len(sentences)))
    #print(S) 

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            
            S[i][j] = sentence_similarity(sentences[i], sentences[j])
    #print(normalize_matrix(S))
    return normalize_matrix(S)

def get_topk_keywords(keyword_ranks, ix_to_word, k=5):
    indexes = list(keyword_ranks.argsort())[-k:]
    return [ix_to_word[ix] for ix in indexes]

def get_topk_sentences(sentence_ranks, sentences, k=3):
    indexes = list(reversed(sentence_ranks.argsort()))[:k]
    return [sentences[i] for i in indexes]

def normalize_matrix(S):
    for i in range(len(S)):
        if S[i].sum() == 0:
            S[i] = np.ones(len(S))
        print(S)
        S[i] /= S[i].sum()
        
    print("End:",S)

    return S

def sentence_similarity(sent1, sent2):                      ##### similarity matrix
    overlap = len(set(sent1).intersection(set(sent2)))
    #print("word_to_ix")

    if overlap == 0:
        return 0
    
    return overlap / (np.log10(len(sent1)) + np.log10(len(sent2)))


def extract_keywords(sentences, k=5):                       #### keyword extract
    filtered_sentences = filter_sentences(sentences)

    word_to_ix, ix_to_word = build_vocabulary(filtered_sentences)
    S = build_coo_matrix(filtered_sentences, word_to_ix)    #### build_vocabulary -> build_coo_matrix

    ranks = pagerank(S)                                     #### S er er poriborte entropy er weighted value dite hobe

    return get_topk_keywords(ranks, ix_to_word, k)


def pagerank(A, eps=0.0001, d=0.85):
    R = np.ones(len(A))
   # print(A)
    while True:
        r = np.ones(len(A)) * (1 - d) + d * A.T.dot(R)
        if abs(r - R).sum() <= eps:
            return r
        R = r
        
#/////////////////////////////////////////////////////////// main      

          ## ikhane word er kaj

def summarize(sentences, k=5):
    filtered_sentences = filter_sentences(sentences)
    #print("hello") ok

    S = build_similarity_matrix(filtered_sentences)
    #print(S) 

    ranks = pagerank(S)

    return get_topk_sentences(ranks, sentences, k)


    
def main():
    
    input_user = int(input('Number of lines to retain:'))
    k = input_user
    
    sentences = load_data("G:/EWU/_Current_course/thesis/codes/input.txt")

    summary = summarize(sentences, k)
   # print(" ".join(summary))
    print("; ".join(extract_keywords(sentences, k)))
    
    r = Rouge()
    
    system_generated_summary = " ".join(summary)
    manual_summmary = "এভারকেয়ার হাসপাতাল  ঢাকা সম্প্রতি কোভিড -১৯ রোগীদের সুস্থতার জন্য সহায়তার জন্য একটি পুনরুদ্ধারের পোস্ট ক্লিনিক শুরু করেছে। তাদের পরিষেবাগুলিতে পালমোনারি কেয়ার এবং পুনর্বাসন, পালমোনারি শারীরবৃত্তীয় পরীক্ষা, ইমেজিং, মনোরোগ ও সামাজিক পরিষেবাগুলিও গবেষণা অধ্যয়ন COVID-19 এর সাথে সম্পর্কিত। ক্লিনিকটি অ্যারিথমিয়া এবং মায়োকার্ডিয়াল ডিসফংশন এর মতো হার্টের সমস্যাগুলির জন্য হোম পুনরুদ্ধারকে সমর্থন করে।"

    [precision, recall, f_score] = r.rouge_l([system_generated_summary], [manual_summmary])

    print("\nPrecision is :"+str(precision)+"\nRecall is :"+str(recall)+"\nF Score is :"+str(f_score))

           
if __name__ == '__main__':
    main()