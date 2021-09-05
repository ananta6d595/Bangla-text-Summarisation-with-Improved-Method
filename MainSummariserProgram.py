# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 08:18:31 2020

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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyRouge.pyrouge import Rouge



Stopwords = set(stopwords.words("stopwords-bn"))
lemmatizer = BengaliLemmatizer()


def sentence_tokenize(text):
    sentences = re.compile('[।!?] ').split(text)
    sentences
    return sentences


def lemmatize_words(words):             
    lemmatized_words = []
    for word in words:
          p = re.sub(r'\d+', '', lemmatizer.lemmatize(word, pos='verb'))
          p = lemmatizer.lemmatize(p, pos='noun')
          lemmatized_words.append(p)
    return lemmatized_words


def freq(words):
    words = [word for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq

def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf

def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word not in Stopwords and len(word)>1]
        sentence = [word for word in sentence]
        sentence = [lemmatize_words(word) for word in sentence]
        
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf

def tf_idf_score(tf,idf):
    return tf*idf

def word_tfidf(dict_freq,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def sentence_importance(sentence,dict_freq,sentences):
     sentence_score = 0
     sentence = re.sub(r'\d+', '', sentence)
     no_of_sentences = len(sentences)
     for word in sentence:
          if word not in Stopwords and word not in Stopwords and len(word)>1: 
                word = lemmatize_words(word)
                sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)
     return sentence_score
############
'''pos tag er kaj lemmatization korar shomoy ek shathei kore felsi.
    ekta file e pos gula tag kora chilo. ekhon iteration chalanor shomoy je word ta
    mile geche shetar root word shei pos onushare kore dilei hobe.
    jemon kono ekta word er pos noun holo tokhon noun er root word ber korar niom/rule onushare 
    root word ber korbo. noun, verb, prottek er jonno alada rule er file ache.Banglakit e.
'''
############
file = 'G:/EWU/_Current_course/thesis/codes/input.txt'
file = open(file , 'r',encoding="utf8")
text = file.read()
tokenized_sentence = sentence_tokenize(text)
text = re.sub(r'\d+', '', text)
tokenized_words_with_stopwords = word_tokenize(text)
tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
tokenized_words = [word for word in tokenized_words if len(word) > 1]
tokenized_words = lemmatize_words(tokenized_words)
word_freq = freq(tokenized_words)

input_user = int(input('Number of lines to retain:'))
no_of_sentences = input_user
#print(no_of_sentences)

c = 1
sentence_with_importance = {}
for sent in tokenized_sentence:
    sentenceimp = sentence_importance(sent,word_freq,tokenized_sentence)
    sentence_with_importance[c] = sentenceimp
    c = c+1

sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)

cnt = 0
summary = []
sentence_no = []
for word_prob in sentence_with_importance:
    if cnt < no_of_sentences:
        sentence_no.append(word_prob[0])
        cnt = cnt+1
    else:
      break
  
sentence_no.sort()
cnt = 1
for sentence in tokenized_sentence:
    if cnt in sentence_no:
       summary.append(sentence)
    cnt = cnt+1
    
summary = " ".join(summary)
print("\n")
print("Summary:")
print(summary)
outF = open('summary.txt',"w",encoding="utf8")
outF.write(summary)


r = Rouge()
    
system_generated_summary = summary
manual_summmary = "নাসা আগামী এপ্রিলের মধ্যে চারটি প্রস্তাব থেকে দুটি গ্রহ বিজ্ঞান মিশনের অনুমোদনের কথা বিবেচনা করছে। এর মধ্যে একটি হ'ল শুক্রকে নির্ধারণ করার জন্য এটি জীবনকে লালন করে কিনা। সোমবার একটি আন্তর্জাতিক গবেষণা দল বর্ণনা করেছে যে ভেনাসিয়ার মেঘে থাকা সম্ভাব্য জীবাণুগুলির কোনও প্রমাণ রয়েছে। এছাড়াও ফসফাইন রয়েছে যেখানে অক্সিজেনমুক্ত পরিবেশে ব্যাকটিরিয়া থাকতে পারে। এগুলি দৃড় প্রমাণ দেয় যে এখনও পৃথিবী ছাড়িয়ে জীবন রয়েছে। মার্কিন যুক্তরাষ্ট্রের মহাকাশ সংস্থা ফেব্রুয়ারিতে চারটি মিশনকে শর্টলিস্ট করেছিল যা নাসা প্যানেল দ্বারা পর্যালোচনা করা হয় এবং এর মধ্যে দুটি ভেনাসে রোবোটিক প্রোব জড়িত। DAVINCI + নামের একটিকে ভেনুসিয়ান বায়ুমণ্ডলে প্রেরণ করা হবে।"

[precision, recall, f_score] = r.rouge_l([system_generated_summary], [manual_summmary])

print("\nPrecision is :"+str(precision)+"\nRecall is :"+str(recall)+"\nF Score is :"+str(f_score))
