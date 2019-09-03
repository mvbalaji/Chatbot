# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:05:59 2019

@author: gssbvenk
"""
from docx import Document 
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import nltk.data
import nltk
import random
from gensim.matutils import softcossim 

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f=open(r'C:\Users\gssbvenk\Desktop\Balaji\08. Work\Chatbot\SOP.txt','r',errors = 'ignore')

raw=f.read()
raw=raw.lower()# converts to lowercase
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
lemmer = nltk.stem.WordNetLemmatizer()

#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
def response(user_response):
    user_response=user_response.lower()
    robo_response=''
    sent_tokens.append(user_response)
     
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
     
    vals = cosine_similarity(tfidf[-1], tfidf)
    ##vals=softcossim(tfidf[-1], tfidf, tfidf)
    print(vals)
    idx=vals.argsort()[0][-2]

    flat = vals.flatten()
    flat.sort()
    
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response=robo_response+sent_tokens[idx]
        robo_response=robo_response.capitalize()
        robo_response=robo_response.lstrip('0123456789.')
        robo_response=robo_response.replace("  ", "")
        return robo_response.capitalize()
  
def score(query_string):
    return(random.randint(1,100)/100)
    
from flask import Flask
app = Flask(__name__)
from flask import Flask

from flask import request
app = Flask(__name__)
@app.route('/postjson', methods=['POST'])
def post():
    print(request.is_json)
    content = request.get_json()

    user_id=content['user_id']
    query_string=content['query_string']
    department=content['department']
    print(user_id)
    print(query_string)
    print(department)
    
    bot_response=response(query_string)
    print(bot_response)
    score=0.65 #random.randint(1,100)/100
    
    json_response="{'user_id': '"+user_id+"', "+'\n'+"'response': '"
    json_response=json_response+bot_response+"'\n"
    json_response=json_response+"'score': '"+str(score)+"'}"
    print(json_response)
    
    sent_tokens.remove(query_string)    
    return (json_response)

app.run(port=5000)