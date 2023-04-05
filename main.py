from flask import Flask,request,jsonify
import joblib
import sklearn
# import streamlit as st
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import streamlit as st
import requests
import json

print('\n')
print("Server Start At Host: 192.168.1.42")
print("Server Start on Port: 5000")
print('\n')
print("Status: Online")

# Load the saved model
model = joblib.load('model2.pkl')
vectortext = joblib.load('vectorizer.pkl')


tokenizer = RegexpTokenizer(r'[A-Za-z]+')
def splitUrlIntoToken(url):
    return tokenizer.tokenize(url)
def removeTextFromList(input, rmTxt):
    input = [x for x in input if (x not in rmTxt)]
    return input

snowball = SnowballStemmer("english")

def textNormByStemType(inputTxt, stemtype):
    inputTxt =[stemtype.stem(word) for word in inputTxt]
    return inputTxt

tfid = vectortext 
# tfid = TfidfVectorizer()


def convertTextToVectorByType(data, vectorType):
    # Get the feature names from the vectorizer
    feature_names = vectorType.get_feature_names_out()


    # Filter out the features that are not present in the training data
    data['split_url_text'] = data['split_url_text'].apply(lambda x: ' '.join([w for w in x.split() if w in feature_names]))
    return vectorType.transform(data['split_url_text'])

def extract_feature(data, vectorType):
    
    # Split url function
    data['split_url_list'] = data['URL'].map(lambda t: splitUrlIntoToken(t))
    # Remove text function
    removeTextList = ['www','http','https']
    data['split_url_list'] = data['split_url_list'].map(lambda t: removeTextFromList(t, removeTextList))
    # Stemming function
    snowball = SnowballStemmer("english")
    data['split_url_list'] = data['split_url_list'].map(lambda t: textNormByStemType(t, snowball))
    # join text
    data['split_url_text'] = data['split_url_list'].map(lambda t: ' '.join(t))


    # Convert to vector function
    vectorText = convertTextToVectorByType(data, vectorType)

    return vectorText, data['split_url_text']



app = Flask(__name__)

@app.route('/')
def my_api():
    # Return the result as JSON
    return jsonify({
    'foo': 'bar',
    'baz': 'boz',
    'stuff': [
        'stuff 1',
        'stuff 2',
        'stuff 3',
        'stuff 5',
        'strufy1234'
    ],
})

@app.route('/project', methods=['POST'])
def project():
    targetURL = request.json['URL']
    

    target = pd.DataFrame({
        "URL":pd.Series([targetURL]),
        # "Label":''
    })


    # predict_data = convertTextToVectorByType(target,tfid)
    # predict_data = extract_feature(target, tfid)
    _,lable_list = extract_feature(target, tfid)
    predict_data =  tfid.transform(lable_list)

    # print(type(predict_data), '<============================================================')
    # print(predict_data.shape, '<============================================================')
    y_predict = model.predict(predict_data)[0]
    
    print(y_predict, '<======= ', targetURL)
    
    # return y_predict.tolist()
    return y_predict

def onClick (title):
    # title = st.text_input('URL','')

    url = title
    if url: 
        payload = json.dumps({
        "URL": title
        })
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", "http://192.168.1.84:5000/project", headers=headers, data=payload)
        st.write(response.text)
        print(response.text)
        

# app.run(debug = False, port = 5000,host = '192.168.1.84')
st.set_page_config(page_title='Web Phishing Detect')
st.write('## Phishing Detection')
title = st.text_input('URL','')
Checking = st.button('Check',on_click=onClick(title=title)) 


app.run(debug = False, port = 5000,host = '192.168.1.84')
# app.run(debug = False, port = 5000,host = 'localhost')
print('\n')