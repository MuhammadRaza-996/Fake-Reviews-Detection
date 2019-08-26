# -*- coding: utf-8 -*-
"""
@author: Raza
"""

import re
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import nltk
import numpy as np
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def reading_file():
   
    data = pd.read_csv("amazon_reviews.txt", delimiter = "\t") 
    data.loc[data["LABEL"] == "__label1__", "LABEL"] = '1'
    data.loc[data["LABEL"] == "__label2__", "LABEL"] = '0'

#    nai yakki

    data['TEXT_LENGTH'] = data['REVIEW_TEXT'].apply(len)
    data['NUM_SENTENCES'] = data['REVIEW_TEXT'].apply(lambda x: len(str(x).split('.')))
#    data['TEXT_LENGTH'] = data['TEXT_LENGTH'] / data['TEXT_LENGTH'] .max()
#    data['num_sentences'] =data['num_sentences']/data['num_sentences'].max()

#yaki khatam
    return data

# function for extracting coloms from data file 
# parameters should be coloms name that you want to extract
# also including nan values
    
def extracting_relevent_data(x,y,mydata):
    data = mydata[[x, y]]
    data = data.dropna()
    return data

#Making Sentence That will be used to clean the data
def cleanup(sentence):
    cleanup_re =re.compile('[^a-z]+')
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ',sentence).strip()
    return sentence

#Cleaning Data sert and removing Stopwords
def clean_data(col1,data):
    data[col1] = data[col1].apply(cleanup)
    stop_words = nltk.corpus.stopwords.words('english')
    data[col1] = data[col1].str.lower().str.split()  
    data[col1] = data[col1].apply(lambda x: [item for item in x if item not in stop_words])
    data[col1] = data[col1].apply(cleanup)
    return data

def tf_idf_processing1(data,y):
    #col,data = 'reviews.text',dataset
    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=5, max_df=0.9, ngram_range=(1,2)) 
    tfidf_vectorizer.fit(data)
    data = tfidf_vectorizer.transform(data)
    
#   for y

    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', ngram_range=(1,2)) 

    tfidf_vectorizer.fit(y)
    y = tfidf_vectorizer.transform(y)
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
    return data,y
def tfidf_features(X_train,X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
     
    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=5, max_df=0.9, ngram_range=(1,1)) ####### YOUR CODE HERE #######
    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    dense_train = X_train.todense()
    X_test = tfidf_vectorizer.transform(X_test)
    dense_test = X_test.todense()
    set_matrix = tfidf_vectorizer.vocabulary_
#    print(len(tfidf_vectorizer.vocabulary_))

    return X_train, X_test, set_matrix, dense_train, dense_test

def write_to_file(data):
    data.to_csv('f1.csv', sep=',')

def split_data(data):
    data = data.values
    np.random.shuffle(data)
    print('shuffle')
    data = pd.DataFrame(data)
    
    return data

def test_train_data(col,col1,data):
    print('in here')
    x_data = data[col]
    y_data = data[col1]
    x_test_values = x_data[14700:].values
    x_train_values = x_data[:14700].values
    y_test_values = y_data[14700:].values.astype('object')
    y_train_values = y_data[:14700].values.astype('object')
    return x_train_values, x_test_values, y_train_values, y_test_values

def trainingSVM(x_train,x_test,y_train,y_test):
    clf = SVC(gamma=1)
    clf.fit(x_train, y_train) 
    print('chal')
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)

    return clf.score(x_test, y_test, sample_weight=None)

def naieve_base(x_train,x_test,y_train,y_test):
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    return clf.score(x_test, y_test, sample_weight=None)


def random_forest_classifier(x_train,x_test,y_train,y_test):
    clf = RandomForestClassifier()
    clf.fit(x_train,y_train)
    return clf.score(x_test, y_test, sample_weight=None)

def log_reg(x_train,x_test,y_train,y_test):
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    return clf.score(x_test, y_test, sample_weight=None),clf



dataset = reading_file()
features, label = 'REVIEW_TEXT','LABEL'
extracted_data = extracting_relevent_data(features,label,dataset)

clean_data = clean_data(features,extracted_data)
shuffled_data = split_data(clean_data)
shuffled_data[features] = shuffled_data[0]
shuffled_data[label] = shuffled_data[1]
shuffled_data = shuffled_data.drop(columns=0)
shuffled_data = shuffled_data.drop(columns=1)
############################################################################################################

x_train,x_test,y_train,y_test = test_train_data(features,label,shuffled_data)

tf_x_train,tf_x_test,matrix,dense_train,dense_test = tfidf_features(x_train,x_test)
#print(dense_test)

new_length = dataset['TEXT_LENGTH']
new_length1 = dataset['TEXT_LENGTH']
sentence = dataset['NUM_SENTENCES']
sentence2 = dataset['NUM_SENTENCES']

sentence = sentence[:14700]
sentence2 = sentence2[14700:]

sentence = np.array(sentence)
sentence2 = np.array(sentence2)
sentence = sentence.reshape(14700,1)
sentence2 = sentence2.reshape(6300,1)



new_length = new_length[:14700]
new_length1 = new_length1[14700:]
#
new_length1 = np.array((new_length1))
new_length1 = new_length1.reshape(6300,1)
#
new_length = np.array((new_length))
new_length = new_length.reshape(14700,1)
new_set_train = np.append(dense_train,new_length,axis=1)
new_set_test = np.append(dense_test,new_length1,axis=1)
new_set_train = np.append(new_set_train,sentence,axis=1)
new_set_test = np.append(new_set_test,sentence2,axis=1)



y_train = y_train.reshape((y_train.shape[0], 1))
y_train = y_train.astype('int')
y_test = y_test.reshape((y_test.shape[0], 1))
y_test = y_test.astype('int')
x_train = x_train.reshape((x_train.shape[0], 1))

#
#print(tf_x_train.shape)
#print(type(tf_x_train))
#print(type(dense_train))
from scipy import sparse

sA = sparse.csr_matrix(new_set_train)
#print(type(sA))
#print(sA[:4])
sB = sparse.csr_matrix(new_set_test)

#trainingSVM(sA.toarray(),sB.toarray(),y_train,y_test)


###############-------------Giving Olny Length And Sentances-------#############
len_data = sA
len_data = len_data.todense()
len_data1 = len_data[:,-2]
len_data2 = len_data[:,-1]
new_length_data = np.append(len_data1,len_data2,axis=1)
train_length_data = new_length_data[:14700]

len_data_test = sB
len_data = len_data_test.todense()
len_data_1 = len_data[:,-2]
len_data_2 = len_data[:,-1]
new_length_data_2 = np.append(len_data_1,len_data_2,axis=1)

test_length_data = new_length_data_2

################################################################################
#print(len_data[:4])
#print(test_length_data.shape)
#
#print(test_length_data.shape)
#svm1 = trainingSVM(train_length_data,test_length_data,y_train,y_test)
#nb1 = naieve_base(train_length_data,test_length_data,y_train,y_test)
#rf1 = random_forest_classifier(train_length_data,test_length_data,y_train,y_test)
#
#lr1,trained_lr = log_reg(train_length_data,test_length_data,y_train,y_test)

#svm1 = trainingSVM(sA,sB,y_train,y_test)
#nb1 = naieve_base(sA.toarray(),sB.toarray(),y_train,y_test)

##svm2 = trainingSVM(tf_x_test,tf_x_train,y_test,y_train)
#nb1 = naieve_base(sA.toarray(),sB.toarray(),y_train,y_test)
#nb2 = naieve_base(sA.toarray(),sB.toarray(),y_train,y_test)
#rf1 = random_forest_classifier(sA,sB,y_train,y_test)
###rf2 = random_forest_classifier(tf_x_test,tf_x_train,y_test,y_train)
#lr1,trained_lr = log_reg(tf_x_train,tf_x_test,y_train,y_test)
###lr2,trained_lr1 = log_reg(tf_x_test,tf_x_train,y_test,y_train)
#
#
#
#print("Accuracy of SVM is \t",svm1*100," %")
#print("Accuracy of NB is \t",nb1*100," %")
#print("Accuracy of RF is \t",rf1*100," %")
#print("Accuracy of LR is \t",lr1*100," %")

#print("Accuracy of SVM on 2 fold is \t", ((svm1+svm2)/2)*100," %")
#print("Accuracy of NB on 2 fold is \t", (nb2*100," %"))
#print("Accuracy of RF on 2 fold is \t", ((rf1+rf2)/2)*100," %")
#print("Accuracy of LR on 2 fold is \t", ((lr1+lr2)/2)*100," %")





















