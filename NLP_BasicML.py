# imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from collections import Counter

# Import modules from natural language toolkit
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns
import matplotlib.pyplot as plt

def clean_the_phrase(phrase):
    # Function to convert a phrase to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    
    #
    # 1. Convert to lower case, 
    text = phrase.lower()
      
    #
    # 2. tokenize 
    words = word_tokenize(text)          
    # 

    #
    #  3. Stem the words 
    stemmer = SnowballStemmer(language='english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    # 4. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( stemmed_words ))   

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def print_performance(clf, X, y, vect: str, model_name: str):
    """
    Prints metrics for a classifier and plots heatmap of the classifications.
    Inputs:
    > clf: already trained classifier to do predictions from.
    > X: test data to predict with
    > y: test values to compare predictions with
    > vect: name of vectorizer
    > model_name: name of the model. 
        vect and model_name are used for plot labels
    returns:
    >prints accruacy, precision, recall and f1 scores. computes confusion matrix
    and plots heatmap off of it. 
    """
    y_pred = clf.predict(X)
    
    accuracy = accuracy_score(y_pred, y)
    print("Accuracy: ", accuracy)
    print("Precision:", precision_score(y_pred, y, average='weighted'))
    print("Recall:   ", recall_score(y_pred, y, average='weighted'))
    print("F1 Score: ", f1_score(y_pred, y, average='weighted'))
    
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(9,9)) 
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    if vect.lower() == 'tfidf':
        all_sample_title = 'Model: {1} \n Accuracy Score with TFIDF: {0}'.format(accuracy,model_name)
    else:
        all_sample_title = 'Model: {1} \n Accuracy Score with Count Vect.: {0}'.format(accuracy, model_name)
    plt.title(all_sample_title, size = 15);


def main (data, test, vect: str, model_name: str, phrase: str, toPredict: str):
    """
    Main function. Executes the main script.
    calls print_performance and clean_the_phrase functions above.
    inputs:
    > data: training dataframe already read in from user input
    > test: test dataframe already read in from user input
    > vect: a string indicating whether to use count vectorizer or tfidf vectorizer.
        These were the only two vectorizers studied in this analysis. 
    > model_name: Name of classifier to use. Right now only written for Logistic regression and         multinomial naieve bayes. These were the best performing two basic ML classifiers in the
        original kaggle contest work
    > phrase: column name in the dataframe corresponding to the phrases to classify.
    > toPredict: column name of the sentiments. 
    returns:
    > trains a model, returns different accuracy scores for said model as well as a 
    heatmap of the classification confusion matrix.
    """
    td_clean=data
    # Apply function to clean the data
    td_clean[toPredict] = td_clean[toPredict].apply(clean_the_phrase)    
    td_clean.head()
    print(td_clean.shape)
    # import vectorizers
    cv = CountVectorizer() # TODO expt with max_features
    tv = TfidfVectorizer(tokenizer =None,
                            stop_words=None,
                            ngram_range=(1,2))
    # Train/test split
    X, y = td_clean.drop(columns=[toPredict]), td_clean[toPredict]
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, stratify=y)
    #Vectorize input based on user indicated vectorizer
    if vect.lower() == 'count':
        features = cv.fit_transform(X_train[phrase])
        X_train = pd.DataFrame.sparse.from_spmatrix(features, columns=cv.get_feature_names_out())
    elif vect.lower() == 'tfidf':
        features = tv.fit_transform(X_train[phrase])
        X_train = pd.DataFrame.sparse.from_spmatrix(features, columns=tv.get_feature_names_out())
    else:
        print("This script does not handle the specified vectorizer.")
    
    # Fit user given model.
    if model_name.lower() == 'multinomial naive bayes':
        model = MultinomialNB()
        model.fit(features, y_train)
    if model_name.lower() == 'logistic regression':
        model = LogisticRegression()
        model.fit(features, y_train)
    # print model scores
    print('Model Score: ', modelNB.score(cv_features, y_train))
    print('-----------------------\n')
    print('vectorizer: {0}, model: {1}'.format(vect, model_name))
    scores = cross_val_score(model, features, y_train, scoring='accuracy', n_jobs=-1, cv=3)
    print(' Cross-validation mean accuracy  {0:.2f}%, std {1:.2f}. 
          \n'.format(np.mean(scores) * 100, np.std(scores) * 100))
    # Get validation features from validation data and plot heatmap
    if vect.lower()='count':   
        val_features = cv.transform(X_validate['Phrase'])
        print(val_features.shape)
    else:
        val_features = tv.transform(X_validate['Phrase'])
        print(val_features.shape)
    X_val = pd.DataFrame.sparse.from_spmatrix(val_features, columns=cv.get_feature_names_out())
    y_pred = model.predict(X_val)
    print_performance(model, X_val, y_validate,vect,model_name)

if __name__ =="__main__":
    #import
    data_name = input("Enter path to training dataset: ")
    test_name = input("Enter path to test data: ")
    try:
        data=pd.read_csv(data_name)
    except FileNotFoundError or pandas.errors.EmptyDataError:
        data_name = input("Path to data not found or not a csv file. Enter correct file path: ")
    try:
        test=pd.read_csv(test_name)
    except FileNotFoundError or pandas.errors.EmptyDataError:
        test_name = input("Path to data not found or not a csv file. Enter correct file path: ")
    print("Head of dataframe: \n")
    data.head()
    # Set working parameters
    toPredict = input("Enter index of sentiment: ")
    try:
        data[toPredict]
    except KeyError:
        toPredict = input("Not a part of dataset, enter a valid index: ")
    phrase = input("Enter index of the phrase to classify: ")
    try:
        data[phrase]
    except KeyError:
        phrase = input("Not a part of dataset, enter a valid index: ")
    model_name = input('Use multinomial naive bayes or logistic regression model? ')
    vect = input("Use 'count' or 'tfidf' vectorizer? ")
    main(data, test, vect, model_name, phrase, toPredict)
   
