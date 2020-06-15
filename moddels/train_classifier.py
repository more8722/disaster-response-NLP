import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pickle
import time
start = time.process_time()

def load_data(database_filepath):
    '''
     Input:
           database_filepath path to SQL  database
     Output:
           X: message (feature)
           y: Categories (target)
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original','genre'])
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    '''
    Input:
          str tokenize and transform input text.
    Ouput:
          tokens= list with processed token
    '''      
    #remove special characters 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #change url to urlpalceholder(str)
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #tokenize    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    #lowercase and strip
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
      Return Grid Search model with pipeline and Classifier 
    '''
    
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    parameters = {'clf__estimator__n_estimators': [40,50],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    
    
    '''
    Evaluate model performance.
    Input:
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    y_pred = model.predict(X_test)

    # Calculate the accuracy for each of them.

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n",                         classification_report(Y_test.iloc[:,i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:,i].values, y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file
    Input:
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()