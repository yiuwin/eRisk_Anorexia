# General
from __future__ import print_function
import os
from xml.etree import ElementTree
import numpy as np
import pandas as pd
from sklearn import svm

# Classifiers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier


# Files and directories
from os import listdir
from os.path import isfile, join

# Preprocessing
from nltk.stem import PorterStemmer
from textblob import TextBlob, Word
import string, re
from nltk.corpus import stopwords
import nltk, ssl

# Pipelining
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin


'''
NOTE:
The program currently processes all chunks from the training data, combines them in a pandas dataframe,
fits them using logistic regression/support vector machine/adaboost, and then emits decisions for
each chunk in the testing data.


HOW TO RUN:
From main directory (task 2 - anorexia), run:
$ python erisk_task2.py
$ python test/aggregate_results.py -path decisions/ -wsource test/writings-per-subject-all-test.txt
$ python test/erisk_eval.py -gpath test/risk-golden-truth-test.txt -ppath decisions/usc_global.txt -o 50


QUESTIONS: 
- Should we build the model based on multiple features (as opposed to just the title/text)? i.e. also
include features like "number of words / submission"?
- How do we build the model based on multiple features?
- How to deal with class data imbalance?


CHANGELOG:
- added custom transformer classes
- added lemmatization (removed porter stemming) - improved from (f=0.58; p=0.71; r=0.49) to (f=0.65; p=0.77; r=0.56)
- added avg_tweet_length column to dataframe

TODO:
- pipeline text and numerical features
- combine pipelines

- try to uncomment "try" in preprocess() to see if CERTIFICATE_VERIFY_FAILED warning disappears

- random forest classifier
- knn

'''


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]
    


def readfile (file): # processes the XML and returns a single string of all titles + texts for this particular subject
    tweet_length = 0
    num_tweets = 0

    result = ''

    dom = ElementTree.parse(file) 
    tmp_subj_id = dom.find('ID').text
    writing = dom.findall('WRITING')

    for w in writing:
        title = w.find('TITLE').text
        text = w.find('TEXT').text
        result = result + title + text
        num_tweets = num_tweets + 1 # increment number of tweets counter
        tweet_length = tweet_length + len(text.split()) # increment number of words in tweet
        
    result = re.sub(r'[^\w\s]','',result) # remove punctuation
    result = result.lower() # make lowercase
    #result = TextBlob(result).correct()

    if (num_tweets==0):
        avg_tweet_length = 0
    else:
        avg_tweet_length = tweet_length/num_tweets

    return tmp_subj_id, avg_tweet_length, result


def build_df():

    blob1 = TextBlob("I hate Mondayy")
    print(format(blob1.sentiment))
    print(blob1.sentiment.polarity)
    print(blob1.sentiment.subjectivity)


    print(TextBlob("lol i know right i kno idk").correct())


    # ----------------------------------------------------------------------------------------
    # ----------------------------------- NEGATIVE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------

    chunks = [f for f in listdir('train/negative_examples/')] # negative - 132
    frames_neg = []

    for c in chunks:
        path = 'train/negative_examples/' + c
        negative = [f for f in listdir(path) if isfile(join(path, f))]

        for n in negative:
            (id, avglen, text) = readfile('train/negative_examples/'+c+'/'+n)

            # -- create NEGATIVE dataframe
            df = {'text' : text, 'avg_tweet_length' : avglen}
            df = pd.DataFrame(data=[df])

            # --
            frames_neg.append(df)


    result_neg = pd.concat(frames_neg)

    # - append target feature
    target = [0] * result_neg.shape[0]
    result_neg['target'] = target

    #result_neg = preprocess(result_neg) # ADDED



    # ----------------------------------------------------------------------------------------
    # ----------------------------------- POSITIVE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------
    chunks = [f for f in listdir('train/positive_examples/')] # negative - 132
    frames_pos = []

    for c in chunks:
        path = 'train/positive_examples/' + c
        positive = [f for f in listdir(path) if isfile(join(path, f))]

        for p in positive:
            (id, avglen, text) = readfile('train/positive_examples/'+c+'/'+p)

            # -- create NEGATIVE dataframe
            df = {'text' : text, 'avg_tweet_length' : avglen}
            df = pd.DataFrame(data=[df])

            # --
            frames_pos.append(df)


    result_pos = pd.concat(frames_pos)

    # - append target feature
    target = [1] * result_pos.shape[0]
    result_pos['target'] = target

    #result_pos = preprocess(result_pos) # ADDED

  
    # ----------------------------------------------------------------------------------------
    # ------------------------------------ COMBINE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------
    result = pd.concat([result_neg, result_pos])
    print(result)
    #print(type(result.iloc[1]), (result.iloc[1]).shape)

    result = preprocess(result)


    return result

def train_adaboost(df):
        # ---------------------------------------- build model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.target, test_size=0.01) # for personal testing (will not be used in final program)

    
    X_train = df.loc[:, 'text']
    y_train = df.loc[:, 'target']

    # -
    cvec = CountVectorizer(stop_words='english').fit(X_train) # use tf-idf instead?
    df_train = pd.DataFrame(cvec.transform(X_train).todense(),columns=cvec.get_feature_names())
    df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())

    model = AdaBoostClassifier(n_estimators=100)
    model.fit(df_train, y_train)

    
    prediction = model.predict_proba(df_test)
    print(prediction) # what the machine predicts is the possibility of class 0 vs. class 1, respectively
    #print( type(model.predict_proba(df_test)) )


    print(model.score(df_test, y_test)) # accuracy of the prediction for personal testing (will not be used in final program)
    return model, cvec

def train_svm(df):

    # ---------------------------------------- build model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.target, test_size=0.01)  # for personal testing (will not be used in final program)

    
    X_train = df.loc[:, 'text']
    y_train = df.loc[:, 'target']

    # -
    cvec = CountVectorizer(stop_words='english').fit(X_train)
    df_train = pd.DataFrame(cvec.transform(X_train).todense(),columns=cvec.get_feature_names())
    df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())

    model = svm.SVC(probability=True)
    model.fit(df_train, y_train)

    
    prediction = model.predict_proba(df_test)
    print(prediction) # what the machine predicts is the possibility of class 0 vs. class 1, respectively


    '''
    # ----------- TESTING AND EVALUATION STAGE
    print(prediction[0][0], prediction[0][1])
    print(model.predict(df_test)) # what the machine predicts are the test values
    '''

    print(model.score(df_test, y_test)) # accuracy of the prediction for personal testing (will not be used in final program)
    return model, cvec

def train_lr(df):
    print(df)

    # ---------------------------------------- build model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.target, test_size=0.33)  # for personal testing (will not be used in final program)

    X_train = df.loc[:, 'text']
    y_train = df.loc[:, 'target']

    ''' full pipeline using homogeneous text transformer '''
    pipeline = Pipeline([
        ('vect',CountVectorizer()),
        #('tfidf',TfidfTransformer()), # ERDE(o=50)=11.55%; f1=0.59; p=1.00; r=0.41
        ('clf', LogisticRegression())
    ])

    '''
    text = Pipeline([
                ('selector', TextSelector(key='text')),
                ('vect', CountVectorizer())
            ])
    text.fit_transform(X_train)

    avglen = Pipeline([
                ('selector', NumberSelector(key='avg_tweet_length')),
                ('standard', StandardScaler())
            ])
    avglen.fit_transform(X_train)


    # combine feature pipelines
    feats = FeatureUnion([
        ('text', text), 
        ('avglen', avglen)
    ])

    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X_train)


    # combine complete pipeline with classifier
    pipeline = Pipeline([
        ('features', feats),
        ('clf', LogisticRegression())
    ])

    prediction = pipeline.predict(X_test)
    np.mean(prediction == y_test)


    '''



    model = pipeline.fit(X_train, y_train)

    print(pipeline.score(X_test, y_test))
    print(np.mean(pipeline.predict(X_test) == y_test))
    
    return model


def preprocess(df):

    '''
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    '''

    # --------------------- stop word removal
    nltk.download("stopwords")
    stop = stopwords.words('english')
    #print(stop)
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    

    # --------------------- common word removal
    common = list((pd.Series(' '.join(df['text']).split()).value_counts()[:10]).index)
    print(common)
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in common))

    # --------------------- rare word removal
    rare = list((pd.Series(' '.join(df['text']).split()).value_counts()[-10:]).index)
    print(rare)
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in rare))
    #print((pd.Series(' '.join(df['text']).split()).value_counts()[:10]).index) # Index([u'it', u'the', u'i', u'you', u'a', u'and', u'of', u'to', u'im',u'that'],dtype='object')


    # --------------------- lemmatization
    df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


    '''
    # --------------------- stemming
    st = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()])) #result['text'][:5] is a <class 'pandas.core.series.Series'>

    '''
    return df

def test(model):
    
    text_list = []
    id_list = []

    for c in range(10): # for each chunk c, c

        path = 'test/chunk' + str(c+1)
        test_file = [f for f in listdir(path) if isfile(join(path, f))] # test_file: list of file names in the directory
        
        # for each subject file in chunk c
        for index,t in enumerate(sorted(test_file)):
            (id, avglen, text) = readfile(path + '/' + t)

            if (len(text_list)<=index):
                id_list.append(id)
                text_list.append(text)
            else:
                id_list[index] = id
                text_list[index] = text_list[index] + ' ' + text

        #print('[' + id_list[0] + ', ' + id_list[1] + ']')
        #print('[' + text_list[0] + ', ' + text_list[1] + ']')

        
        dict = {'id' : id_list, 'text' : text_list}
        result = pd.DataFrame(data=dict)
        result = preprocess(result)
        #print(result)

        X_test = result.loc[:, 'text']

        prediction_proba = model.predict_proba(X_test)
        prediction = model.predict(X_test)

        '''
        df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())
        prediction_proba = model.predict_proba(df_test)
        prediction = model.predict(df_test)
        #print(prediction)
        '''

        # --- open file
        f = open("decisions/usc_" + str(c+1) + ".txt", "a")

        i = 0
        for id in id_list:
            #print(prediction_proba[i][0], prediction_proba[i][1])

            #if (c!=9): # emit a decision regardless of prediction score
            if (c<3 and prediction_proba[i][0] < 0.99 and prediction_proba[i][1] < 0.99): # 0
                f.write(id + '\t\t0\n')
            elif (c<4 and prediction_proba[i][0] < 0.95 and prediction_proba[i][1] < 0.95): # 3
                f.write(id + '\t\t0\n')
            elif (c<7 and prediction_proba[i][0] < 0.85 and prediction_proba[i][1] < 0.85): # 4,5,6
                f.write(id + '\t\t0\n')
            elif (c<9 and prediction_proba[i][0] < 0.80 and prediction_proba[i][1] < 0.80): # 7,8 - if less than 97% sure, do not decide
                f.write(id + '\t\t0\n')
            else: # emit a decision
                if (prediction[i] == 0): # if the machine predicted 0 then write 2 to the file
                    f.write(id + '\t\t2\n')
                else: # write 1 because machine predicted 1
                    f.write(id + '\t\t1\n')

            i=i+1


        


# ----------------------------------------------------------------------------------------
# ------------------------------------- MAIN PROGRAM -------------------------------------
# ----------------------------------------------------------------------------------------

np.set_printoptions(threshold='nan') # --- display whole numpy array
pd.set_option('display.expand_frame_repr', False) # --- display whole dataframe


# ------ Build dataframe using training data
df = build_df()


# ------ Training stage - choose one of the following classification algorithms and comment out the others
model = train_lr(df) # --- logistic regression
#model, cvec = train_svm(df) # --- support vector machine
#model, cvec = train_adaboost(df) # --- ada boost


# ------ Testing stage
test(model)
#test()