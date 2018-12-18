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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


# Files and directories
from os import listdir
from os.path import isfile, join

# Preprocessing
from nltk.stem import PorterStemmer
from textblob import TextBlob, Word
import string, re
from nltk.corpus import stopwords, wordnet
import nltk, ssl
from googletrans import Translator
from nltk.stem import WordNetLemmatizer 

# Pipelining
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin


#Cross-validation and tuning
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_20newsgroups

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
- How to deal with class data imbalance?


CHANGELOG:
- added custom transformer classes
- added lemmatization (removed porter stemming) - improved from (f=0.58; p=0.71; r=0.49) to (f=0.65; p=0.77; r=0.56)
- added avg_tweet_length column to dataframe
- fixed average length column in test() 
- combined pipelines - (f=0.55; p=0.68; r=0.46)
- added numtweets feature column - (f=0.56; f=0.70; r=0.46)
- average out "avglen" after each chunk in test() - (f=0.54; p=0.63; r=0.46)
- separated accumulated numtweets (used for calculating average) and numtweets/persubj/perchunk ('numtweetsfeat_list') - (f=0.54; p=0.66; r=0.46)
- commented out "nltk download stopwords"

- try: remove numerical transformation

TODO:
- try to uncomment "try" in preprocess() to see if CERTIFICATE_VERIFY_FAILED warning disappears
- is lemmatization working?
- removing long words i.e. "httpswwwdropboxcoms5exolh7jd8rstizinstall_" probably doesn't mean anything
- change "tweets" to "posts"
- clitics - "they're" to "they are"

- random forest classifier
- knn


ERRORS ENCOUNTERED:
- Error from evaluation: "Some file or directory doesn't exist or an error has occurred" after implementing avglen
    * pos_hits and pos_decisions are both 0 results in division by 0 error
    * machine classifies all 2s (all negative)

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

    '''
    if(tmp_subj_id == 'subject1137_10.xml'):
        print('*****************************************************')
        print('******'+ result + '************')
        print('*****************************************************')


    if (num_tweets==0):
        avg_tweet_length = 0
    else:
        avg_tweet_length = tweet_length/num_tweets
    '''



    return tmp_subj_id, tweet_length, num_tweets, result


def build_df():

    # ----------------------------------------------------------------------------------------
    # ----------------------------------- NEGATIVE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------

    chunks = [f for f in listdir('train/negative_examples/')] # negative - 132
    frames_neg = []


    numposts=0
    for c in chunks:
        path = 'train/negative_examples/' + c
        negative = [f for f in listdir(path) if isfile(join(path, f))]

        for n in negative:
            (id, tweetlen, numtweets, text) = readfile('train/negative_examples/'+c+'/'+n)

            if (numtweets==0):
                avglen = 0
            else:
                avglen = tweetlen/numtweets

            # -- create NEGATIVE dataframe
            df = {'text' : text, 'avg_tweet_length' : avglen, 'num_tweets' : numtweets}
            df = pd.DataFrame(data=[df])

            # --
            frames_neg.append(df)
            numposts=numposts+numtweets

    print("total number negative posts: " + str(numposts))
    result_neg = pd.concat(frames_neg)

    # - append target feature
    target = [0] * result_neg.shape[0]
    result_neg['target'] = target

    #print('mean neg: ' + str(result_neg['num_tweets'].mean()))
    #print('mean avg_post_len neg: ' + str(result_neg['avg_tweet_length'].mean()))
    #result_neg = preprocess(result_neg) # ADDED

    result_neg['text'].str.strip()
    #print((result_neg['text'].values == '').sum())
    #print(result_neg.shape)


    totcount = 0
    for index, row in result_neg.iterrows():
        count = row['text'].count('calories')
        totcount = totcount + count

    print("total occurences of 'calories' in negative samples: " + str(totcount))


    # ----------------------------------------------------------------------------------------
    # ----------------------------------- POSITIVE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------
    chunks = [f for f in listdir('train/positive_examples/')] # negative - 132
    frames_pos = []

    numposts=0
    for c in chunks:
        path = 'train/positive_examples/' + c
        positive = [f for f in listdir(path) if isfile(join(path, f))]

        for p in positive:
            (id, tweetlen, numtweets, text) = readfile('train/positive_examples/'+c+'/'+p)

            if (numtweets==0):
                avglen = 0
            else:
                avglen = tweetlen/numtweets

            # -- create NEGATIVE dataframe
            df = {'text' : text, 'avg_tweet_length' : avglen, 'num_tweets' : numtweets}
            df = pd.DataFrame(data=[df])

            # --
            frames_pos.append(df)
            numposts=numposts+numtweets

    print("total number positive posts: " + str(numposts))


    result_pos = pd.concat(frames_pos)

    # - append target feature
    target = [1] * result_pos.shape[0]
    result_pos['target'] = target

    #print('mean pos: ' + str(result_pos['num_tweets'].mean()))
    #print('mean avg_post_len pos: ' + str(result_pos['avg_tweet_length'].mean()))
    #result_pos = preprocess(result_pos) # ADDED

    result_pos['text'].str.strip()
    #print((result_pos['text'].values == '').sum())
    #print(result_pos.shape)


    totcount = 0
    for index, row in result_pos.iterrows():
        count = row['text'].count('calories')
        totcount = totcount + count

    print("total occurences of 'calories' in positive samples: " + str(totcount))

  
    # ----------------------------------------------------------------------------------------
    # ------------------------------------ COMBINE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------
    result = pd.concat([result_neg, result_pos])
    #print(type(result.iloc[1]), (result.iloc[1]).shape)

    result = preprocess(result)
    

    result['text'].str.strip()
    print((result['text'].values == '').sum())
    

    return result


def tune(df):

    features = ['avg_tweet_length', 'num_tweets', 'text']
    target = 'target'
    
    X = df[features]
    y = df[target]

    text = Pipeline([
                ('selector', TextSelector(key='text')),
                #('vect', TfidfVectorizer()),
                ('count', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])
    text.fit_transform(X)

    avglen = Pipeline([
                ('selector', NumberSelector(key='avg_tweet_length')),
                ('standard', StandardScaler())
            ])
    avglen.fit_transform(X)

    numtweets = Pipeline([
                ('selector', NumberSelector(key='num_tweets')),
                ('standard', StandardScaler())
            ])
    numtweets.fit_transform(X)


    # combine feature pipelines
    feats = FeatureUnion([
        ('text', text),
        ('numtweets', numtweets), 
        ('avglen', avglen)
    ])

    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X)

    pipeline = Pipeline([
        ('features', feats),
        ('clf', RandomForestClassifier())
    ])


    #rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

    #print(X)
    #print(y)
    
    param_grid = { 
        'clf__n_estimators': [5, 10, 20, 50],
        'clf__criterion': ['gini', 'entropy' ],
        'clf__class_weight': ['balanced', 'balanced_subsample', None]
    }

    CV_rfc = GridSearchCV(scoring='f1', estimator=pipeline, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X, y)
    print (CV_rfc.best_params_)
    print(CV_rfc.best_score_)

    '''
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33)
    # knn tuning
    #k_range = np.linspace(0.1,4,40)
    k_range = range(60, 150, 5)
    k_scores = []
    for k in k_range:
        print(k)
        pipeline = Pipeline([
            ('features', feats),
            ('clf', LogisticRegression(C=k, class_weight='balanced'))
        ])
        #knn = KNeighborsClassifier(n_neighbors=k)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        k_scores.append(recall_score(y_test, y_pred, average='macro'))
        #scores = cross_val_score(pipeline, X, y, cv=10, scoring='recall')
        #k_scores.append(scores.mean())
    
    print (k_scores)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of c for LR')
    plt.ylabel('Cross-Validated score')
    plt.show()
    '''


def train(df, classifier):

    features = ['avg_tweet_length', 'num_tweets', 'text']
    target = 'target'

    # ---------------------------------------- build model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33)  # for personal testing (will not be used in final program)

    X_train = df[features]
    y_train = df[target]

    text = Pipeline([
                ('selector', TextSelector(key='text')),
                #('vect', TfidfVectorizer()),
                ('count', CountVectorizer()), # ngram_range=(1,2)
                ('tfidf', TfidfTransformer())
            ])
    text.fit_transform(X_train)

    avglen = Pipeline([
                ('selector', NumberSelector(key='avg_tweet_length')),
                ('standard', StandardScaler())
            ])
    avglen.fit_transform(X_train)

    numtweets = Pipeline([
                ('selector', NumberSelector(key='num_tweets')),
                ('standard', StandardScaler())
            ])
    numtweets.fit_transform(X_train)


    # combine feature pipelines
    feats = FeatureUnion([
        ('text', text),
        ('numtweets', numtweets), 
        ('avglen', avglen)
    ])

    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X_train)


    # combine complete pipeline with classifier
    if (classifier == 'lr'):
        pipeline = Pipeline([
            ('features', feats),
            ('clf', LogisticRegression(C=70, class_weight='balanced'))
        ])
    elif (classifier == 'svm'):
        pipeline = Pipeline([
            ('features', feats),
            ('clf', svm.SVC(C=1.5, kernel='linear', class_weight='balanced', probability=True))
        ])
    elif (classifier == 'adaboost'):
        pipeline = Pipeline([
            ('features', feats),
            ('clf', AdaBoostClassifier(n_estimators=2500, learning_rate=0.05)) # <--
        ])
    elif (classifier == 'randomforest'):
        pipeline = Pipeline([
            ('features', feats),
            ('clf', RandomForestClassifier(criterion='gini', n_estimators=5)) # 5000 to 1000
        ])
    elif (classifier == 'dt'):
        pipeline = Pipeline([
            ('features', feats),
            ('clf', DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=2)) # entropy: (f1=0.46; p=0.34; 0.71)
        ])
    elif (classifier == 'knn'):
        pipeline = Pipeline([
            ('features', feats),
            ('clf', KNeighborsClassifier(n_neighbors=3, n_jobs=2))
        ])
    else:
        pipeline = Pipeline([
            ('features', feats),
            ('clf', DummyClassifier(strategy='stratified'))
        ])


    pipeline.fit(X_train, y_train)

    model = pipeline.fit(X_train, y_train)


    prediction = model.predict(X_test)

    #print(prediction)
    #print('mean: ' + str(df['avg_tweet_length'].mean()))

    print('pipeline.score: ' + str(pipeline.score(X_test, y_test)))
    #print(np.mean(pipeline.predict(X_test) == y_test))

    # try print recall
    print('recall_score: ' + str(recall_score(y_test, prediction)))
    print('accuracy_score: ' + str(accuracy_score(y_test, prediction)))
    print('f1_score: ' + str(f1_score(y_test, prediction)))
    print('roc_auc: ' + str(roc_auc_score(y_test, prediction)))

    '''
    fpr, tpr, thres = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label='Test')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    '''
    
    '''
    # create a mesh to plot in
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()
    '''

    return model


def preprocess(df):
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    

    # --------------------- stop word removal
    #nltk.download("stopwords")
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


    '''
    # --------------------- improved lemmatization with POS
    #df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    lemmatizer = WordNetLemmatizer()

    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(x)]])) #result['text'][:5] is a <class 'pandas.core.series.Series'>
 
    '''
    # --------------------- lemmatization
    df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    '''
    # --------------------- stemming   
    st = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()])) #result['text'][:5] is a <class 'pandas.core.series.Series'>
    '''

    return df

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def test(model):
    
    text_list = []
    avglen_list = [] # average lengths as of this chunk
    tweetlen_list = [] # total accumulative length of all tweets for this subject
    numtweets_list = [] # total accumulative number of tweets for this subject

    numtweetsfeat_list = []
    id_list = []

    numposts=0
    for c in range(10): # for each chunk c, c

        path = 'test/chunk' + str(c+1)
        test_file = [f for f in listdir(path) if isfile(join(path, f))] # test_file: list of file names in the directory
        
        # for each subject file in chunk c
        for index,t in enumerate(sorted(test_file)):
            (id, tweetlen, numtweets, text) = readfile(path + '/' + t)

            if (numtweets==0):
                avglen = 0
            else:
                avglen = tweetlen/numtweets

            #if (len(text_list)<=index): # first chunk, fill list
            if (c==0):
                id_list.append(id)
                tweetlen_list.append(tweetlen)
                numtweets_list.append(numtweets)
                avglen_list.append(avglen)
                numtweetsfeat_list.append(numtweets)
                text_list.append(text)
            elif (id_list[index] == id):
                #print('id_list['+str(index)+'] = ' + str(id_list[index]))
                #id_list[index] = id
                #print('tweetlen_list['+str(index)+'] = ' + str(tweetlen_list[index]))
                tweetlen_list[index] = tweetlen_list[index] + tweetlen
                #print('numtweets_list['+str(index)+'] = ' + str(numtweets_list[index]))
                numtweets_list[index] = numtweets_list[index] + numtweets
                #print('avglen_list['+str(index)+'] = ' + str(avglen_list[index]))
                avglen_list[index] = tweetlen_list[index]/numtweets_list[index]
                #print('numtweetsfeat_list['+str(index)+'] = ' + str(numtweetsfeat_list[index]))
                numtweetsfeat_list[index] = numtweets
                #print('numtweets = ' + str(numtweets))
                text_list[index] = text_list[index] + ' ' + text
            numposts=numposts+numtweets

        print("total number test posts: " + str(numposts))

        #print('[' + id_list[0] + ', ' + id_list[1] + ']')
        #print('[' + text_list[0] + ', ' + text_list[1] + ']')

        
        dict = {'id' : id_list, 'avg_tweet_length' : avglen_list, 'num_tweets' : numtweetsfeat_list, 'text' : text_list}
        result = pd.DataFrame(data=dict)
        result = preprocess(result)

        #features = ['text'] # (f=0.65; p=0.77; r=0.56)
        features = ['avg_tweet_length', 'num_tweets', 'text']
        X_test = result[features]


        prediction_proba = model.predict_proba(X_test)
        prediction = model.predict(X_test)
        #print(result)
        #print(prediction_proba)
        #print(prediction)
        print('mean: ' + str(result['avg_tweet_length'].mean()))
        #print(prediction_proba)

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
            if (c<2 and prediction_proba[i][0] < 0.98 and prediction_proba[i][1] < 0.98): # 0,1
                f.write(id + '\t\t0\n')
            elif (c<4 and prediction_proba[i][0] < 0.95 and prediction_proba[i][1] < 0.95): # 2,3
                f.write(id + '\t\t0\n')
            elif (c<7 and prediction_proba[i][0] < 0.80 and prediction_proba[i][1] < 0.80): # 4,5,6
                f.write(id + '\t\t0\n')
            elif (c<9 and prediction_proba[i][0] < 0.70 and prediction_proba[i][1] < 0.70): # 7,8 - if less than 97% sure, do not decide
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

print(np.linspace(0.1,4,40))
print(range(60, 150, 5))


'''
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)


print(X)
print(y)
print(type(X))
print(X.shape)
print(type(y))
print(y.shape)

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print (CV_rfc.best_params_)
'''


# ------ Build dataframe using training data
df = build_df()


# ------ Training stage - choose one of the following classification algorithms and comment out the others
model = train(df, 'svm')


# ------ Tuning stage
#tune(df)

# ------ Testing stage
test(model)