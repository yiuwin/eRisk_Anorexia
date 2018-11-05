from __future__ import print_function
import os
from xml.etree import ElementTree
import numpy as np
import pandas as pd
from sklearn import svm


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer


from os import listdir
from os.path import isfile, join


'''
Note:

The program currently processes both negative and positive exampls from chunk 1 in the training data,
divides it randomly into its own sets of training data (80% for training, 20% for testing) and then
tests it on a logistic regression model (dividing the data up for training and testing is for my personal
testing purposes - not instructions from the project).

It currently achieves 92% accuracy but that is also because 90% of the data is negative (so guessing 
that everything is negative would yield 90% accuracy but does not tell me anything because I am more
interested in the positive classification)

- How to deal with this class data imbalance?
- Is it possible to 

- Should we build the model based on multiple features (as opposed to just the title/text)? i.e. also
include features like "number of words / submission"?
- How do we build the model based on multiple features?

'''




def process_chunk():

    # ------------ PROCESS CHUNK 1
    negative = [f for f in listdir('train/negative_examples/chunk1') if isfile(join('train/negative_examples/chunk1', f))] # negative - 132
    positive = [f for f in listdir('train/positive_examples/chunk1') if isfile(join('train/positive_examples/chunk1', f))] # positive - 20



    # ----------------------- create dataframe for NEGATIVE data
    frames_neg = []
    for n in negative:
        (tmp_subj_id, tmp_date, tmp_info, tmp_text) = readfile('train/negative_examples/chunk1/'+n)

        # -- create NEGATIVE dataframe
        df = {'date' : tmp_date, 'info' : tmp_info, 'text' : tmp_text}
        df = pd.DataFrame(data=df)

        # --
        frames_neg.append(df)

    result_neg = pd.concat(frames_neg)
    print(result_neg.shape)

    # - append target feature
    target = [0] * result_neg.shape[0]
    result_neg['target'] = target




    # ---------------------- create dataframe for POSITIVE data
    frames_pos = []
    for p in positive:
        #print ('train/negative_examples/chunk1/'+n)
        (tmp_subj_id, tmp_date, tmp_info, tmp_text) = readfile('train/positive_examples/chunk1/'+p)

        # --- create NEGATIVE dataframe
        df = {'date' : tmp_date, 'info' : tmp_info, 'text' : tmp_text}
        df = pd.DataFrame(data=df)

        # ---
        frames_pos.append(df)

    result_pos = pd.concat(frames_pos)

    # -- append target feature
    target = [1] * result_pos.shape[0]
    result_pos['target'] = target






    # ------------ COMBINE NEGATIVE AND POSITIVE dataframes
    result = pd.concat([result_neg, result_pos])
    print(result)

    

    # --- build model
    X_train, X_test, y_train, y_test = train_test_split(result.text, result.target, test_size=0.2)

    
    ''' test print the X_train, X_test, y_train, y_test values
    print('-------------------- X_train ------------------') # X_train and Y_train must have same columns in order to run logistic regression
    print(X_train) # shape is (19, 6)
    print('-------------------- Y_train ------------------')
    print(y_train) # shape is (19,)
    print('-------------------- X_test ------------------')
    print(X_test)
    print('-------------------- y_test ------------------')
    #print(y_test, end=' ') # actual test values
    for x in y_test:
        print (x,end='  ')
    #print((y_test).shape)
    '''

    # -
    cvec = CountVectorizer().fit(X_train)
    df_train = pd.DataFrame(cvec.transform(X_train).todense(),columns=cvec.get_feature_names())
    df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())


    # - pass data to classifier after using tfidftransformer
    lr = LogisticRegression()
    model = lr.fit(df_train, y_train)
    #print(model.predict_proba(df_test)) # what the machine predicts is the possibility of class 0 vs. class 1, respectively
    #print( type(model.predict_proba(df_test)) )

    #for x in np.nditer(model.predict_proba(df_test)):
    #    print (x)


    #print(model.predict(df_test)) # what the machine predicts are the test values
    #print((model.predict(df_test)).shape)

    '''
    confidence = model.decision_function(df_test)
    print(confidence)
    print(confidence.shape)
    '''


    
    print(lr.score(df_test, y_test)) # accuracy of the prediction




    '''
    # --- loop through model's predict_proba and print decision
    f = open("usc_1.txt",a)
    f.write()
    '''
  


def readfile (file):
    tmp_subj_id=[]
    tmp_date=[]
    tmp_info=[]
    tmp_text=[]

    dom = ElementTree.parse(file)
    writing = dom.findall('WRITING')

    for w in writing:
        title = w.find('TITLE').text
        date = w.find('DATE').text
        info = w.find('INFO').text
        text = w.find('TEXT').text

        # print('* {} - {} - {} - {}'.format(title,date,info,text))
        tmp_date.append(date)
        tmp_info.append(info)
        tmp_text.append(title + text)

    return tmp_subj_id, tmp_date, tmp_info, tmp_text



np.set_printoptions(threshold='nan') # --- display whole numpy array
pd.set_option('display.expand_frame_repr', False) # --- display whole dataframe
process_chunk()