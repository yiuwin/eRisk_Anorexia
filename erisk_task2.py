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




def process_chunks():

    # ----------------------------------------------------------------------------------------
    # ----------------------------------- NEGATIVE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------

    chunks = [f for f in listdir('train/negative_examples/')] # negative - 132
    frames_neg = []

    for c in chunks:
        path = 'train/negative_examples/' + c
        negative = [f for f in listdir(path) if isfile(join(path, f))]

        for n in negative:
            (id, text) = readfile('train/negative_examples/'+c+'/'+n)

            # -- create NEGATIVE dataframe
            df = {'text' : text}
            df = pd.DataFrame(data=[df])

            # --
            frames_neg.append(df)


    result_neg = pd.concat(frames_neg)

    # - append target feature
    target = [0] * result_neg.shape[0]
    result_neg['target'] = target


    # ----------------------------------------------------------------------------------------
    # ----------------------------------- POSITIVE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------
    chunks = [f for f in listdir('train/positive_examples/')] # negative - 132
    frames_pos = []

    for c in chunks:
        path = 'train/positive_examples/' + c
        positive = [f for f in listdir(path) if isfile(join(path, f))]

        for p in positive:
            (id, text) = readfile('train/positive_examples/'+c+'/'+p)

            # -- create NEGATIVE dataframe
            df = {'text' : text}
            df = pd.DataFrame(data=[df])

            # --
            frames_pos.append(df)


    result_pos = pd.concat(frames_pos)

    # - append target feature
    target = [1] * result_pos.shape[0]
    result_pos['target'] = target

  
    # ----------------------------------------------------------------------------------------
    # ------------------------------------ COMBINE CHUNKS ------------------------------------
    # ----------------------------------------------------------------------------------------
    result = pd.concat([result_neg, result_pos])
    print(result)


    # ---------------------------------------- build model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(result.text, result.target, test_size=0.05)

    
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
        print (x,end=' ')
    #print((y_test).shape)

    print(X_train.shape) 
    print(y_train.shape) 
    
    print(X_test.shape)
    print((y_test).shape)
    '''

    print('\n')
    # -
    cvec = CountVectorizer().fit(X_train)
    df_train = pd.DataFrame(cvec.transform(X_train).todense(),columns=cvec.get_feature_names())
    df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())

    print(df_train.shape)
    print(y_train.shape)
    print(df_test.shape)
    print(y_test.shape)
    # - pass data to classifier after using tfidftransformer
    lr = LogisticRegression()
    model = lr.fit(df_train, y_train)
    prediction = model.predict_proba(df_test)
    #print(prediction) # what the machine predicts is the possibility of class 0 vs. class 1, respectively
    #print( type(model.predict_proba(df_test)) )


    '''
    # ----------- TESTING AND EVALUATION STAGE

    # --- open a file
    f = open("usc_1.txt", "a")

    # --- loop over predictions and print 
    for i in range(prediction.shape[0]):
        if (prediction[i][0] < 0.95 and prediction[i][1] < 0.95 ):
            f.write()

    print(prediction[0][0], prediction[0][1])


    print(model.predict(df_test)) # what the machine predicts are the test values
    #print((model.predict(df_test)).shape)

    '''
    print(lr.score(df_test, y_test)) # accuracy of the prediction



def readfile (file):
    result = ''

    dom = ElementTree.parse(file)
    tmp_subj_id = dom.find('ID').text
    writing = dom.findall('WRITING')

    for w in writing:
        title = w.find('TITLE').text
        text = w.find('TEXT').text

        # print('* {} - {} - {} - {}'.format(title,date,info,text))
        result = result + title + text

    return tmp_subj_id, result



np.set_printoptions(threshold='nan') # --- display whole numpy array
pd.set_option('display.expand_frame_repr', False) # --- display whole dataframe
process_chunks()

'''
chunks = [f for f in listdir('train/negative_examples/')] # negative - 132
for c in chunks:
    path = 'train/negative_examples/' + c
    negative = [f for f in listdir(path) if isfile(join(path, f))] # negative - 132
    for n in negative:
        (id, text) = readfile(path + '/' + n)
        print(id)

'''


# test
#(id, result) = readfile('train/negative_examples/chunk1/subject31_1.xml')
#print(result)
