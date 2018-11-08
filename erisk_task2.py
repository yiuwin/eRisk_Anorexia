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

def train_svm():

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
    #print(result)
    #print(type(result.iloc[1]), (result.iloc[1]).shape)


    # ---------------------------------------- build model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(result.text, result.target, test_size=0.005) # for personal testing (will not be passed on / used)

    

    X_train = result.loc[:, 'text']
    y_train = result.loc[:, 'target']

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
    '''

    # -
    cvec = CountVectorizer().fit(X_train)
    df_train = pd.DataFrame(cvec.transform(X_train).todense(),columns=cvec.get_feature_names())
    df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())

    model = svm.SVC(probability=True)
    model.fit(df_train, y_train)

    
    prediction = model.predict_proba(df_test)
    print(prediction) # what the machine predicts is the possibility of class 0 vs. class 1, respectively
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
    print(model.score(df_test, y_test)) # accuracy of the prediction



def train_lr():

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
    #print(result)
    #print(type(result.iloc[1]), (result.iloc[1]).shape)


    # ---------------------------------------- build model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(result.text, result.target, test_size=0.005)

    

    X_train = result.loc[:, 'text']
    y_train = result.loc[:, 'target']

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
    '''

    # -
    cvec = CountVectorizer().fit(X_train)
    df_train = pd.DataFrame(cvec.transform(X_train).todense(),columns=cvec.get_feature_names())
    df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())

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
    return model, cvec, lr



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


def test_lr(model, cvec, lr):
    # --- read chunks


    for c in range(10): #process chunk
        text_list = []
        id_list = []

        path = 'test/chunk' + str(c+1)
        test_file = [f for f in listdir(path) if isfile(join(path, f))] # TODO: does it matter that this is not in order?
        
        # for each subject file in chunk c
        for t in test_file:
            (id, text) = readfile(path + '/' + t)
            id_list.append(id)
            text_list.append(text)


        dict = {'id' : id_list, 'text' : text_list}
        result = pd.DataFrame(data=dict)
        #print(result)

        X_test = result.loc[:, 'text']
        #print(X_test.shape)

        df_test = pd.DataFrame(cvec.transform(X_test).todense(),columns=cvec.get_feature_names())
        prediction_proba = model.predict_proba(df_test)
        prediction = model.predict(df_test)
        print(prediction_proba) 
        #print(model.predict(df_test))
        #print( (model.predict(df_test))[2] )

        #print(X_test.shape)
        #print(prediction.shape)



        # --- open file
        f = open("usc_" + str(c+1) + ".txt", "a")

        i = 0
        for id in id_list:
            print(prediction_proba[i][0], prediction_proba[i][1])
            if (c==9): # chunk 10 - emit a decision regardless
                if (prediction[i] == 0):
                    f.write(id + '\t\t2\n')
                else:
                    f.write(id + '\t\t1\n')
            elif (c>7 and prediction_proba[i][0] < 0.75 and prediction_proba[i][1] < 0.75): # chunk 8,9
                f.write(id + '\t\t0\n')
            elif (c>5 and prediction_proba[i][0] < 0.85 and prediction_proba[i][1] < 0.85): # chunk 6,7
                f.write(id + '\t\t0\n')
            elif (prediction_proba[i][0] < 0.95 and prediction_proba[i][1] < 0.95): # chunk 3,4,5
                f.write(id + '\t\t0\n')
            else:
                if (prediction[i] == 0):
                    f.write(id + '\t\t2\n')
                else:
                    f.write(id + '\t\t1\n')

            i=i+1





# ----------------------------------------------------------------------------------------
# ------------------------------------- MAIN PROGRAM -------------------------------------
# ----------------------------------------------------------------------------------------

'''
python erisk_task2.py
python test/aggregate_results.py -path decisions/ -wsource test/writings-per-subject-all-test.txt
python test/erisk_eval.py -gpath test/risk-golden-truth-test.txt -ppath decisions/usc_global.txt -o 50

'''

np.set_printoptions(threshold='nan') # --- display whole numpy array
pd.set_option('display.expand_frame_repr', False) # --- display whole dataframe


# ------ Build model using logistic regression


#model, cvec, lr = train_lr()
train_svm()

# ------ Test
#test_lr(model, cvec, lr)

#prediction = model.predict_proba(df_test)
#print(lr.score(df_test, y_test)) # accuracy of the prediction


# test
#(id, result) = readfile('train/negative_examples/chunk1/subject31_1.xml')
#print(result)
