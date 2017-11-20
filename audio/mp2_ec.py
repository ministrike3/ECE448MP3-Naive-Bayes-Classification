import sys
import os
import numpy as np
import pandas as pd
import csv
import re
from statistics import mean
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mp21_22 import BernoulliNB
import mp21_22
np.set_printoptions(precision=6)

# ---------------------------------------------------------------------------------------
# MultinomialNB Class                                                                    |
# ---------------------------------------------------------------------------------------

class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)


# ---------------------------------------------------------------------------------------
# Utility Functions                                                                      |
# ---------------------------------------------------------------------------------------

def feature_tuner(training_data, length2, width):
    new_text=open(training_data, "r")
    count=0
    three_d_list=[]
    partial_list=[]
    for line in new_text.readlines():
        new_line=line.replace("\n", "")
        parsed_line=new_line.replace(" ", "1")
        parsed_line=parsed_line.replace("%","0")
        if (len(parsed_line)==width):
            for sym in parsed_line:
                partial_list.append(int(sym))
        if (len(partial_list)>0):
            three_d_list.append(partial_list)
            partial_list=[]

    train_no_data=[]
    total_train=[]
    length=len(three_d_list)
    count=0
    while(count<length):
        avg_line=mean(three_d_list[count])
        total_train.append(avg_line)
        count=count+1
        if(count%length2==0):
            train_no_data.append(total_train)
            total_train=[]

    return train_no_data

def binarize(input_data):
    length=len(input_data)
    for i in list(range(length)):
        for j in list(range(len(input_data[i]))):
            if (input_data[i][j]<0.5):
                input_data[i][j]=0
            else:
                input_data[i][j]=1

    return input_data

def label_grab():
    training_path=os.getcwd()+'/txt_yesno/training'
    training_labels_full=[]
    for txt in os.listdir(training_path):
        if ("txt" in txt):
            new_txt=txt.replace(".txt", "")
            new_txt=new_txt.replace(".csv", "")
            str_numr_txt=new_txt.replace("_", "")
            temp_list=[]
            for num in str_numr_txt:
                temp_list.append(int(num))
            training_labels_full.append(temp_list)

    label_data=[]
    for line in training_labels_full:
        for label in line:
            label_data.append(label)

    return label_data

def csv_gen():
    training_path=os.getcwd()+'/txt_yesno/training'
    for txt in os.listdir(training_path):
        if (".txt" in txt):
            path=training_path+"/"+txt
            txt_file=open(path, "r")
            notxt=txt.replace(".txt", "")
            three_d_list=[]
            partial_list=[]
            for line in txt_file.readlines():
                new_line=line.replace("\n", "")
                new_line=new_line.replace(" ", "1")
                new_line=new_line.replace("%", "0")

                for num in new_line:
                    partial_list.append(int(num))
                if(len(partial_list)==150):
                    three_d_list.append(partial_list)
                    partial_list=[]

            headers=[]
            for num in list(range(150)):
                headers.append("Label-"+str(num))

            three_d_list.insert(0, headers)
            with open(alt_path, 'w') as data:
                a=csv.writer(data)
                a.writerows(three_d_list)

def feature_grab():
    feature_data=[]
    training_path=os.getcwd()+'/txt_yesno/training'
    for csv in os.listdir(training_path):
        if (".csv" in csv):
            path=training_path+"/"+csv
            df=pd.read_csv(path)
            mat_1=df.iloc[:,10:20].as_matrix(columns=None).ravel().tolist()
            mat_2=df.iloc[:,20:30].as_matrix(columns=None).ravel().tolist()
            mat_3=df.iloc[:,30:40].as_matrix(columns=None).ravel().tolist()
            mat_4=df.iloc[:,40:50].as_matrix(columns=None).ravel().tolist()
            mat_5=df.iloc[:,50:60].as_matrix(columns=None).ravel().tolist()
            mat_6=df.iloc[:,60:70].as_matrix(columns=None).ravel().tolist()
            mat_7=df.iloc[:,70:80].as_matrix(columns=None).ravel().tolist()
            mat_8=df.iloc[:,80:90].as_matrix(columns=None).ravel().tolist()

            feature_data.append(mat_1)
            feature_data.append(mat_2)
            feature_data.append(mat_3)
            feature_data.append(mat_4)
            feature_data.append(mat_5)
            feature_data.append(mat_6)
            feature_data.append(mat_7)
            feature_data.append(mat_8)

    return feature_data

def test_data_gen(pathway):
    yesno_path=os.getcwd()+'/txt_yesno/'+pathway
    no_data=[]
    for txt in os.listdir(yesno_path):
        if (".txt" in txt):
            path=yesno_path+"/"+txt
            txt_file=open(path, "r")
            notxt=txt.replace(".txt", "")
            three_d_list=[]
            partial_list=[]
            for line in txt_file.readlines():
                new_line=line.replace("\n", "")
                new_line=new_line.replace(" ", "1")
                new_line=new_line.replace("%", "0")
                if (len(new_line)==10):
                    for sym in new_line:
                        partial_list.append(int(sym))
                if (len(partial_list)>0):
                    three_d_list.append(partial_list)
                    partial_list=[]

            test_no_data=[]
            total_train=[]
            length=len(three_d_list)
            count=0

            while(count<length):
                for symbol in three_d_list[count]:
                    total_train.append(symbol)
                count=count+1
                if(count%25==0):
                    test_no_data.append(total_train)
                    total_train=[]

            no_data.append(test_no_data)

    ret_data=[]
    for error in no_data:
        for real_list in error:
            ret_data.append(real_list)

    return ret_data



# ---------------------------------------------------------------------------------------
# Extra Credit Part 1                                                                    |
# ---------------------------------------------------------------------------------------

def part1():

    # Accumulating the training and testing labels and data
    train_labels=label_grab()
    train_features=feature_grab()
    test_labels=[0]*50 + [1]*50
    test_features=test_data_gen('no_test')+test_data_gen('yes_test')

    X_train=np.array(train_features)
    y_train=np.array(train_labels)
    X_test=np.array(test_features)
    y_test=np.array(test_labels)

    # Doing the machine learning
    nb=BernoulliNB(alpha=2).fit(X_train,y_train)
    predictions=nb.predict(X_test)

    # The end result
    print("Confusion Matrix: "+ "\n" +str(confusion_matrix(y_test, predictions)))
    print()
    print("Classification_report: "+ "\n" + str(classification_report(y_test, predictions)))
    print()
    print("Accuracy: "+str(accuracy_score(y_test, predictions)))



# ---------------------------------------------------------------------------------------
# Extra Credit Part 2                                                                    |
# ---------------------------------------------------------------------------------------

def part2(yesno=True):

    #Running the LDA on yesno corpus
    if (yesno==True):
        no_training=os.getcwd()+'/no/no_train.txt'
        yes_training=os.getcwd()+'/yes/yes_train.txt'
        no_testing=os.getcwd()+'/no/no_test.txt'
        yes_testing=os.getcwd()+'/yes/yes_test.txt'

        # Independent trainping data
        no_train=mp21_22.feature_tuner(no_training, 25, 10)
        yes_train=mp21_22.feature_tuner(yes_training, 25, 10)

        # Independent testing data
        no_test=mp21_22.feature_tuner(no_testing, 25, 10)
        yes_test=mp21_22.feature_tuner(yes_testing,25, 10)

        # Labels for each of the train data
        no_labels=[0]*131
        yes_labels=[1]*140

        # Labels for the test data
        no_test_labels=[0]*50
        yes_test_labels=[1]*50


        #Setting the parameters for training
        train=no_train+yes_train
        results=no_labels+yes_labels

        # Setting the parameters for the testing
        test=no_test+yes_test
        test_results=no_test_labels+yes_test_labels

        X_train=np.array(train)
        y_train=np.array(results)
        X_test=np.array(test)
        y_test=np.array(test_results)

        # Running the LDA model on the data
        lda_model=LDA(n_components=270)
        lda_model.fit_transform(X_train, y_train)
        predictions=lda_model.predict(X_test)

        # The end result
        print("Confusion Matrix: "+ "\n" +str(confusion_matrix(y_test, predictions)))
        print()
        print("Classification_report: "+ "\n" + str(classification_report(y_test, predictions)))
        print()
        print("Accuracy: "+str(accuracy_score(y_test, predictions)))

    # Running LDA on digits corpus
    else:
        training_data_path=os.getcwd()+'/data22/training_data.txt'
        training_labels_path=os.getcwd()+'/data22/training_labels.txt'
        testing_data_path=os.getcwd()+'/data22/testing_data.txt'
        testing_labels_path=os.getcwd()+'/data22/testing_labels.txt'

        # Getting the training and test data  from the data22 file
        train_data=mp21_22.feature_tuner(training_data_path, 30, 13)
        test_data=mp21_22.feature_tuner(testing_data_path, 30, 13)

        #print(len(train_data))
        #print(len(test_data))

        # Getting the training and test labels from the data22 file
        train_labels=mp21_22.label_extraction(training_labels_path)
        test_labels=mp21_22.label_extraction(testing_labels_path)

        #print(len(train_labels))
        #print(len(test_labels))

        X_train=np.array(train_data)
        y_train=np.array(train_labels)
        X_test=np.array(test_data)
        y_test=np.array(test_labels)

        # Running the LDA model on the data
        lda_model=LDA(n_components=59)
        lda_model.fit_transform(X_train, y_train)
        predictions=lda_model.predict(X_test)

        # The end result
        print("Confusion Matrix: "+ "\n" +str(confusion_matrix(y_test, predictions)))
        print()
        print("Classification_report: "+ "\n" + str(classification_report(y_test, predictions)))
        print()
        print("Accuracy: "+str(accuracy_score(y_test, predictions)))



# ---------------------------------------------------------------------------------------
# Extra Credit Part 3                                                                    |
# ---------------------------------------------------------------------------------------

def part3(bin_val=True):
    no_training=os.getcwd()+'/no/no_train.txt'
    yes_training=os.getcwd()+'/yes/yes_train.txt'
    no_testing=os.getcwd()+'/no/no_test.txt'
    yes_testing=os.getcwd()+'/yes/yes_test.txt'

    # Independent training data
    no_train=feature_tuner(no_training, 25, 10)
    yes_train=feature_tuner(yes_training, 25, 10)

    # Independent testing data
    no_test=feature_tuner(no_testing, 25, 10)
    yes_test=feature_tuner(yes_testing,25, 10)

    # Labels for each of the train data
    no_labels=[0]*131
    yes_labels=[1]*140

    # Labels for the test data
    no_test_labels=[0]*50
    yes_test_labels=[1]*50

    if (bin_val==True):

        # Binarizing the test and train data
        no_bin_train=binarize(no_train)
        yes_bin_train=binarize(yes_train)
        no_bin_test=binarize(no_test)
        yes_bin_test=binarize(yes_test)

        #Setting the parameters for training
        train_data=no_bin_train+yes_bin_train
        train_labels=no_labels+yes_labels

        # Setting the parameters for the testing
        test_data=no_bin_test+yes_bin_test
        test_labels=no_test_labels+yes_test_labels

        X_train=np.array(train_data)
        y_train=np.array(train_labels)
        X_test=np.array(test_data)
        y_test=np.array(test_labels)

        # Doing the machine learning
        bnb=BernoulliNB(alpha=5).fit(X_train,y_train)
        predictions=bnb.predict(X_test)

        # The end result
        print("Confusion Matrix: "+ "\n" +str(confusion_matrix(y_test, predictions)))
        print()
        print("Classification_report: "+ "\n" + str(classification_report(y_test, predictions)))
        print()
        print("Accuracy: "+str(accuracy_score(y_test, predictions)))

    else:

        #Setting the parameters for training
        train_data=no_train+yes_train
        train_labels=no_labels+yes_labels

        # Setting the parameters for the testing
        test_data=no_test+yes_test
        test_labels=no_test_labels+yes_test_labels

        X_train=np.array(train_data)
        y_train=np.array(train_labels)
        X_test=np.array(test_data)
        y_test=np.array(test_labels)

        # Doing the machine learning
        mnb=MultinomialNB(alpha=3).fit(X_train,y_train)
        predictions=mnb.predict(X_test)

        # The end result
        print("Confusion Matrix: "+ "\n" +str(confusion_matrix(y_test, predictions)))
        print()
        print("Classification_report: "+ "\n" + str(classification_report(y_test, predictions)))
        print()
        print("Accuracy: "+str(accuracy_score(y_test, predictions)))
