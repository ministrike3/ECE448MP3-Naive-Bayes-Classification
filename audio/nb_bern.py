import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

np.set_printoptions(precision=6)

class BernoulliNB(object):
    def __init__(self, alpha=1.0):
        self.alpha=alpha

    def fit(self,X, y):
        count_sample=X.shape[0]
        seperated=[[x for x, t in zip(X,y) if t==c] for c in np.unique(y)]
        self.class_log_prior_=[np.log(len(i)/count_sample) for i in seperated]
        count=np.array([np.array(i).sum(axis=0) for i in seperated]) + self.alpha
        smoothing=2*self.alpha
        n_doc = np.array([len(i) + smoothing for i in seperated])
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self

    def predict_log_proba(self, X):
        return [(np.log(self.feature_prob_) * x + np.log(1 - self.feature_prob_) * np.abs(x - 1)).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

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
                partial_list.append(sym)
        if (len(partial_list)>0):
            three_d_list.append(partial_list)
            partial_list=[]

    train_no_data=[]
    total_train=[]
    length=len(three_d_list)
    count=0

    while(count<length):
        for symbol in three_d_list[count]:
            total_train.append(symbol)
        count=count+1
        if(count%length2==0):
            train_no_data.append(total_train)
            total_train=[]


    for i in list(range(len(train_no_data))):
        for j in list(range(len(train_no_data[i]))):
            train_no_data[i][j]=int(train_no_data[i][j])

    return train_no_data

def label_extraction(data_path):
    labels=[]
    data_labels_train=open(data_path, "r")
    for line in data_labels_train.readlines():
        num_line=line.replace('\n', "")
        number=int(num_line)
        labels.append(number)

    return labels



def part_1():
    no_training=os.getcwd()+'/no/no_train.txt'
    yes_training=os.getcwd()+'/yes/yes_train.txt'
    no_testing=os.getcwd()+'/no/no_test.txt'
    yes_testing=os.getcwd()+'/yes/yes_test.txt'

    # Independent trainping data
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

    # Doing the machine learning
    nb=BernoulliNB(alpha=1).fit(X_train,y_train)
    predictions=nb.predict(X_test)

    # The end result
    print("Confusion Matrix: "+ "\n" +str(confusion_matrix(y_test, predictions)))
    print()
    print("Classification_report: "+ "\n" + str(classification_report(y_test, predictions)))
    print()
    print("Accuracy: "+str(accuracy_score(y_test, predictions)))


def part_2():

    training_data_path=os.getcwd()+'/data22/training_data.txt'
    training_labels_path=os.getcwd()+'/data22/training_labels.txt'
    testing_data_path=os.getcwd()+'/data22/testing_data.txt'
    testing_labels_path=os.getcwd()+'/data22/testing_labels.txt'

    # Getting the training and test data  from the data22 file
    train_data=feature_tuner(training_data_path, 30, 13)
    test_data=feature_tuner(testing_data_path, 30, 13)

    #print(len(train_data))
    #print(len(test_data))

    # Getting the training and test labels from the data22 file
    train_labels=label_extraction(training_labels_path)
    test_labels=label_extraction(testing_labels_path)

    #print(len(train_labels))
    #print(len(test_labels))

    X_train=np.array(train_data)
    y_train=np.array(train_labels)
    X_test=np.array(test_data)
    y_test=np.array(test_labels)

    # Doing the machine learning
    nb=BernoulliNB(alpha=1).fit(X_train,y_train)
    predictions=nb.predict(X_test)

    # Tuning the predictions by adding 1
    for i in list(range(len(predictions))):
        predictions[i]=predictions[i]+1

    # The end result
    print("Confusion Matrix: "+ "\n" +str(confusion_matrix(y_test, predictions)))
    print()
    print("Classification_report: "+ "\n" + str(classification_report(y_test, predictions)))
    print()
    print("Accuracy: "+str(accuracy_score(y_test, predictions)))


if __name__=="__main__":

    # Depending on which part you want to run, make sure to comment out th other part
    #part_1()
    part_2()
