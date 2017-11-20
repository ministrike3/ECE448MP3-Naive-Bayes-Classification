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

def feature_tuner(training_data):
    new_text=open(training_data, "r")
    count=0
    three_d_list=[]
    partial_list=[]
    for line in new_text.readlines():
        new_line=line.replace("\n", "")
        parsed_line=new_line.replace(" ", "1")
        parsed_line=parsed_line.replace("%","0")
        if (len(parsed_line)==10):
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
        if(count%25==0):
            train_no_data.append(total_train)
            total_train=[]


    for i in list(range(len(train_no_data))):
        for j in list(range(len(train_no_data[i]))):
            train_no_data[i][j]=int(train_no_data[i][j])

    return train_no_data

if __name__=="__main__":
    no_training=os.getcwd()+'/no/no_train.txt'
    yes_training=os.getcwd()+'/yes/yes_train.txt'
    no_testing=os.getcwd()+'/no/no_test.txt'
    yes_testing=os.getcwd()+'/yes/yes_test.txt'

    # Independent trainping data
    no_train=feature_tuner(no_training)
    yes_train=feature_tuner(yes_training)

    # Independent testing data
    no_test=feature_tuner(no_testing)
    yes_test=feature_tuner(yes_testing)

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
