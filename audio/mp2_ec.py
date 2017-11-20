import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mp21_22 import BernoulliNB as nb
np.set_printoptions(precision=6)

training_path=os.getcwd()+'/txt_yesno/training'
training_labels_full=[]
for txt in os.listdir(training_path):
    new_txt=txt.replace(".txt", "")
    str_numr_txt=new_txt.replace("_", "")
    temp_list=[]
    for num in str_numr_txt:
        temp_list.append(int(num))
    training_labels_full.append(temp_list)

for labels in training_labels_full:
    print(labels)

path=training_path+"/0_0_0_0_1_1_1_1.txt"
txt_file=open(path, "r")
for line in txt_file.readlines():
    new_line=line.replace("\n", "")
    new_line=new_line.replace(" ", "1")
    new_line=new_line.replace("%", "0")
    print(len(new_line))
