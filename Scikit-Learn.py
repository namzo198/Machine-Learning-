#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:36:26 2019

@author: moses

The logistic regression model using the scikit-learn ML 
library on the banknote authentication dataset.
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#import scikitplot as skplt #to make things easy
from sklearn import preprocessing
from sklearn.decomposition import PCA  #reduce dimensions of dat

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve




"""
 This function prints and plots the confusion matrix. Normalization can be applied by setting 'normalize=True'
"""
def plot_confusion_matrix(cm,classes,title,normalize=False,cmap=plt.cm.Blues):
    #print(cm)
    #cm = confusion
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation =45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
  
    for i in range(cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j,i,format(cm[i,j],fmt),horizontalalignment="center", color="white" if cm[i,j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
    
def plot_roc(logit_roc_auc, y_pred_proba,y_test): #The receiver operating characteristic
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
    plt.figure()
    plt.plot(fpr,tpr,label='Logistic Regression (area = %0.2f)' %logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.axis('tight')
    plt.show()
 
    
def metrics(tn, fp,fn,tp):
    
    accuracy = (tp+tn)*100/ (tn+tp+fn+fp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 =  2*(1/((1/recall) + (1/precision)))
    specificity = tn/(tn+fp)
   # f1 = 2*((precision*recall)/(precision+recall))
    
    return accuracy, precision, recall, f1,specificity 

def data_exploration(Bank_Notes):
     #Viewing the data
    model = PCA(n_components = 2)
    model.fit(Bank_Notes)
    X_2D = model.transform(Bank_Notes)
    Bank_Notes['PCA1'] = X_2D[:,0]
    Bank_Notes['PCA2'] = X_2D[:,1]
    
    label = Bank_Notes['class']
    colors = ['red','green']
    #Visualize dataset
    plt.scatter( Bank_Notes['PCA1'], Bank_Notes['PCA2'], c=label, cmap= matplotlib.colors.ListedColormap(colors), label=label)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()
    
    print(Bank_Notes['class'].value_counts())
    sns.countplot(x='class', data=Bank_Notes, palette= colors)
    plt.show()
    

def main():
    
    Bank_Notes = pd.read_csv("data_banknote_authentication.csv",delimiter=",")
    
    #feature and target separation
    X = Bank_Notes.drop('class',axis = 1)
    Y = Bank_Notes['class']
    data_exploration(Bank_Notes)
   
   
    
   
    
    
   
    #Normalizing/standardizing the data
    X_scaled = preprocessing.scale(X)
    
    #Split the data_set in train(80%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,Y, test_size= 0.3, random_state = 1 )
    
    
    print("The train_set size is: ", len(X_train))
    print("The test_set size is: ", len(X_test))
    

    #Predicting using Logistic Regression for Binary Classification
    LR_model = LogisticRegression()
    LR_model.fit(X_train, y_train) #fitting the model
    y_pred = LR_model.predict(X_test) #prediction
   
   
    

   
    #Evaluation
    cnf_matrix = confusion_matrix(y_test, y_pred)
    tn, tp, fn, fp = confusion_matrix(y_test,y_pred).ravel()
    accuracy, precision, recall, f1, specificity = metrics(tn, tp, fn, fp)
    
    #Plot non-normalized confusion matrix
    #plt.figure
    plot_confusion_matrix(cnf_matrix,classes=['Forged','Authorized'], title='Confusion matrix, without normalization')
    
    
    print("True Negatives: ", tn)
    print("True Positives: ", tp)
    print("False Negatives: ", fn)
    print("False Positives: ", fp)
    
    print("Accuracy: {:0.2f}% ".format(accuracy))
    print("Precision: {:0.2f} ".format(precision))
    print("Recall: {:0.2f} ".format(recall))
    print("F1 Score: {:0.2f} ".format(f1))
    print("Specificity: {:0.2f} ".format(specificity))
    
     #ROC
    y_pred_proba = LR_model.predict_proba(X_test)
    #logit_roc_auc = roc_auc_score(y_test, y_pred)
    plot_roc(accuracy, y_pred_proba,y_test)
    
    
   
    
    
  
if __name__ == '__main__':
    main()