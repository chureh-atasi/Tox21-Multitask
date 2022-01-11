import pandas as pd
from sklearn import metrics
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import matplotlib.pyplot as plt

def model_report(y_test, y_pred,path,label):
    ''' Creates confusion Matrix and Classificaiton report for the test set 
    Ouputs them in CSV files

    Args:
      y_test (list): the actual outcome list for the test set.
      y_pred (list): the predicted outcome list for the test set.
      path(string): The path to the directory to where this chart will be saved. 
      label(list): A list of the labels in the correct order used to print the matrix.
    '''
    label = ['0', '1']
    matrix = confusion_matrix(y_test, y_pred)
    ## Use the Axes attribute 'ax_' to get to the underlying Axes object.
    ## The Axes object controls the labels for the X and the Y axes. It
    ## also controls the title.

    DF = pd.DataFrame(matrix)
    label_pred = []
    label_actual = []
    for lab in label:
        pred_label = ('pred-' + lab)
        true_label = ('actual-' + lab)
        label_pred.append(pred_label)
        label_actual.append(true_label)

    DF = DF.set_axis(label_pred, axis=1, inplace=False)
    DF = DF.set_axis(label_actual, axis=0, inplace=False)
    matrix  = os.path.join(path, "confusion_matrix.csv")
    report = os.path.join(path, "class_report.csv")
    DF.to_csv(matrix)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    DF = pd.DataFrame(class_report).transpose()
    DF.to_csv(report)


def cv_matrix(clf,y,df, path,label):
    ''' Creates Confusion Matrices for 5 CV runs. Outputs them in one CSV file
    
   Args:
        clf(RF model): return the model instance.
        y(list or dataframe):  The outcome of each row in df.
        df(dataframe): a dataframe with just the features.
        path(string): The path to the directory to where this chart will be saved. 
         label(list): A list of the labels in the correct order used to print the matrix.
      
    '''
    cv_path = os.path.join(path, "cv_report.csv")
    skf = StratifiedKFold(n_splits=5)
    #kf = KFold(n_splits=5)
    i = 0
    label_pred = []
    label_actual = []
    for lab in label:
        pred_label = ('pred-' + lab)
        true_label = ('actual-' + lab)
        label_pred.append(pred_label)
        label_actual.append(true_label)
        dict = {}
    for train_index, test_index in skf.split(df, y):

        X_train_CV, X_test_CV = df.iloc[train_index].copy(), df.iloc[test_index].copy()
        y_train_CV, y_test_CV = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        clf.fit(X_train_CV, y_train_CV)
        #print("Accuracy:",metrics.accuracy_score(y_test_CV, clf.predict(X_test_CV)))
        matrix = confusion_matrix(y_test_CV, clf.predict(X_test_CV), labels = label)
        DF = pd.DataFrame(matrix)
        DF = DF.set_axis(label_pred, axis=1, inplace=False)
        DF = DF.set_axis(label_actual, axis=0, inplace=False)
        dict["df" + str(i)] = DF
        i = i +1

    with open(cv_path,'a') as f:
        for df in dict.values() :
            df.to_csv(f)
            f.write("\n")