from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn import metrics
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import logging
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer
import matplotlib.pyplot as plt
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tox21_models.forest_model import report_forest_model
import forestci as fci


def model(df,y,pipeline = False, conf = False):
   ''' Runs the Random Forest Model

   Args:
      df(dataframe): a dataframe with just the features  
      y(list or dataframe):  The outcome of each row in df 
   
   Return:
      clf(RF model): return the model instance 
      y_test (list): the actual outcome list for the test set
      y_pred (list): the predicted outcome list for the test set
      rfc_cv_score (float): CV scores

   '''
   X_train, X_test, y_train, y_test = train_test_split(df, y, stratify = y, test_size=0.2)
   #y_actual = X_test.pop('CHANNEL_OUTCOME_ACTUAL')
   #X_train.pop('CHANNEL_OUTCOME_ACTUAL')
   clf=RandomForestClassifier(n_estimators=100, class_weight ='balanced')
   if (pipeline):
      pipe(clf, X_train,y_train,X_test,y_test) 

   clf.fit(X_train,y_train)
   y_pred=clf.predict(X_test)
   rfc_cv_score = cross_val_score(clf, df, y, cv=5)
   print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
   if (conf):
      confidence_results(clf, X_train, X_test, y_pred, y_test)

   return clf,y_test,y_pred,rfc_cv_score,X_test,X_train,y_train

def confidence_results(clf, X_train, X_test, y_pred, y_test):
   conf = fci.random_forest_error(clf, X_train, X_test)
   conf = 1- conf
   print("this is confidence" , len(conf))
   plt.plot(conf)
   plt.savefig("confidence.png", format = 'png')
   y_pred_conf = []
   mean = sum(conf)/len(conf)
   for x in range(len(conf)):
      if (conf[x]>(mean+0.1)) and y_pred[x] == 'inactive':
         y_pred_conf.append("conf inactive")
      elif( y_pred[x] == 'inactive'):
         y_pred_conf.append("inconlusive inactive")
      elif (conf[x]>(mean+0.1)) and y_pred[x] == 'active':
         y_pred_conf.append("conf active")
      elif( y_pred[x] == 'active'):
         y_pred_conf.append("inconlusive active")
   
   print(conf)
   conf_dict = {}
   df_conf = pd.DataFrame()
   conf_counter = 0
   for x in range(len(y_pred_conf)):
      print(conf[x] ,y_pred_conf[x], y_pred[x],y_test.iloc[x] )
      if ('conf' in y_pred_conf[x]):
         print ('confident')
         conf_counter += 1
         conf_dict['y_pred'] = y_pred[x]
         conf_dict['y_test'] = y_test.iloc[x]
         df_conf = df_conf.append(conf_dict, ignore_index=True)
   
   print("this is conf counter " , conf_counter)
   i = 0
   print(df_conf)
   for index,row in df_conf.iterrows():
      if (row['y_pred'] == 'inactive' and row['y_test'] == 'active'):
         i = i+1 
   
   print ('the amount of inaccurate inactive is', i)
   
   return y_pred_conf
def UMAP_model(reduced_feature, df,y, path):
   '''' Runs the same model but using the reduced feature from UMAP
   Creates classification reports
      Args:
      reduced_feature (dataframe): a dataframe with just the features  
      y(list or dataframe):  The outcome of each row in df 
   
   Return:
      y_test (list): the actual outcome list for the test set
      y_pred (list): the predicted outcome list for the test set
      X_test(df): dataframe with the reduced feature for the test set
   '''
   clf,y_test,y_pred,rfc_cv_score,X_test = model(reduced_feature,y)
   report_forest_model.model_report(y_test,y_pred, path)
   report_forest_model.cv_matrix(clf,y,df, path)
   return y_pred,y_test,X_test

def UMAP_RF_viz(X_test,X_train,y_test,y_pred,y_train):
   accuracy_test = []
   i = 0
   for x in y_pred:
      print (x, y_test.iloc[i])
      if (x == y_test.iloc[i]) and (y_test.iloc[i] == 'inactive'):
         accuracy_test.append(0)
      elif ((x != y_test.iloc[i]) and (y_test.iloc[i]  == 'inactive')):
         accuracy_test.append(1)
      elif (x == y_test.iloc[i]) and (y_test.iloc[i] =='active'):
         accuracy_test.append(2)
      elif(x != y_test.iloc[i]) and (y_test.iloc[i]  == 'active'):
         accuracy_test.append(3)
      i = i+1

   print (accuracy_test)
   accuracy_train = []
   for x in y_train:
      if (x == "active"):
         accuracy_train.append(4)
      else:
         accuracy_train.append(5)

   X_train['umap'] = accuracy_train
   X_test['umap'] = accuracy_test

   df_tot = pd.concat([X_train,X_test])
   class_dict = {'accurate inactive': 0 , 'inaccurate active': 1 , 'accurate active':2,'inaccurate inactive' : 3, 'train active' : 4, 'train inactive' : 5 }
   return df_tot, class_dict
def pipe(clf,X_train,y_train,X_test,y_test):
   '''Creates a pipeline of models
   yet to be completed
   '''
   umap = UMAP(random_state=456)
   pipeline = Pipeline([("umap", umap), ("randomforest", clf)])
   params_grid_pipeline = { "umap__n_neighbors": [100],
    "umap__n_components": [2], "umap__min_dist" : [0.5], "umap__metric":["manhattan"],
    "randomforest__n_estimators": [100]
   }
   clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline)
   clf_pipeline.fit(X_train, y_train)
   print(
    "Accuracy on the test set with UMAP transformation: {:.3f}".format(
        clf_pipeline.score(X_test, y_test)
    ))
   f= open("testcheck.txt","w+")
   f.write("The test accuracy is  {:.3f} ".format(
        clf_pipeline.score(X_test, y_test)))



def split_active(y_pred, y_actual,path):
   '''Function to split the predicted data into its constituents
   Saves plots 
   Args:
      path(string): The path to the directory to where this chart will be saved. 
      y_test (list): the actual outcome list for the test set.
      y_pred (list): the predicted outcome list for the test set.
      '''
   dict= {}
   #y_actual = X_test['CHANNEL_OUTCOME_ACTUAL']
   dict['agonist'] = 0
   dict['antagonist'] = 0
   for index, y in enumerate(y_pred):
      if y == ('active') and y_actual.iloc[index]=='agonist' :
         dict['agonist'] = dict['agonist'] + 1
      elif y == ('active') and y_actual.iloc[index]=='antagonist' :
          dict['antagonist'] = dict['antagonist'] + 1
   
   print (dict)
   total = dict['agonist'] + dict['antagonist']
   Outcome = list(dict.keys())
   Results = [(dict['agonist']/total)*100, (dict['antagonist']/total)*100]

  
   fig = plt.figure(figsize = (10, 5))
   # creating the bar plot
   plt.bar(Outcome, Results, color ='maroon', width = 0.4)
   
   plt.xlabel("Channel Outcome for Active")
   plt.ylabel("% Results")
   plt.title("Types of active outcomes")
   path = os.path.join(path, "'barchart.png'")
   plt.savefig(path)




   