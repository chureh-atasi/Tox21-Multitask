import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
import os
from multiprocessing import Pool, Value
import sys
import os 
import seaborn as sns
from tox21_models.forest_model import Forest

tox21_folder = os.path.dirname(os.path.realpath(__file__))
file_name = "figures_UMAP/Umaps_tox21_binary_super_seaborn"
save_folder = os.path.join(tox21_folder, file_name)

contents = os.listdir(save_folder)
dim = 2
neigh = 200
dist = 0.5
met = 'manhattan'
name_1 = 'UMAP_4'

def umapping(df_train,df_target,classes, class_name = None):
    name = (name_1 + "_neigh_" + str(neigh)
               + "_mindist_"+str(dist)+"_metric_"
               + str(met)+'.png')
    title = ("Nearest Neighbors: " + str(neigh) 
            + ", Minimum Distance: " + str(dist)
            + ", Metric: " + met)
    img_name = os.path.join('./',save_folder, name)
    print(name)
    fit = umap.UMAP(
            n_neighbors=neigh,
            min_dist=dist,
            metric=met,
            n_components=dim
            ).fit(df_train.values)
    #fit train
    #embed (fit.transform) - test do test_train_split
    train_fit = fit.transform(df_train.values)
    #RandomForest(train_fit,title,img_name)
    
    if (dim ==2):
        plot_2d(train_fit,title,img_name, classes, target_1 =df_target, class_name = class_name)
    if (dim ==3):
        plot_3d(train_fit,title,img_name, classes, target_1 = df_target)


def RandomForest(train_fit, title,img_name, df_target):
    print(" we are now starting RF")
    name = ("RF_UMAP_2" + "_neigh_" + str(neigh)
        + "_mindist_"+str(dist)+"_metric_"
        + str(met)+'.png')
    img_name = os.path.join('./',save_folder, name)
    train_fit_df = pd.DataFrame(train_fit, columns = ['Column_A','Column_B'])
    #target_df = pd.DataFrame(target, columns=['Outcome'])
    y_pred,y_test,X_test = Forest.UMAP_model(train_fit_df,df_target)
    print("We are done with RF")
    print (X_test.shape)
    plot_2d_model(X_test, title, img_name,y_pred,y_test)
    


def plot_2d_model(X_test, title, img_name, y_pred,y_test):
    model_target = []
    classes_model = [] 
    index = 0
    for value in y_pred:
        if (value == y_test[index]):
            model_target.append(1)
        else:
            model_target.append(0)
        index = index +1
        
    for val in model_target:
        if val == 1:
            classes_model.append('Accurate')
        else:
            classes_model.append("Inaccurate")

    plot_2d(X_test,title,img_name,classes_model,target_1 = model_target)

def plot_2d(train_fit,title,img_name, classes_1, target_1 = None, class_name = None):
    try:
        train_fit = train_fit.to_numpy()
    except AttributeError:
        pass
    print(train_fit)
    print('This is type of train_fit',type(train_fit))
    train_fit_df = pd.DataFrame(train_fit, columns = ['Column_A','Column_B'])
    extracted_col = target_1.to_numpy()
    train_fit_df['umap'] = extracted_col
    for index,row in train_fit_df.iterrows():
        if (row['umap'] == 0) :
            train_fit_df.loc[index,'umap_words'] = 'accurate inactive'
            train_fit_df.loc[index,'activity'] = 'inactive'
        elif (row['umap'] == 1):
           train_fit_df.loc[index,'umap_words'] = 'inaccurate active'
           train_fit_df.loc[index,'activity'] = 'inactive'
        elif (row['umap'] == 2):
            train_fit_df.loc[index,'umap_words'] = 'accurate active'
            train_fit_df.loc[index,'activity'] = 'active'
        elif(row['umap'] == 3):
            train_fit_df.loc[index,'umap_words'] = 'inaccurate inactive'
            train_fit_df.loc[index,'activity'] = 'active'
        elif(row['umap'] == 4):
             train_fit_df.loc[index,'umap_words'] = ' train active'
             train_fit_df.loc[index,'activity'] = 'active'
        elif(row['umap'] == 5):   
            train_fit_df.loc[index,'umap_words'] = 'train inactive'
            train_fit_df.loc[index,'activity'] = 'inactive'

    size_dict = {'active': 40, 'inactive': 10}
    markers = {"active": "X", "inactive": "o"}
    sns.relplot(x= 'Column_A', y='Column_B', data = train_fit_df, hue='umap_words',
        style='activity', size = 'activity', sizes = size_dict, markers = markers)
    #sns.relplot(x= train_fit[:,0], y=train_fit[:,1], hue=1, style=1)
    plt.title(title)
    plt.savefig(img_name, format='png')
    '''
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(train_fit[:,0], train_fit[:,1], s=7,c=target_1, alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange((len(classes_1)+1)),ax = ax)
    ticks = np.arange(len(classes_1))
    ticks = [x + .5 for x in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(class_name.keys()))
    #fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_name, format='png')
    '''

def plot_3d(train_fit,title,img_name,classes_1, target_1 = None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(train_fit[:,0], train_fit[:,1],train_fit[:,2] ,s=7,c=target_1, alpha=1.0)
    #plt.setp(ax, xticks=[], yticks=[])
    plt.colorbar(p)
    #ticks = np.arange(len(classes))
    #ticks = [x + .5 for x in ticks]
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels(classes)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_name, format='png')
