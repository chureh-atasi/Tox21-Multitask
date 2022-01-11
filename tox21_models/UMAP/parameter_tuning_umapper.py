"""Script to generate umaps. Not to be included in the gui."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
import os
from multiprocessing import Pool
import sys
#sys.path.append('/data/chureh/tox21_models/tox21_models/forest_model')
#import Forest


save_folder = "Umaps_tox21_binary_super_1"
contents = os.listdir(save_folder)
dim = 2

def umapping(df):
    for index, row in df.iterrows():
        neigh = row['nearest neighbors']
        dist = row['minimum distance']
        met = row['metric']
        name = ('umap_' + str(index) + "_neigh_" + str(neigh)
               + "_mindist_"+str(dist)+"_metric_"
               + str(met)+'.png')
        try:
            if name not in contents:
                title = ("Nearest Neighbors: " + str(neigh) 
                         + ", Minimum Distance: " + str(dist)
                         + ", Metric: " + met)
                img_name = os.path.join('./',save_folder, name)
                print(img_name)
                fit = umap.UMAP(
                                n_neighbors=neigh,
                                min_dist=dist,
                                metric=met,
                                n_components=dim
                               ).fit(train.values, y=target )
                train_fit = fit.transform(train.values)
                RandomForest(train_fit)
                if (dim ==2):
                    plot_2d(train_fit,title,img_name)
                if (dim ==3):
                    plot_3d(train_fit,title,img_name)
        except Exception as e:
            print('For ' + name)
            print(str(e))
    

def RandomForest(train):
    print (train)

def plot_2d(train_fit,title,img_name):
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(train_fit[:,0], train_fit[:,1], s=7,c=target, alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange((len(classes)+1)))
    ticks = np.arange(len(classes))
    ticks = [x + .5 for x in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(classes)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_name, format='png')

def plot_3d(train_fit,title,img_name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(train_fit[:,0], train_fit[:,1],train_fit[:,2] ,s=7,c=target, alpha=1.0)
    #plt.setp(ax, xticks=[], yticks=[])
    plt.colorbar(p)
    #ticks = np.arange(len(classes))
    #ticks = [x + .5 for x in ticks]
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels(classes)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_name, format='png')



df = pd.read_csv("/data/chureh/tox21_models/tox21_models/forest_model/df_actual_outcome_binary.csv")
df.pop('CHANNEL_OUTCOME_ACTUAL')
df = df.fillna(0)
curr = 0

'''
for index, row in df.iterrows():
    string = '{} / {}'.format(curr, df.shape[0])
    curr += 1
    sys.stdout.write('\r%s' % string)
    sys.stdout.flush()
    if  row["CHANNEL_OUTCOME_ACTUAL"] == 0:
        df.loc[index, 'CHANNEL_OUTCOME_ACTUAL'] = 'inactive'
        '''
headers = []
for header in df.columns:
    if 'fp_' in header:
        headers.append(header)
train = df[headers]
train = train.fillna(0)

classes = [x.split(',')[0].split('/')[0] for x in df['CHANNEL_OUTCOME']]
df['Reduced Class'] = classes
classes = np.unique(classes)
classes = [x for x in classes]
target = [classes.index(x) for x in df['Reduced Class']]
save_df = pd.DataFrame()

def parameter_tuning():
    #n_neighbors = [5, 10, 15, 25, 50, 100, 200]
    n_neighbors = [5, 10]
    #min_dist = [0, .1, .25, .5, .8, .99]
    min_dist = [0, .1]
    """
    metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra',
            'braycurtis', 'mahalanobis', 'wminkowski',
            'cosine', 'correlation']
    """
    metric = ['manhattan', 'cosine']

    runs = {}
    count = 0
    for dist in min_dist:
        for neigh in n_neighbors:
            for met in metric:
                run_dict = {
                            'nearest neighbors': neigh,
                            'minimum distance': dist,
                            'metric': met
                        }
                runs[count] = run_dict
                count += 1
    run_df = pd.DataFrame.from_dict(runs)
    run_df = run_df.transpose()
    run_df = run_df.sample(frac=1)
    num_cores = min(os.cpu_count(), 30)
    df_split = np.array_split(run_df, num_cores)
    pool = Pool(num_cores)
    pool.map(umapping, df_split)
    pool.close()
    pool.join()
