"""Script to generate plots to anaylze fingerprints"""
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

def create_plots(df, savename):
  fp_cols = []
  for col in df.columns:
      if 'fp_' in col:
          fp_cols.append(col)

  # X[:,i] correspond to the columns, X[i] to the row
  X = df[fp_cols].values
  transformer = MaxAbsScaler().fit(X)
  X_trans = transformer.transform(X)
  fig, axs = plt.subplots(figsize=(12,9), nrows=2, ncols=2)
  xs_trans_less_than_one = []
  xs_trans_greater_than_one = []

  xs_less_than_one = []
  xs_greater_than_one = []
  for i in range(X.shape[1]):
      x = X_trans[:,i]
      if max(X[:,i]) > 1:
          xs_greater_than_one.append(X[:,i])
          xs_trans_greater_than_one.append(x)
      else:
          xs_less_than_one.append(X[:,i])
          xs_trans_less_than_one.append(x)

  axs[0][0].boxplot(xs_less_than_one)
  axs[1][0].boxplot(xs_trans_less_than_one)
  t1 = "Number of fingerprints: {}".format(len(xs_less_than_one))
  t2 = "Number of fingerprints: {}".format(len(xs_greater_than_one))
  axs[0][0].set_title(t1)
  axs[0][0].get_xaxis().set_visible(False)
  axs[0][1].get_xaxis().set_visible(False)
  axs[1][0].get_xaxis().set_visible(False)
  axs[1][1].get_xaxis().set_visible(False)
  axs[0][1].boxplot(xs_greater_than_one)
  axs[1][1].boxplot(xs_trans_greater_than_one)
  axs[0][1].set_title(t2)
  x = 0
  plt.savefig(savename, dpi=300)
  plt.show()

df = pd.read_csv(os.path.join('..', '..', 'tox21_models', 'data', 
    'HEP_Liver_fps.csv'))
new_save = 'fp_info.png'
create_plots(df, new_save)
