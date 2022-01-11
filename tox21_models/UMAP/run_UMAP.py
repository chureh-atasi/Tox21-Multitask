import pandas as pd
import numpy as np
from tox21_models.forest_model import Forest
from tox21_models.UMAP import UMAP_runner


df = pd.read_csv("/data/chureh/tox21_models/tox21_models/forest_model/df_actual_outcome_binary.csv")
df.pop('CHANNEL_OUTCOME_ACTUAL')
y = df.pop['CHANNEL_OUTCOME']
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
print(len(target))
save_df = pd.DataFrame()
UMAP_runner.umapping(df,y)
"""
save_df['PID'] = df['PID']
save_df['X'] = train_fit[:,0]
save_df['Y'] = train_fit[:,1]
save_df.to_csv('reference_polymers_umapping.csv', index=False)
"""
