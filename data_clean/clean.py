"""Script to determine data to keep for analysis"""
import logging
import sys
import pandas as pd

df = pd.read_csv('HEP_Liver_data_copy_csv.csv')
fp_df = pd.read_csv('HEP_Liver_fps.csv')

fp_smiles = list(set(fp_df.SMILES.values))

# Drop any molecules that failed to fingerprint or that were not listed
df = df.loc[df.SMILES.isin(fp_smiles)]

# Display possible outcomes
# potential_outcomes = list(set(df.CHANNEL_OUTCOME))
# print(potential_outcomes)
# ['inconclusive agonist', 'active antagonist', 'inconclusive antagonist', 
# 'inactive', 'active agonist']


# Columns we want for analysis
# cols_to_keep = ['CHANNEL_OUTCOME', 'SMILES']

clean_data = []
# There are five types of outcomes, but let's assume inconclusive is conclusive
curr = 0
for smiles in fp_smiles:
    string = '{} / {}'.format(curr, len(fp_smiles))
    curr += 1
    sys.stdout.write('\r%s' % string)
    sys.stdout.flush()
    data = {} 
    data['SMILES'] = smiles
    tdf = df.loc[df['SMILES'] == smiles]
    if ('active antagonist' in tdf.CHANNEL_OUTCOME.values or 
            'inconclusive antagonist' in tdf.CHANNEL_OUTCOME.values):
        data['CHANNEL_OUTCOME'] = 2
    elif ('active agonist' in tdf.CHANNEL_OUTCOME.values or 
            'inconclusive agonist' in tdf.CHANNEL_OUTCOME.values):
        data['CHANNEL_OUTCOME'] = 1
    elif ('inactive' in tdf.CHANNEL_OUTCOME.values):
        data['CHANNEL_OUTCOME'] = 0
    else:
        logging.warning("No CHANNEL_OUTCOME recorded")

    clean_data.append(data.copy())
print()

# Downsides: throwing out datapoints, not considering time factor, 
#            inconclusive assumption may be inaccurate, not using efficacy or
#            AC50 values. Need to keep these in mind in the future.
save_df = pd.DataFrame(clean_data)
save_df.to_csv('HEP_Liver_clean_data.csv', index=False)
