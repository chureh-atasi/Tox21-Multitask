"""Script to determine data to keep for analysis"""
import logging
import sys
import pandas as pd

# Will change this in the future to just be the fps of all molecules
fp_df = pd.read_csv('tox21_chem_fps.csv')
fp_smiles = list(set(fp_df.SMILES.values))


def clean_data(path: str, assay_type: str, all_df: pd.DataFrame):
    """Cleans each tox21 assay
    
    Args:   
        path (str):  
            Path to datafile  
        assay_type (str):  
            Name for the tox21 assay
        all_df (pd.DataFrame):  
            dataframe to store all data
    """
    df = pd.read_csv(path)


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
        tdf = df.loc[df['SMILES'] == smiles]
        if len(tdf) != 0:
            data = {} 
            data['SMILES'] = smiles
            data['prop'] = assay_type
            if ('active antagonist' in tdf.CHANNEL_OUTCOME.values or 
                    'inconclusive antagonist' in tdf.CHANNEL_OUTCOME.values):
                data['CHANNEL_OUTCOME'] = 'antagonist'
            elif ('active agonist' in tdf.CHANNEL_OUTCOME.values or 
                    'inconclusive agonist' in tdf.CHANNEL_OUTCOME.values):
                data['CHANNEL_OUTCOME'] = 'agonist'
            elif ('inactive' in tdf.CHANNEL_OUTCOME.values):
                data['CHANNEL_OUTCOME'] = 'inactive'
            else:
                logging.warning("No CHANNEL_OUTCOME recorded")

            clean_data.append(data.copy())
    print()

    # Downsides: throwing out datapoints, not considering time factor, 
    #            inconclusive assumption may be inaccurate, not using efficacy or
    #            AC50 values. Need to keep these in mind in the future.
    append_df = pd.DataFrame(clean_data)
    all_df = all_df.append(append_df)
    return all_df

# All data will be appended here
all_df = pd.DataFrame(columns=['SMILES', 'CHANNEL_OUTCOME', 'assay_type']) 
files = {'Liver': 'HEP_Liver_data_copy_csv.csv'}
for assay_type in files:
    all_df = clean_data(files[assay_type], assay_type, all_df.copy())

all_df.to_csv('Clean_Tox_Data.csv', index=False)
