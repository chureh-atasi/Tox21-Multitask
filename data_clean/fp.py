"""Script to fingerprint smiles"""
import os
import sys
import pandas as pd
from pgfingerprinting import fp

data_files = ['HEP_Liver_data_copy_csv.csv']
all_smiles = []
for dat_file in data_files:
    df = pd.read_csv(data_files)
    smiles = list(set(df.SMILES.values))
    new_smiles = [smile in smiles if smile not in all_smiles]
    all_smiles = all_smiles + new_smiles

params = {
    "fp_identifier": "fp_",
    "write_property": 0,
    "col_property": "",
    "normalize_a": 0,
    "normalize_b": 0,
    "normalize_m": 0,
    "normalize_e": 0,
    "block_list_version": "20201210",
    "ismolecule": 1,
    "polymer_fp_type": ["aT", "bT", "m", "e"],
    "calculate_side_chain": 0,
}
fps = []
curr = 0
for smile in all_smiles:
    if not smile != smile:
        string = '{} / {}: {}'.format(curr, len(smiles), smile)
        curr += 1
        os.system('clear')
        sys.stdout.write('\r%s' % string)
        sys.stdout.flush()
        try:
            fingerprint = fp.fingerprint_from_smiles(smile, params)
            fingerprint['SMILES'] = smile
            fps.append(fingerprint)
        except Exception as e:
            print(e)

save_df = pd.DataFrame(fps)
save_df = save_df.fillna(0)
save_df.to_csv('tox21_chem_fps.csv')
