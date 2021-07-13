"""Script to fingerprint smiles"""
import os
import sys
import pandas as pd
from pgfingerprinting import fp

df = pd.read_csv('HEP_Liver_data_copy_csv.csv')

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
smiles = list(set(df.SMILES.values))
curr = 0
for smile in smiles:
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
save_df.to_csv('HEP_Liver_fps.csv')
