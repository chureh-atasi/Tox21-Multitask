# TOX21 Models

This code will be used to generate [tox21](https://tox21.gov) models for
Polymer Genome with PyTorch.

## Progress

- [x] Add Chureh to github repository

- [ ] Simple neural network for HEP Liver channel outcomes.

  - [x] Fingerprint chemicals.

  - [x] Clean data.

  - [x] Create model infrastructure.

  - [ ] Add MaxAbsScaler.

  - Data Caveats:

    - Inconclusive outcomes assumed to be conclusive for model predictions.
      
    - Only one datapoint is used per chemical, others are thrown out.

    - If a chemical has both inactive and antagonist/agonist channel outcomes, 
      antagonist/agonist chosen for single datapoint.

    - If a chemical has both antagonist and agonist channel outcome, 
      antagonist chosen for datapoint.

- [ ] Create multitask model for all assays.


## Poetry Install

1. If you don't have poetry installed

    1. Install [poetry](https://python-poetry.org/docs/)

2. If you don't have a python 3.7.11 version installed... 

    1. `pyenv install 3.7.11`
  
    2. `poetry env use /path/to/version/3.7.11/python3.7` 

    3. If pyenv fails, try using miniconda to manage python env instead

        1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

        2. `conda create --name py37 python=3.7`

        3. `poetry env use /path/to/version/3.7.11/python3.7` 

3. Clone github folder

4. `cd tox21_models`

5. `poetry install`
