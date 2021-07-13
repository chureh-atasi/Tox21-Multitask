============
TOX21 Models
============

This code will be used to generate `tox21 <https://tox21.gov>`_ models for
Polymer Genome.

Progress
--------

- [] Simple neural network for HEP Liver channel outcomes.

  - [X] Fingerprint polymers.
  - [X] Clean data.

  - [] Create model infrastructure.

  - Data Caveats:

    - Inconclusive outcomes assumed to be conclusive for model predictions.
      
    - Only one datapoint is used per chemical, others are thrown out.

    - If a chemical has both inactive and antagonist/agonist channel outcomes, 
      antagonist/agonist chosen for single datapoint.

    - If a chemical has both antagonist and agonist channel outcome, 
      antagonist chosen for datapoint.

- [] Create multitask model for all assays.
