============
TOX21 Models
============

This code will be used to generate `tox21 <https://tox21.gov>`_ models for
Polymer Genome.

Progress
--------

- U+2610 Simple neural network for HEP Liver channel outcomes.

  - U+2611 Fingerprint polymers.

  - U+2611 Clean data.

  - U+2610 Create model infrastructure.

  - Data Caveats:

    - Inconclusive outcomes assumed to be conclusive for model predictions.
      
    - Only one datapoint is used per chemical, others are thrown out.

    - If a chemical has both inactive and antagonist/agonist channel outcomes, 
      antagonist/agonist chosen for single datapoint.

    - If a chemical has both antagonist and agonist channel outcome, 
      antagonist chosen for datapoint.

- U+2610 Create multitask model for all assays.
