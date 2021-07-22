# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tox21_models', 'tox21_models.utils']

package_data = \
{'': ['*'],
 'tox21_models': ['data/Clean_Tox_Data.csv',
                  'data/Clean_Tox_Data.csv',
                  'data/tox21_chem_fps.csv',
                  'data/tox21_chem_fps.csv']}

install_requires = \
['keras-tuner>=1.0.2,<2.0.0',
 'kerastuner-tensorboard-logger>=0.2.3,<0.3.0',
 'matplotlib>=3.4.2,<4.0.0',
 'pandas>=1.2.4,<2.0.0',
 'pgfingerprinting @ '
 'git+ssh://git@github.com/Ramprasad-Group/pgfingerprinting.git@main',
 'pyarrow>=4.0.1,<5.0.0',
 'scikit-learn>=0.24.2,<0.25.0',
 'seaborn>=0.11.1,<0.12.0',
 'tensorflow-addons>=0.12.1,<0.13.0',
 'tensorflow>=2.4.1,<2.5']

setup_kwargs = {
    'name': 'tox21-models',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'jdkern11',
    'author_email': 'leolion29@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.1,<3.8.0',
}


setup(**setup_kwargs)
