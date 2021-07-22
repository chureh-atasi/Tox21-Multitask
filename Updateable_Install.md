# Updateable Install

1. Add:
```
[build-system]
requires = ["setuptools", "poetry_core>=1.0"]
build-backend = "poetry.core.masonry.api"
```
to pyproject.toml

2. Run: 
```
poetry build --format sdist
# Linux
tar --wildcards -xvf dist/*-`poetry version -s`.tar.gz -O '*/setup.py' > setup.py
# Mac
tar -xvf dist/*-`poetry version -s`.tar.gz -O '*/setup.py' > setup.py
```
3. Run `poetry run pip install -e .` while in folder. 
