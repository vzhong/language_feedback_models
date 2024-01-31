# Verbalized Envs

### Install

```
pip install -e .[touchdown,scienceworld,alfworld]
```

You need to also download the data to `./data` using `python bin/download_data.py` and then set `VERBENVS_DATA=$PWD/data`.
After this you can check that the environments work via:

```
python -m unittest discover tests
```
