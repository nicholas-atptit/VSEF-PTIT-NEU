# vnstock_data Environment Verification

## Interpreter

- sys.executable: `<python-dir>\python.exe`
- sys.version: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- current working directory: `<repo>`

## sys.path First 10

1. `<repo>\scripts\research`
2. `<python-dir>\python313.zip`
3. `<python-dir>\DLLs`
4. `<python-dir>\Lib`
5. `<python-dir>`
6. `<python-dir>\Lib\site-packages`

## pip

### python -m pip --version

- returncode: `0`
```text
pip 26.0.1 from <python-dir>\Lib\site-packages\pip (python 3.13)

```

### python -m pip show vnstock_data

- returncode: `1`
```text

WARNING: Package(s) not found: vnstock_data
```

### python -m pip show vnstock

- returncode: `0`
```text
Name: vnstock
Version: 3.5.0
Summary: A beginner-friendly yet powerful Python toolkit for financial analysis and automation � built to make modern investing accessible to everyone
Home-page:
Author:
Author-email: Thinh Vu <support@vnstocks.com>
License: Custom: Personal, research, non-commercial; contact support@vnstocks.com for other use
Location: <python-dir>\Lib\site-packages
Requires: beautifulsoup4, importlib-metadata, openpyxl, packaging, pandas, psutil, pydantic, pytz, requests, seaborn, tenacity, vnai, vnstock_ezchart
Required-by: vnstock_ezchart

```

### pip list entries matching vnstock

```text
vnstock                                  3.5.0
vnstock_ezchart                          0.0.3
vnstock-installer                        3.1.1
```

## importlib Specs

- importlib.util.find_spec("vnstock_data"): `{'found': False, 'origin': '', 'submodule_search_locations': []}`
- importlib.util.find_spec("vnstock"): `{'found': True, 'origin': '<python-dir>\\Lib\\site-packages\\vnstock\\__init__.py', 'submodule_search_locations': ['<python-dir>\\Lib\\site-packages\\vnstock']}`

## Imports

- import vnstock_data success: `False`
- import vnstock_data error type: `ModuleNotFoundError`
- import vnstock_data error message: `No module named 'vnstock_data'`

### vnstock_data traceback

```text
Traceback (most recent call last):
  File "<repo>\scripts\research\verify_vnstock_data_environment.py", line 84, in import_attempt
    module = importlib.import_module(package_name)
  File "<python-dir>\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'vnstock_data'
```

- import vnstock success: `True`
- vnstock module file: `<python-dir>\Lib\site-packages\vnstock\__init__.py`
- vnstock version: `3.5.0`

## Local Shadow Checks

| candidate | exists | kind | path |
| --- | --- | --- | --- |
| vnstock_data.py | False |  |  |
| vnstock_data | False |  |  |
| vnstock.py | False |  |  |
| vnstock | False |  |  |
