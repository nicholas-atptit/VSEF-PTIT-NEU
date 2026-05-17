# vnstock_data Intended Venv Verification

- Intended venv executable: `<repo-approved-venv-python>`
- Intended venv exists: yes
- `sys.executable`: `<repo-approved-venv-python>`
- `sys.version`: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- pip version: `pip 26.0.1 from <repo-approved-venv>/Lib/site-packages/pip (python 3.13)`
- vnstock_data installed: yes
- vnstock_data spec found: yes
- vnstock_data import success: yes
- vnstock_data version: `3.0.0`
- vnstock installed: yes
- vnstock import success: yes
- vnstock version: `3.5.1` from package metadata; module `__version__` is absent

## pip show vnstock_data

`python -m pip show vnstock_data` failed inside pip metadata parsing:

```text
ValueError: invalid literal for int() with base 10: ''
```

The package is still importable from:

```text
<repo-approved-venv>/Lib/site-packages/vnstock_data/__init__.py
```

## pip show vnstock

```text
Name: vnstock
Version: 3.5.1
Location: <repo-approved-venv>/Lib/site-packages
```

## Import Traceback

No vnstock_data import traceback was produced because `import vnstock_data` succeeded.
