# vnstock_data Interpreter Fix Plan

## Current Finding

The current Codex/repo script runs are using:

```powershell
C:\Python\python.exe
```

The environment verification showed that `vnstock_data` is not installed in that interpreter:

```text
ModuleNotFoundError: No module named 'vnstock_data'
```

The same interpreter does have legacy `vnstock` installed:

```text
vnstock 3.5.0
```

This explains why previous provider probes used legacy `vnstock` instead of `vnstock_data`. The next step is to rerun provider verification using the intended repo virtual environment, then only run fetch/probe scripts with that venv after `vnstock_data` imports successfully.

## Required Checks

1. Check the intended venv interpreter:

```powershell
C:\Users\luong\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

2. Check whether `vnstock_data` is installed in that venv:

```powershell
C:\Users\luong\.venv\Scripts\python.exe -m pip show vnstock_data
```

3. Run the environment verification script with that venv:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts\research\verify_vnstock_data_environment.py
```

4. Run fetch/probe scripts with that venv only after `vnstock_data` imports successfully:

```powershell
C:\Users\luong\.venv\Scripts\python.exe <fetch-or-probe-script>
```

Do not fetch data, run benchmarks, train models, write paper outputs, or generate DOCX until the intended venv verifies `vnstock_data` importability.
