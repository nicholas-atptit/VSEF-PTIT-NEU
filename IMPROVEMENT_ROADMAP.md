# Improvement Roadmap

The following prioritized roadmap outlines the next steps for enhancing the `AI-ML-LLM in Stock` system.

## P0: Critical Refinement
### Path Standardization in `settings.py`
Standardize all string-based paths into `Path` objects derived from `PROJECT_ROOT`. This eliminates potential runtime relative path issues.

### Centralized Command-Line Interface (CLI)
Create a single `main.py` entry point (using `click` or `argparse`) that wraps all scripts in `scripts/`. This would provide a unified command like `python main.py sync` or `python main.py train`.

## P1: Important Scalability
### Consistent Documentation Standards
Audit all modules in `src/` to ensure they follow a consistent docstring format (NumPy or Google style) and include type hinting.

### Automated Test Coverage
Expand `tests/` to include integration tests for the full pipeline: Ingestion -> Training -> Prediction.

## P2: Quality Upgrades
### Containerization
Develop a `Dockerfile` and `docker-compose.yml` to standardize the environment, including external dependencies like `PostgreSQL` or `ChromADB`.

### CI/CD Integration
Implement GitHub Actions to automate linting, testing, and documentation generation on every pull request.
