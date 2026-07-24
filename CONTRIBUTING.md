# Contributing to stratapro

## Getting Started
1. Fork and clone: `git clone https://github.com/<you>/stratapro.git`
2. Install: `pip install -e ".[dev]"`
3. Branch: `git checkout -b feature/your-feature`

## Development
```bash
python -m pytest tests/ -q
ruff check .
```

## Submitting
1. All tests pass
2. Ruff clean
3. Open a Pull Request
