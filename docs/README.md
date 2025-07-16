## Documentation

Use the following command to build the documentation

```bash
pip install -r docs/requirements-docs.txt
sphinx-build docs docs/_build/html
python -m http.server -d docs/_build/html
```