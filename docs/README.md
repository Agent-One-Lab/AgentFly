## Documentation

Use the following command to build the documentation with MkDocs:

```bash
# Install dependencies
pip install -r docs/requirements-docs.txt

# Build and serve documentation locally
mkdocs serve

# Or build static site
mkdocs build
```

The documentation will be available at `http://127.0.0.1:8000` when using `mkdocs serve`.

To deploy the documentation, you can use:

```bash
# Build static site
mkdocs build

# The site will be in docs/site/ directory
```
