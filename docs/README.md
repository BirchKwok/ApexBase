# ApexBase Documentation

This directory contains the Markdown source for the ApexBase documentation site.

The MkDocs site homepage is [index.md](index.md). For a compact reading order, see [documentation-index.md](documentation-index.md).

## Local Preview

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
```

## Production Build

```bash
python -m mkdocs build --strict
```

GitHub Pages deployment is configured in `.github/workflows/docs.yml`.

Versioned documentation is published with `mike` to the `gh-pages` branch. Configure GitHub Pages to deploy from the `gh-pages` branch root.
