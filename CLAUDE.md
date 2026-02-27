# manim-metal

## Development Environment

- Use **uv** for all tooling. Prefer `uv` subcommands over their standalone equivalents:
  - `uv pip` instead of `pip`
  - `uv run` instead of direct `python`/`pytest`/etc.
  - `uv venv` instead of `python -m venv`
  - `uv add` / `uv remove` for dependency management
- Use **ruff** for linting and formatting throughout the project.
