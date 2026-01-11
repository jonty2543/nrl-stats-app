# Repository Guidelines

## Project Structure & Module Organization
- `Analysis/` contains the Streamlit app (`main.py`), analysis scripts, and helper modules in `Analysis/functions/`.
- Supabase (schema `nrl`) is the source of match and player data; see `Analysis/json_to_csv.py` for queries.
- `ENVIRONMENT_VARIABLES.py` defines shared constants (teams, stat labels, color maps) used across analysis code.
- `requirements.txt` lists the core Python dependencies used by the app.

## Build, Test, and Development Commands
- `python -m venv .venv` and `source .venv/bin/activate`: create and activate a local virtualenv.
- `pip install -r requirements.txt`: install runtime dependencies for the Streamlit app.
- `streamlit run Analysis/main.py`: launch the interactive NRL stats UI locally.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; keep lines readable and prefer one import per line.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, and lower_snake_case for files.
- No formatter or linter is configured; match the existing style in `Analysis/`.

## Testing Guidelines
- No automated test framework is currently present in the repo.
- If you add new logic, consider adding lightweight tests and document how to run them in this file.

## Commit & Pull Request Guidelines
- Commit messages follow short, imperative summaries (e.g., “Update title”, “Move legend down”).
- PRs should include a concise description, the data or UI areas impacted, and screenshots when Streamlit visuals change.

## Data & Configuration Notes
- Data changes should go through Supabase; avoid adding local data dumps to the repo.
- When changing stat labels or team metadata, keep `ENVIRONMENT_VARIABLES.py` in sync with chart expectations.
