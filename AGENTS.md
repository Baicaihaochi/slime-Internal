# Repository Guidelines

## Project Structure & Module Organization
Core runtime lives in `slime/` (training loops, utils) and opt-in modules ship in `slime_plugins/`. Experiment assets and runnable blueprints sit under `scripts/` (bash launchers) and `tools/` (conversion utilities). End-to-end docs and diagrams are in `docs/` and `imgs/`; share user-facing notebooks under `examples/`. Integration and unit coverage belongs in `tests/`. Generated artifacts and checkpoints must stay in `outputs/` or a user-specific path ignored by git.

## Build, Test, and Development Commands
Use `pip install -e .` after cloning to get an editable install; run in the project root. `bash build_conda.sh` provisions a GPU-ready Conda env when Docker is unavailable. Pull the maintained runtime with `docker pull zhuzilin/slime:latest`; rebuild locally via `docker build -f docker/Dockerfile .`. Run `pytest` or `pytest -m unit` for targeted suites. Before pushing, run `pre-commit run --all-files` to execute lint, format, and static checks.

## Coding Style & Naming Conventions
Target Python 3.10 syntax, 4-space indents, and 119-char lines (shared by Black, isort, Ruff). Prefer explicit module imports; rely on isort's Black profile. Name modules and packages with lowercase underscores (`slime/utils/data_buffer.py`) and classes in CapWords. Tests should mirror source names, e.g., `tests/test_data_buffer.py`. Register pre-commit hooks to apply Black, Ruff, and isort automatically.

## Testing Guidelines
Write pytest suites under `tests/` using `test_*.py` or `*_test.py` naming. Use the provided markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.) so CI can select runs. When adding rollout or training logic, include synthetic fixtures to avoid heavy checkpoints; stub GPU calls with mocks where feasible. Run `pytest --durations=0` before opening a PR to catch slow regressions. Add regression data to `outputs/` only when it is small and documented.

## Commit & Pull Request Guidelines
History favors short, imperative summaries (`wandb bug fix`); follow `<scope> <action>` at ~50 characters. Group related changes into logical commits and avoid mixing formatting with feature work. PRs should describe motivation, highlight breaking changes, and list validation commands (`pytest`, `scripts/run-glm4-9B.sh`). Link issues or tasks in the description and attach logs or screenshots for UI-facing components. Request at least one reviewer familiar with the touched subsystem and wait for CI to finish before merging.

## Environment & Configuration Tips
Keep Megatron and SGLang paths in sync with `scripts/models/*.sh` templates; source a model script before running `train.py`. Store credentials and API keys via environment variables rather than committing config files. Verify GPU availability with `nvidia-smi` inside the container before launching training. Large checkpoints should be referenced via object storage URLs instead of pushing to the repo.
