@echo off
setlocal
set SOMNIGRAPH_DATA_DIR=%USERPROFILE%\.claude\data
set SOMNIGRAPH_EMBEDDING_BACKEND=fastembed
uv run --directory "%~dp0.." --extra fastembed python scripts/build_ground_truth.py --resume %*
