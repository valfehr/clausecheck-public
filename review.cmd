@echo off
rem Shortcut for the contract review tool. From PowerShell in this folder:  .\review --list
rem Uses the project's own Python environment directly, so it works whatever "py"/"python"/conda point to.
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Project environment missing. Run once:  py -3.14 -m uv sync
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\review.py" %*
