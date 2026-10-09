@echo off
REM Lance l'application complete (backend FastAPI + frontend build) sur http://localhost:8001
cd /d "%~dp0"
.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8001
