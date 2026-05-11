@echo off
REM ============================================================================
REM  wh40k-rag launcher
REM  "Knowledge is power. Guard it well."
REM ----------------------------------------------------------------------------
REM  Click this file (or a shortcut to it) to start the Librarius.
REM  It will:
REM    1. Ensure Ollama is running (start it if not)
REM    2. Activate the local Python virtual environment
REM    3. Start the FastAPI app on http://127.0.0.1:8000
REM    4. Open your default browser to the app
REM ============================================================================

setlocal enabledelayedexpansion

REM --- Always run from the directory this script lives in ---------------------
cd /d "%~dp0"

echo.
echo  ============================================================
echo    wh40k-rag  --  Knowledge is power. Guard it well.
echo  ============================================================
echo.

REM --- 1. Ollama check ---------------------------------------------------------
echo  [1/4] Checking Ollama...
curl -s -o NUL -w "%%{http_code}" http://127.0.0.1:11434/api/tags > "%TEMP%\ollama_check.txt" 2>NUL
set /p OLLAMA_STATUS=<"%TEMP%\ollama_check.txt"
del "%TEMP%\ollama_check.txt" >NUL 2>&1

if "!OLLAMA_STATUS!"=="200" (
    echo        Ollama is already running.
) else (
    echo        Ollama not detected. Starting it...
    start "Ollama" /min cmd /c "ollama serve"

    REM Poll for Ollama to come up (15s timeout)
    set OLLAMA_READY=0
    for /L %%i in (1,1,15) do (
        timeout /t 1 /nobreak >NUL
        curl -s -o NUL -w "%%{http_code}" http://127.0.0.1:11434/api/tags > "%TEMP%\ollama_check.txt" 2>NUL
        set /p CHECK=<"%TEMP%\ollama_check.txt"
        del "%TEMP%\ollama_check.txt" >NUL 2>&1
        if "!CHECK!"=="200" (
            set OLLAMA_READY=1
            goto :ollama_up
        )
    )
    :ollama_up
    if "!OLLAMA_READY!"=="1" (
        echo        Ollama is up.
    ) else (
        echo  [ERROR] Ollama failed to start within 15 seconds.
        echo          Try running "ollama serve" manually to see what's wrong.
        pause
        exit /b 1
    )
)

REM --- 2. Verify embedding model is pulled ------------------------------------
echo  [2/4] Checking nomic-embed-text model...
ollama list | findstr /C:"nomic-embed-text" >NUL
if errorlevel 1 (
    echo        Model not found. Pulling nomic-embed-text (this is a one-time download)...
    ollama pull nomic-embed-text
    if errorlevel 1 (
        echo  [ERROR] Failed to pull nomic-embed-text.
        pause
        exit /b 1
    )
) else (
    echo        Model is present.
)

REM --- 3. Activate virtual environment ----------------------------------------
echo  [3/4] Activating virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo  [ERROR] No .venv found in %CD%.
    echo          Create one with:
    echo              python -m venv .venv
    echo              .venv\Scripts\activate
    echo              pip install -r requirements.txt
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

REM --- 4. Launch uvicorn + open browser ---------------------------------------
echo  [4/4] Starting the Librarius on http://127.0.0.1:8000 ...
echo.
echo        ---  Close this window to stop the server  ---
echo.

REM Open browser after a short delay so uvicorn has time to bind the port
start "" /b cmd /c "timeout /t 3 /nobreak >NUL & start http://127.0.0.1:8000"

REM Bind to 127.0.0.1 explicitly so we don't accidentally expose to the LAN
uvicorn main:app --host 127.0.0.1 --port 8000

endlocal
