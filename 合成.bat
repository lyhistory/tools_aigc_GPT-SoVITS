@echo off
rem ============================================================
rem  GPT-SoVITS ???? / one-click TTS
rem  Put this .bat next to gsv_tts_local.py and gsv_tts.ini,
rem  in the GPT-SoVITS root folder (the one with webui.py).
rem
rem  Usage:
rem    double-click            -> read text from Windows clipboard
rem    ??.bat ??.txt        -> synthesize every line of that file
rem    ??.bat "???"        -> synthesize that sentence
rem    (drag a .txt onto me)   -> same as ??.bat ??.txt
rem
rem  This file is deliberately ASCII-only to avoid codepage issues.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title GPT-SoVITS TTS

set "PY=%~dp0runtime\python.exe"
set "S=%~dp0gsv_tts_local.py"

if not exist "%PY%" goto nopy
if not exist "%S%"  goto noscript

if "%~1"=="" (
    "%PY%" "%S%" --root . --clip
    goto done
)
if exist "%~1" (
    "%PY%" "%S%" --root . --text-file "%~1"
    goto done
)
"%PY%" "%S%" --root . --text "%~1"
goto done

:nopy
echo [x] runtime\python.exe not found.
echo     Move this .bat into the GPT-SoVITS root folder.
goto done

:noscript
echo [x] gsv_tts_local.py not found.
echo     Copy it next to this .bat.
goto done

:done
echo.
echo ---- finished ----
pause
