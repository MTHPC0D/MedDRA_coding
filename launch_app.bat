@echo off
setlocal enabledelayedexpansion

REM Go to this script's directory
pushd "%~dp0"

REM Try to activate the 'xtars' conda environment
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat" xtars
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\miniconda3\Scripts\activate.bat" xtars
) else (
  where conda >nul 2>&1
  if %errorlevel%==0 (
    call conda activate xtars
  ) else (
    echo [Erreur] Impossible de trouver Anaconda/Miniconda. Assurez-vous que 'conda' est installe et dans le PATH.
    pause
    exit /b 1
  )
)

REM Run the Streamlit app
python -m streamlit run app.py

popd
endlocal
