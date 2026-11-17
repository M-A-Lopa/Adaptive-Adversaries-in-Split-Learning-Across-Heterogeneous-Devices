@echo off
REM ------------------------------------------------------------------
REM  Experiment 2 runner for Windows: ONE command runs everything
REM    models  : KAGN, PyramidCNN
REM    datasets: CIFAR10, MNIST
REM    cuts    : 1, 2, 3, 4
REM    poison  : 0% (clean, false-alarm check), 1%, 5%, 10%
REM    rules   : none (no defence), smaller, crossclass, oracle (upper bound)
REM              -> 2 x 2 x 4 x 4 x 4 = 256 runs
REM
REM  Run from the project root (PowerShell):  .\experiments\run_exp2.bat
REM  Extra options are passed on, e.g.        .\experiments\run_exp2.bat --epochs 20
REM
REM  Resume: run the SAME command again after any stop. Finished runs are
REM  skipped and an interrupted run continues from its last finished epoch.
REM  If Python crashes it is restarted automatically (up to 5 times per dataset).
REM  Ctrl+C stops everything; answer Y to "Terminate batch job".
REM ------------------------------------------------------------------
setlocal
set "EXTRA=%*"
set MAX_TRIES=5

for %%D in (CIFAR10 MNIST) do (
    call :run_dataset %%D
    if errorlevel 1 goto end
)

echo.
echo ================================================================
echo  All 256 runs finished. Reports:
echo    results\exp2_defence\exp2_results_CIFAR10_report.pdf
echo    results\exp2_defence\exp2_results_MNIST_report.pdf
echo ================================================================
goto end


:run_dataset
set TRY=1
:retry
echo.
echo ===== Experiment 2 ^| dataset %1 ^| attempt %TRY% of %MAX_TRIES% =====
python experiments\exp2_detector_defence.py --dataset %1 --models KAGN PyramidCNN --cuts 1 2 3 4 --rates 0 0.01 0.05 0.10 --rules none smaller crossclass oracle %EXTRA%
set CODE=%ERRORLEVEL%
if "%CODE%"=="0" exit /b 0
if "%CODE%"=="130" (
    echo Stopped by you. Run the same command again to continue.
    exit /b 130
)
if %TRY% GEQ %MAX_TRIES% (
    echo Python failed %MAX_TRIES% times in a row on %1 ^(exit code %CODE%^). Check the error above.
    exit /b 1
)
set /a TRY=%TRY%+1
echo Python exited with code %CODE%. Restarting in 10 seconds from the last checkpoint...
timeout /t 10 /nobreak >nul
goto retry


:end
endlocal