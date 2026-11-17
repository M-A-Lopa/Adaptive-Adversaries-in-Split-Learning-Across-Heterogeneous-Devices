@echo off
REM ------------------------------------------------------------------
REM  Experiment 3 runner for Windows: ONE command runs everything
REM    models  : KAGN, PyramidCNN
REM    datasets: CIFAR10, MNIST
REM    cuts    : 1, 2, 3, 4
REM    poison  : 0, 1, 5, 10 percent of all training data
REM    methods : none, detect, oracle
REM              2 x 2 x 4 x 4 x 3 = 192 runs
REM
REM  Run from the project root in PowerShell:   .\experiments\run_exp3.bat
REM  Extra options are passed on, for example:  .\experiments\run_exp3.bat --z-thr 10
REM
REM  Resume: run the SAME command again after any stop. Finished runs are
REM  skipped and an interrupted run continues from its last finished round.
REM  If Python crashes it is started again automatically, up to 5 times per dataset.
REM  Ctrl+C stops everything. Answer Y to "Terminate batch job".
REM ------------------------------------------------------------------
setlocal enabledelayedexpansion

for %%D in (CIFAR10 MNIST) do (
    set DONE=0
    for /L %%T in (1,1,5) do (
        if !DONE!==0 (
            echo.
            echo ===== Experiment 3 - dataset %%D - attempt %%T of 5 =====
            python experiments\exp3_client_defence.py --dataset %%D --models KAGN PyramidCNN --cuts 1 2 3 4 --rates 0 0.01 0.05 0.10 --methods none detect oracle %*
            set CODE=!ERRORLEVEL!
            if !CODE!==0 (
                set DONE=1
            ) else (
                if !CODE!==130 (
                    echo Stopped by you. Run the same command again to continue.
                    exit /b 130
                )
                echo Python exited with code !CODE!. Restarting in 10 seconds from the last checkpoint...
                timeout /t 10 /nobreak >nul
            )
        )
    )
    if !DONE!==0 (
        echo Python failed 5 times in a row on %%D. Check the error above.
        exit /b 1
    )
)

echo.
echo ================================================================
echo  All 192 runs finished. Reports:
echo    results\exp3_client\exp3_results_CIFAR10_report.pdf
echo    results\exp3_client\exp3_results_MNIST_report.pdf
echo ================================================================
endlocal