@echo off
REM ------------------------------------------------------------------
REM  Multi-client undefended attack suite: all attacks of mc_suite.py,
REM  2 datasets x 2 models x 4 cuts.
REM  Clients, split, victim, malicious client, seed: config.py (NUM_CLIENTS ...)
REM
REM  Run from the project root in PowerShell:
REM     .\multiclient_attacks_run\run_mc_attacks.bat                    (split from config.py)
REM     .\multiclient_attacks_run\run_mc_attacks.bat --split iid
REM     .\multiclient_attacks_run\run_mc_attacks.bat --split noniid
REM  Extra options are passed on, e.g. --clients 10 or --attacks ae_decoder fsha
REM  Resume: run the SAME command again after any stop.
REM ------------------------------------------------------------------
setlocal enabledelayedexpansion
for %%D in (CIFAR10 MNIST) do (
    set DONE=0
    for /L %%T in (1,1,5) do (
        if !DONE!==0 (
            echo.
            echo ===== multi-client suite - %%D - attempt %%T of 5 =====
            python multiclient_attacks_run\mc_suite.py --datasets %%D %*
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
echo  Finished. Reports are in results\mc\^<setting^>\undefended_suite\
echo  Comparison with single-client:  python multiclient_attacks_run\mc_report.py --setting K5_iid
endlocal