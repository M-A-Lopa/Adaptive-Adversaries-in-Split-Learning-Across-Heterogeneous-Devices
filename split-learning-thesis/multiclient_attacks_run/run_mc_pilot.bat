@echo off
REM ------------------------------------------------------------------
REM  Multi-client PILOT (undefended): 3 attacks x 4 configurations, IID
REM    attacks : AE decoder, label leakage, client backdoor
REM    configs : CIFAR10 KAGN cut 1, 3 and MNIST PyramidCNN cut 1, 3
REM  Clients, split, victim, malicious client, seed: config.py (NUM_CLIENTS ...)
REM
REM  Run from the project root in PowerShell:   .\multiclient_attacks_run\run_mc_pilot.bat
REM  Resume: run the SAME command again after any stop.
REM  At the end: comparison with the single-client results
REM    results\mc\K5_iid\undefended_suite\compare_single_vs_multi.pdf
REM ------------------------------------------------------------------
setlocal enabledelayedexpansion
set ATT=ae_decoder label_leakage backdoor_client

call :run --datasets CIFAR10 --models KAGN --cuts 1 3 --attacks %ATT% --split iid %*
if errorlevel 1 exit /b 1
call :run --datasets MNIST --models PyramidCNN --cuts 1 3 --attacks %ATT% --split iid %*
if errorlevel 1 exit /b 1

python multiclient_attacks_run\mc_report.py --setting K5_iid
echo.
echo  Pilot finished. Send: results\mc\K5_iid\undefended_suite\compare_single_vs_multi.pdf
echo                  and   results\mc\K5_iid\undefended_suite\undefended_attack_suite.pdf
endlocal
exit /b 0

:run
set DONE=0
for /L %%T in (1,1,5) do (
    if !DONE!==0 (
        echo.
        echo ===== multi-client pilot %* - attempt %%T of 5 =====
        python multiclient_attacks_run\mc_suite.py %*
        set CODE=!ERRORLEVEL!
        if !CODE!==0 (
            set DONE=1
        ) else (
            if !CODE!==130 (
                echo Stopped by you. Run the same command again to continue.
                exit /b 1
            )
            echo Python exited with code !CODE!. Restarting in 10 seconds from the last checkpoint...
            timeout /t 10 /nobreak >nul
        )
    )
)
if !DONE!==0 (
    echo Failed 5 times in a row. Check the error above.
    exit /b 1
)
exit /b 0