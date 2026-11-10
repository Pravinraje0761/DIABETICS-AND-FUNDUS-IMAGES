@echo off
REM ===========================================================================
REM  Round-4 GPU jobs: matched-recipe sensitivity analysis (reviewer item 7).
REM  You can double-click this NOW: it waits until run_round3_gpu.bat has
REM  finished (seeds 43 and 44), then runs automatically. Resumable.
REM   1. EfficientNet-B3 trained with the RETFound recipe            ~15 min
REM   2. RETFound full fine-tune trained with the CNN recipe          ~7-9 h
REM  Log: logs\round4_gpu.log    When it says ALL DONE, tell Claude.
REM  Keep the laptop plugged in and close other heavy programs.
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set LOG=logs\round4_gpu.log
set CKS=C:\dr_ckpts_full_seeds
set O=outputs\phase28
if exist C:\dr_run\data_processed set DR_DATA_ROOT=C:\dr_run
if not exist %O% mkdir %O%
if not exist %CKS% mkdir %CKS%
set PYTHONUNBUFFERED=1

:waitloop
findstr /C:"ALL DONE" logs\round3_gpu.log >nul 2>&1
if errorlevel 1 (
  echo Waiting for run_round3_gpu.bat to finish ... %TIME%
  timeout /t 300 /nobreak >nul
  goto waitloop
)
echo START %DATE% %TIME% >> %LOG%

REM ---------------- 1. EfficientNet-B3 with the RETFound recipe
if not exist outputs\phase4_baseline\efficientnet_b3_rfrecipe_best.pt (
  echo === 1 EffB3 RETFound-recipe >> %LOG%
  %PY% scripts\train_baseline.py --arch efficientnet_b3 --label-smoothing 0.1 --weight-decay 0.05 --batch-size 16 --warmup-epochs 3 --tag rfrecipe --num-workers 2 >> %LOG% 2>&1
)
call :ext baseline_efficientnet_b3_rfrecipe outputs\phase4_baseline\efficientnet_b3_rfrecipe_best.pt

REM ---------------- 2. RETFound full fine-tune with the CNN recipe
if not exist %CKS%\CNNRECIPE_DONE.txt (
  set ATTEMPT=0
  call :trainrf
)
call :ext retfound_full_cnnrecipe %CKS%\full_cnnrecipe_best.pt
echo ALL DONE %DATE% %TIME% >> %LOG%
echo.
echo ALL DONE. Tell Claude "round 4 finished".
goto end

:trainrf
set /a ATTEMPT+=1
echo === 2 RETFound CNN-recipe attempt !ATTEMPT! %DATE% %TIME% >> %LOG%
%PY% scripts\train_retfound.py --mode full --weights weights\RETFound_mae_natureCFP.pth --batch-size 16 --layer-decay 1.0 --drop-path 0.0 --label-smoothing 0.0 --weight-decay 0.0001 --warmup-epochs 0 --tag cnnrecipe --ckpt-dir %CKS% --save-every 1 --num-workers 2 --resume >> %LOG% 2>&1
if !ERRORLEVEL! EQU 0 (
  echo done %DATE% %TIME% > %CKS%\CNNRECIPE_DONE.txt
  exit /b 0
)
if !ATTEMPT! LSS 8 (
  timeout /t 60 /nobreak >nul
  goto trainrf
)
echo GAVE UP - double-click again to resume >> %LOG%
exit /b 1

:ext
if not exist %2 exit /b 0
if not exist %O%\%1__Messidor-2_preds.csv %PY% scripts\predict.py --checkpoint %2 --manifest data_splits\messidor2_manifest.csv --split external --out-prefix %O%\%1__Messidor-2 >> %LOG% 2>&1
if not exist %O%\%1__IDRiD-official_preds.csv %PY% scripts\predict.py --checkpoint %2 --manifest data_splits\idrid_official_manifest.csv --split external --out-prefix %O%\%1__IDRiD-official >> %LOG% 2>&1
if not exist %O%\%1__APTOS-internal_preds.csv %PY% scripts\predict.py --checkpoint %2 --manifest data_splits\aptos_manifest.csv --split test --out-prefix %O%\%1__APTOS-internal >> %LOG% 2>&1
exit /b 0

:end
pause
