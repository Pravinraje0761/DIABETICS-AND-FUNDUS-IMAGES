@echo off
REM ===========================================================================
REM  Round-3 GPU jobs (International Ophthalmology submission). Double-click.
REM  Resumable: re-run after any crash / reboot; finished steps are skipped.
REM   A. Official IDRiD (516 images): predictions from the 4 frozen models   ~15 min
REM   B. EfficientNet-B3 + label smoothing 0.1 (sensitivity analysis)         ~1 h
REM   C. RETFound full fine-tune, seeds 43 and 44                             ~8 h
REM  Log: logs\round3_gpu.log     When it says ALL DONE, tell Claude.
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set LOG=logs\round3_gpu.log
set CKS=C:\dr_ckpts_full_seeds
set O=outputs\phase28
if exist C:\dr_run\data_processed set DR_DATA_ROOT=C:\dr_run
if not exist %O% mkdir %O%
if not exist %CKS% mkdir %CKS%
set PYTHONUNBUFFERED=1
echo START %DATE% %TIME% >> %LOG%

if not exist data_splits\idrid_official_manifest.csv (
  echo Official IDRiD manifest missing - tell Claude the B. Disease Grading folder is downloaded first.
  echo NO MANIFEST >> %LOG%
  goto end
)

REM ---------------- A. official IDRiD predictions, primary checkpoints
for %%M in (baseline_efficientnet_b3:outputs\phase4_baseline\efficientnet_b3_best.pt baseline_resnet50:outputs\phase4_baseline\resnet50_best.pt retfound_linprobe:outputs\phase5_retfound\linprobe_best.pt retfound_full:outputs\phase5_retfound\full_best.pt) do (
  for /f "tokens=1,* delims=:" %%a in ("%%M") do (
    if not exist %O%\%%a__IDRiD-official_preds.csv (
      echo === A %%a >> %LOG%
      %PY% scripts\predict.py --checkpoint %%b --manifest data_splits\idrid_official_manifest.csv --split external --out-prefix %O%\%%a__IDRiD-official >> %LOG% 2>&1
    )
  )
)

REM ---------------- B. EfficientNet-B3 with label smoothing 0.1
if not exist outputs\phase4_baseline\efficientnet_b3_ls01_best.pt (
  echo === B train >> %LOG%
  %PY% scripts\train_baseline.py --arch efficientnet_b3 --label-smoothing 0.1 --tag ls01 --num-workers 2 >> %LOG% 2>&1
)
call :ext baseline_efficientnet_b3_ls01 outputs\phase4_baseline\efficientnet_b3_ls01_best.pt

REM ---------------- C. RETFound full fine-tune, extra seeds
for %%S in (43 44) do (
  if not exist %CKS%\SEED%%S_DONE.txt (
    set ATTEMPT=0
    call :trainseed %%S
  )
  call :ext retfound_full_seed%%S %CKS%\full_seed%%S_best.pt
)
echo ALL DONE %DATE% %TIME% >> %LOG%
echo.
echo ALL DONE. Tell Claude "round 3 finished".
goto end

:trainseed
set /a ATTEMPT+=1
echo === C seed %1 attempt !ATTEMPT! %DATE% %TIME% >> %LOG%
%PY% scripts\train_retfound.py --mode full --weights weights\RETFound_mae_natureCFP.pth --batch-size 16 --seed %1 --tag seed%1 --ckpt-dir %CKS% --save-every 1 --num-workers 2 --resume >> %LOG% 2>&1
if !ERRORLEVEL! EQU 0 (
  echo done %DATE% %TIME% > %CKS%\SEED%1_DONE.txt
  exit /b 0
)
if !ATTEMPT! LSS 8 (
  timeout /t 60 /nobreak >nul
  goto trainseed
)
echo GAVE UP seed %1 - double-click again to resume >> %LOG%
exit /b 1

:ext
REM %1 = model name, %2 = checkpoint; external + internal predictions
if not exist %2 exit /b 0
if not exist %O%\%1__Messidor-2_preds.csv %PY% scripts\predict.py --checkpoint %2 --manifest data_splits\messidor2_manifest.csv --split external --out-prefix %O%\%1__Messidor-2 >> %LOG% 2>&1
if not exist %O%\%1__IDRiD_preds.csv %PY% scripts\predict.py --checkpoint %2 --manifest data_splits\idrid_manifest.csv --split external --out-prefix %O%\%1__IDRiD >> %LOG% 2>&1
if not exist %O%\%1__IDRiD-official_preds.csv %PY% scripts\predict.py --checkpoint %2 --manifest data_splits\idrid_official_manifest.csv --split external --out-prefix %O%\%1__IDRiD-official >> %LOG% 2>&1
if not exist %O%\%1__APTOS-internal_preds.csv %PY% scripts\predict.py --checkpoint %2 --manifest data_splits\aptos_manifest.csv --split test --out-prefix %O%\%1__APTOS-internal >> %LOG% 2>&1
exit /b 0

:end
pause
