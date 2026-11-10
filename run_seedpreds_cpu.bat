@echo off
REM ===========================================================================
REM  CPU-only predictions on the official IDRiD release for the extra training
REM  seeds of EfficientNet-B3 (43-45) and the RETFound linear probe (43-45).
REM  Safe to run NOW, alongside the GPU training window (uses the CPU only).
REM  About 20-40 minutes. Log: logs\seedpreds_cpu.log
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set LOG=logs\seedpreds_cpu.log
set O=outputs\phase28
if exist C:\dr_run\data_processed set DR_DATA_ROOT=C:\dr_run
echo START %DATE% %TIME% >> %LOG%
for %%S in (43 44 45) do (
  if not exist %O%\baseline_efficientnet_b3_seed%%S__IDRiD-official_preds.csv (
    echo === EffB3 seed %%S >> %LOG%
    %PY% scripts\predict.py --checkpoint outputs\phase4_baseline\efficientnet_b3_seed%%S_best.pt --manifest data_splits\idrid_official_manifest.csv --split external --out-prefix %O%\baseline_efficientnet_b3_seed%%S__IDRiD-official --device cpu --num-workers 2 >> %LOG% 2>&1
  )
)
for %%S in (43 44 45) do (
  if not exist %O%\retfound_linprobe_seed%%S__IDRiD-official_preds.csv (
    echo === RETFound linear probe seed %%S >> %LOG%
    %PY% scripts\predict.py --checkpoint C:\dr_ckpts_seeds\linprobe_seed%%S_best.pt --manifest data_splits\idrid_official_manifest.csv --split external --out-prefix %O%\retfound_linprobe_seed%%S__IDRiD-official --device cpu --num-workers 2 --batch-size 16 >> %LOG% 2>&1
  )
)
echo ALL DONE %DATE% %TIME% >> %LOG%
echo.
echo ALL DONE. You can close this window.
pause
