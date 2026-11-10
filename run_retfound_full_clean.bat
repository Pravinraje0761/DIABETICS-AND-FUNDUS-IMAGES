@echo off
REM ===========================================================================
REM  Clean, completed RETFound full fine-tune + full downstream refresh.
REM  Double-click to run. Log: logs\phase5_retfound_full_v4.log
REM  Training is resumable: if Python crashes / runs out of memory / the PC
REM  reboots, just double-click this file again -- it continues from the last
REM  completed epoch (C:\dr_ckpts_full_v4\full_resume.pt), same protocol
REM  (30 epochs max, early stopping on val QWK, patience 7).
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set CK=C:\dr_ckpts_full_v4
set LOG=logs\phase5_retfound_full_v4.log
set RLOG=logs\refresh_after_full_v4.log
if exist C:\dr_run\data_processed set DR_DATA_ROOT=C:\dr_run
if not exist %CK% mkdir %CK%
set PYTHONUNBUFFERED=1

REM -- 0. archive the provisional (epoch-6) outputs ONCE, before anything is overwritten
if not exist outputs\_archive_provisional_epoch6\DONE.txt (
  echo archiving provisional outputs ...
  for %%D in (phase5_retfound phase6 phase7 phase8 phase9 phase11 phase13 phase15 phase15c phase19 phase20 figures) do (
    robocopy outputs\%%D outputs\_archive_provisional_epoch6\%%D /E /XF *.pt /NFL /NDL /NJH /NJS >nul
  )
  copy /Y paper\main_standalone.tex outputs\_archive_provisional_epoch6\main_standalone.provisional.tex >nul
  echo archived %DATE% %TIME% > outputs\_archive_provisional_epoch6\DONE.txt
)

REM -- 1. train (auto-resume up to 8 attempts)
if exist %CK%\TRAINING_DONE.txt goto refresh
echo START %DATE% %TIME% >> %LOG%
%PY% -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" >> %LOG% 2>&1
set ATTEMPT=0
:trainloop
set /a ATTEMPT+=1
echo === training attempt !ATTEMPT! %DATE% %TIME% >> %LOG%
%PY% scripts\train_retfound.py --mode full --weights weights\RETFound_mae_natureCFP.pth --batch-size 16 --ckpt-dir %CK% --save-every 1 --num-workers 2 --resume >> %LOG% 2>&1
if !ERRORLEVEL! EQU 0 (
  echo training finished %DATE% %TIME% > %CK%\TRAINING_DONE.txt
  goto refresh
)
echo attempt !ATTEMPT! exited with code !ERRORLEVEL! >> %LOG%
if !ATTEMPT! LSS 8 (
  timeout /t 60 /nobreak >nul
  goto trainloop
)
echo GAVE UP after 8 attempts -- double-click again to resume. >> %LOG%
goto end

:refresh
REM -- 2. swap in the new checkpoint (old one kept, renamed)
if not exist outputs\phase5_retfound\full_best_provisional_epoch6.pt (
  move /Y outputs\phase5_retfound\full_best.pt outputs\phase5_retfound\full_best_provisional_epoch6.pt
)
copy /Y %CK%\full_best.pt outputs\phase5_retfound\full_best.pt
echo REFRESH START %DATE% %TIME% > %RLOG%
call :run scripts\predict.py --checkpoint outputs\phase5_retfound\full_best.pt --manifest data_splits\aptos_manifest.csv --split test --out-prefix outputs\phase5_retfound\full_internal_test
call :run scripts\phase6_external_eval.py --models retfound_full
call :run scripts\phase9_stats.py --boot 2000 --seed 42
call :run scripts\phase11_review_response.py --boot 2000 --seed 42
call :run scripts\phase23_primary_inference.py --boot 2000 --seed 42
call :run scripts\predict.py --checkpoint outputs\phase5_retfound\full_best.pt --manifest data_splits\aptos_manifest.csv --split val --out-prefix outputs\phase13\calib\retfound_full__APTOS-val
call :run scripts\predict.py --checkpoint outputs\phase5_retfound\full_best.pt --manifest data_splits\aptos_calib_train.csv --split calib --out-prefix outputs\phase13\calib_train\retfound_full__APTOS-traincalib
call :run scripts\phase13_temperature_scaling.py --calib-preds outputs/phase13/calib --calib-tag APTOS-val --bootstrap-T 2000
call :run scripts\phase13_temperature_scaling.py --calib-preds outputs/phase13/calib_train --calib-tag APTOS-traincalib --out-dir outputs/phase13/traincalib
call :run scripts\phase19_referable_dr.py --boot 2000 --seed 42
call :run scripts\phase20_ood_detection.py --device cuda --boot 2000 --seed 42
call :run scripts\phase8_uncertainty.py --preds-dir outputs\phase6 --tag singlepass
call :run scripts\phase8_mc_dropout.py --passes 20 --device cuda --models retfound_full
call :run scripts\phase8_uncertainty.py --preds-dir outputs\phase8\mc --tag mc
call :run scripts\phase9_stats.py --boot 2000 --seed 42 --preds-dir outputs/phase8/mc --tag mc
call :run scripts\phase23_primary_inference.py --boot 2000 --seed 42 --preds-dir outputs/phase8/mc --tag mc
call :run scripts\phase7_explainability.py --device cuda --n-correct 40 --n-incorrect 25
call :run scripts\phase7_rating_packet.py --device cuda
call :run scripts\phase15_ablation.py --device cuda
call :run scripts\phase15c_resolution.py --device cuda
call :run scripts\phase10_figures.py
call :run scripts\phase10b_figures.py
call :run scripts\phase11_figures.py
copy /Y outputs\figures\*.pdf paper\figures\ >nul
copy /Y outputs\figures\*.png paper\figures\ >nul
copy /Y outputs\phase7\fig_explainability.png paper\figures\ >nul
copy /Y outputs\phase8\fig_uncertainty_singlepass.png paper\figures\ >nul
echo REFRESH DONE %DATE% %TIME% >> %RLOG%
echo.
echo All done. Tell Claude "the run finished".
goto end

:run
echo === %* >> %RLOG%
%PY% %* >> %RLOG% 2>&1
if errorlevel 1 echo !!! FAILED: %* >> %RLOG%
exit /b 0

:end
pause
