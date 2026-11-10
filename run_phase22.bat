@echo off
REM Phase 22 head-depth x activation ablation. Log: outputs\phase22\run_log.txt
cd /d "%~dp0"
if not exist outputs\phase22 mkdir outputs\phase22
set LOG=outputs\phase22\run_log.txt
set PY=.venv\Scripts\python.exe
echo START %DATE% %TIME% > %LOG%
%PY% -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" >> %LOG% 2>&1
%PY% -m pip install -q tabulate >> %LOG% 2>&1
for %%C in (outputs\phase4_baseline\efficientnet_b3_best.pt outputs\phase5_retfound\linprobe_best.pt) do (
  echo === EMBED %%C aptos >> %LOG%
  %PY% scripts\phase22_head_depth_ablation.py --stage embed --checkpoint %%C --manifest data_splits\aptos_manifest.csv --device cuda >> %LOG% 2>&1
  echo === EMBED %%C messidor2 >> %LOG%
  %PY% scripts\phase22_head_depth_ablation.py --stage embed --checkpoint %%C --manifest data_splits\messidor2_manifest.csv --external --device cuda >> %LOG% 2>&1
  echo === EMBED %%C idrid >> %LOG%
  %PY% scripts\phase22_head_depth_ablation.py --stage embed --checkpoint %%C --manifest data_splits\idrid_manifest.csv --external --device cuda >> %LOG% 2>&1
)
echo === SWEEP >> %LOG%
%PY% scripts\phase22_head_depth_ablation.py --stage sweep --outdir outputs\phase22 >> %LOG% 2>&1
echo DONE %DATE% %TIME% >> %LOG%
