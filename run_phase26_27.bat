@echo off
REM Reviewer round 2, point 3 - lesion-mask localisation (GPU) + rater agreement.
REM 1) Download "A. Segmentation" (IDRiD, IEEE DataPort, free account, CC BY 4.0) and unzip, e.g. to C:\dr_data\IDRiD_seg
REM 2) Edit SEG below if you unzip elsewhere, then double-click this file.
set SEG=%~dp0A. Segmentation
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\phase26_lesion_localization.py --seg-root "%SEG%" --device cuda > outputs\phase26_run_log.txt 2>&1
type outputs\phase26_run_log.txt
REM 3) When two raters have returned their sheets (outputs\phase7\rating_packet\rater_XX.csv):
REM .venv\Scripts\python.exe scripts\phase27_rater_agreement.py --sheets outputs\phase7\rating_packet\rater_AA.csv outputs\phase7\rating_packet\rater_BB.csv
pause
