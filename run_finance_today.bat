@echo off
cd /d C:\Project_Chimera
echo [1/3] Pipeline basliyor...
python main.py --channel CKFinanceCore
echo [2/3] Altyazi uretiliyor...
python -c "from auto_captions import generate_multi_captions; print(generate_multi_captions(r'outputs\CKFinanceCore\2025-08-12\final_video.mp4'))"
echo [3/3] Cikti listesi:
powershell -NoLogo -NoProfile -Command "Get-ChildItem -Recurse -File 'outputs\CKFinanceCore\2025-08-12' | Select FullName,Length,LastWriteTime"
echo Tamamlandi.
pause
