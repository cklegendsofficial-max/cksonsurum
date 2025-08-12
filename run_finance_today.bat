@echo off
cd /d C:\Project_Chimera
echo ========================================
echo    Enhanced Master Director Pipeline
echo ========================================
echo.
echo [1/1] Tek komut ile tüm akış başlıyor...
echo.
python main.py --channel CKFinanceCore --steps all
echo.
echo ========================================
echo    Pipeline tamamlandi!
echo ========================================
echo.
echo Cikti dosyalari:
echo - outputs\CKFinanceCore\YYYY-MM-DD\
echo - final_video.mp4 (ana video)
echo - captions\ (altyazılar)
echo - shorts\ (kısa videolar)
echo - report.md (rapor)
echo - metrics.jsonl (metrikler)
echo.
echo Raporu görüntülemek için:
echo notepad outputs\CKFinanceCore\YYYY-MM-DD\report.md
echo.
pause
