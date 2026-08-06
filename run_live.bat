@echo off
rem ============================================================
rem  Запуск SMT/QM системы на постоянку (проп-режим, EURUSD).
rem  Перед использованием подставьте свой стартовый баланс пропа
rem  в --initial-balance. Скрипт сам ждёт терминал при старте
rem  и переподключается при обрыве; этот батник перезапускает
rem  его, если процесс Python по любой причине завершился.
rem ============================================================
chcp 65001 >nul
cd /d %~dp0
rem Текущий профиль: АГРЕССИВНЫЙ (риск 2%, дневной 4%) — медианная просадка ~10.8%,
rem шансы пройти проп-челлендж ~50/50, медианный срок до +8% около 2 месяцев.
rem Консервативная альтернатива (риск 0.5%, дневной 3%): просадка p99 ~7%, срок ~7 мес:
rem python main.py --symbols EURUSD --models M15/M5 H1/M5 --require-qm --risk 0.005 --max-daily-loss 0.03 --close-on-limit --max-total-loss 0.08 --initial-balance 100000
:loop
python main.py --symbols EURUSD --models M15/M5 H1/M5 --require-qm --risk 0.02 --max-daily-loss 0.04 --close-on-limit --max-total-loss 0.08 --initial-balance 100000
echo [%date% %time%] Скрипт завершился, перезапуск через 15 секунд >> restart.log
timeout /t 15 /nobreak >nul
goto loop
