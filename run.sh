#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ] || [ ! -x "venv/bin/python" ]; then
    echo "[RobotCart] Створення віртуального середовища venv (--system-site-packages)..."
    python3 -m venv --system-site-packages venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

if [ -e /dev/ttyUSB0 ] && ! groups "$USER" | grep -qE '\bdialout\b'; then
    echo "[RobotCart] Увага: користувач $USER не в групі dialout. Для доступу до /dev/ttyUSB0 виконайте: sudo usermod -aG dialout $USER"
fi

exec ./venv/bin/python app.py

