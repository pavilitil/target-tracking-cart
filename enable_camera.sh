#!/usr/bin/env bash
set -e

CFG="/boot/firmware/config.txt"
if [ ! -f "${CFG}.bak" ]; then
    cp "$CFG" "${CFG}.bak"
fi

sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' "$CFG"
if ! grep -q "^dtoverlay=ov5647" "$CFG"; then
    echo "dtoverlay=ov5647" >> "$CFG"
fi

echo "========================================================"
echo "✅ Драйвер камери Waveshare OV5647 (dtoverlay=ov5647) увімкнено у $CFG!"
echo "Тепер виконайте перезавантаження: sudo reboot"
echo "========================================================"

