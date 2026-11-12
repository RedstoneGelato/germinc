#!/bin/bash
# Fresh Raspberry Pi OS + git clone -> ready to run. As the normal user (it uses sudo):
#   bash setup.sh attack      striker: installs everything and starts main_attack.py at boot
#   bash setup.sh defense     goalie: same with main_defence.py
#   bash setup.sh             installs everything, no autostart
# Fixed IPv4 address (optional, for the network connected right now):
#   STATIC_IP=192.168.1.51/24 GATEWAY=192.168.1.1 bash setup.sh attack
set -euo pipefail

ROLE="${1:-}"
case "$ROLE" in
    attack|defense|"") ;;
    *) echo "usage: bash $0 [attack|defense]"; exit 1 ;;
esac
if [ "$(id -u)" -eq 0 ]; then
    echo "run as the normal user, not with sudo (the venv and service belong to that user)"; exit 1
fi
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/env}"

echo "== system packages"
sudo apt-get update
# picamera2 / libcamera only come from apt; numpy and OpenCV from apt too so they match picamera2's build
sudo apt-get install -y git i2c-tools python3-venv python3-pip python3-picamera2 python3-numpy python3-opencv \
    python3-gpiozero python3-lgpio

echo "== I2C and VNC on, user groups"
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_vnc 0
BOOT_CONFIG=/boot/firmware/config.txt
I2C_OVERLAY="dtoverlay=i2c1,pins_2_3,baudrate=1000000"
grep -qxF "$I2C_OVERLAY" "$BOOT_CONFIG" || echo "$I2C_OVERLAY" | sudo tee -a "$BOOT_CONFIG" >/dev/null
sudo usermod -aG i2c,gpio,video "$USER"

if [ -n "${STATIC_IP:-}" ]; then
    [ -n "${GATEWAY:-}" ] || { echo "STATIC_IP needs GATEWAY too (the router's address)"; exit 1; }
    CONNECTION="$(nmcli -t -f NAME,TYPE connection show --active | grep -m1 -E ':(802-11-wireless|802-3-ethernet)$' | cut -d: -f1 || true)"
    [ -n "$CONNECTION" ] || { echo "no active Wi-Fi / Ethernet connection to give a fixed address"; exit 1; }
    echo "== fixed address $STATIC_IP on '$CONNECTION' (from the next reconnect / reboot)"
    sudo nmcli connection modify "$CONNECTION" ipv4.method manual ipv4.addresses "$STATIC_IP" \
        ipv4.gateway "$GATEWAY" ipv4.dns "${DNS:-$GATEWAY}"
fi

echo "== python venv at $VENV"
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install --upgrade adafruit-blinka adafruit-circuitpython-bno08x smbus2 \
    "git+https://github.com/Aw3someAndrew/SteelBar_CircuitPython_powerful_bldc_driver.git"

echo "== checking the robot code's imports"
"$VENV/bin/python" -c "import cv2, numpy, picamera2, gpiozero, board, busio, smbus2, adafruit_bno08x, steelbar_powerful_bldc_driver; print('all imports OK')"
"$VENV/bin/python" -m py_compile "$CODE_DIR"/*.py

if [ -n "$ROLE" ] && [ -f "$CODE_DIR/robot_vision_config.json" ]; then
    echo "== autostart service ($ROLE)"
    VENV="$VENV" bash "$CODE_DIR/deploy/install_service.sh" "$ROLE"
elif [ -n "$ROLE" ]; then
    # without the camera calibration the robot program exits at once and systemd would restart it every second
    echo "== autostart skipped: no robot_vision_config.json yet - calibrate (step 3 below), then run: bash setup.sh $ROLE"
fi

cat <<EOF

Done. Next:
  1. sudo reboot                          (I2C and the new groups need it)
  2. source $VENV/bin/activate && cd $CODE_DIR
  3. python3 test_camera.py               -> calibrate and "Generate config file" (robot_vision_config.json, per robot)
  4. i2cdetect -y 1                       -> motor drivers, IMU and PCB should all show up
  5. bash setup.sh attack|defense         (again, once calibrated: installs the autostart service)
EOF
