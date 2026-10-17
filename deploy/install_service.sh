#!/bin/bash
# Install / update the systemd service that runs the robot at boot and restarts it if it ever stops.
#   ./deploy/install_service.sh attack      (striker: main_attack.py)
#   ./deploy/install_service.sh defense     (goalie: main.py)
# Run it on the Pi as the normal user (it uses sudo). Replaces /etc/systemd/system/germinc.service.
# Afterwards:  sudo systemctl stop germinc     before running anything by hand (only one program can open the camera)
#              sudo systemctl start germinc    to start it again
#              journalctl -u germinc -f        to watch its output (crashes, watchdog restarts)
set -e
case "$1" in
    attack)  SCRIPT=main_attack.py; STOP=STOP_attack.py ;;
    defense) SCRIPT=main.py;        STOP=STOP_defense.py ;;
    *) echo "usage: $0 attack|defense"; exit 1 ;;
esac
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VENV:-$HOME/env}/bin/python3"
[ -x "$PY" ] || { echo "no python at $PY (set VENV=/path/to/venv)"; exit 1; }
sed -e "s|VENV_PYTHON|$PY|g" -e "s|CODE_DIR|$CODE_DIR|g" -e "s|STOP_SCRIPT|$STOP|g" -e "s|SCRIPT|$SCRIPT|g" \
    -e "s|^User=USER|User=$USER|" "$CODE_DIR/deploy/germinc.service" | sudo tee /etc/systemd/system/germinc.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable germinc
sudo systemctl restart germinc
echo "germinc.service runs $CODE_DIR/$SCRIPT as $USER with $PY (restarts on any exit)."
