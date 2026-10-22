#!/bin/sh
# Starts the Ling chat server (strata-ling.json) as a user service, unit strata-ling-8082
systemctl --user stop strata-ling-8082 2>/dev/null
systemctl --user reset-failed strata-ling-8082 2>/dev/null
exec systemd-run --user --unit strata-ling-8082 -p WorkingDirectory=/home/debian/Strata-flint \
  /home/debian/Strata-up/.venv/bin/python -m serve.server --engine strata --config strata-ling.json --port 8082
