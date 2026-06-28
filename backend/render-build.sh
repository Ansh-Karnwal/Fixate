#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install Chromium INTO the Python package dir (site-packages) instead of
# ~/.cache, which Render's native runtime does not persist past the build.
# PLAYWRIGHT_BROWSERS_PATH=0 must also be set as a runtime env var in Render
# so the app looks in the same place.
export PLAYWRIGHT_BROWSERS_PATH=0
python -m playwright install chromium
