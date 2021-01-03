#!/usr/bin/env bash

python3.7 -m venv venv
python3.7 -m pip install -r requirements.txt

bash get-third-party.sh
