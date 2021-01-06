##!/usr/bin/env bash

# Roundtrip:
# Russian
echo "russian"
python paraphrase.py --method roundtrip --translators marian-en-ru marian-ru-en --input input/tweets.csv
