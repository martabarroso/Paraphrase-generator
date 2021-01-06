##!/usr/bin/env bash

# Roundtrip:

# Icelandic
echo "icelandic"
python paraphrase.py --method roundtrip --translators marian-en-is marian-is-en --input input/tweets.csv
