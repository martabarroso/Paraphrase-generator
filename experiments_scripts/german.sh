##!/usr/bin/env bash

# Roundtrip:

# German
echo "german"
python paraphrase.py --method roundtrip --translators marian-en-de marian-de-en --input input/tweets.csv
