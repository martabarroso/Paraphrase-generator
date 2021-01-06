##!/usr/bin/env bash

# Roundtrip:

# All
echo "all"
python paraphrase.py --method roundtrip --translators marian-en-de marian-de-en marian-en-ru \
       marian-ru-en marian-en-is marian-is-en silero_asr_en tacotron_pytorch --input input/tweets.csv
