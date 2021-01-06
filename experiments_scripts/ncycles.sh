##!/usr/bin/env bash

# N-cycles:

# All
echo "ncycles"
python paraphrase.py --method ncycles --translators marian-en-de marian-de-en marian-en-ru \
       marian-ru-en marian-en-is marian-is-en silero_asr_en tacotron_pytorch --input input/tweets.csv