##!/usr/bin/env bash

# Roundtrip:

# Speech
echo "speech"
python paraphrase.py --method roundtrip --translators silero_asr_en tacotron_pytorch --input input/tweets.csv
