##!/usr/bin/env bash

# N-cycles:
# Speech
python paraphrase.py --method ncycles --translators silero_asr_en tacotron_pytorch --input input/tweets.csv