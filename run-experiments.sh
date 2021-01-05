##!/usr/bin/env bash

# Roundtrip:

# German
python paraphrase.py --method roundtrip --translators fair-wmt19-en-de fair-wmt19-de-en --input input/tweets.csv

# Russian
python paraphrase.py --method roundtrip --translators fair-wmt19-en-ru fair-wmt19-ru-en --input input/tweets.csv

# Tamil
python paraphrase.py --method roundtrip --translators fair-wmt20-en-ta fair-wmt20-ta-en --input input/tweets.csv

# Speech
python paraphrase.py --method roundtrip --translators silero_asr_en tacotron_pytorch --input input/tweets.csv

# All
python paraphrase.py --method roundtrip --translators fair-wmt19-en-de fair-wmt19-de-en fair-wmt19-en-ru \
       fair-wmt19-ru-en fair-wmt20-en-ta fair-wmt20-ta-en silero_asr_en tacotron_pytorch --input input/tweets.csv

####

# N-cycles:

# All
python paraphrase.py --method ncycles --translators fair-wmt19-en-de fair-wmt19-de-en fair-wmt19-en-ru \
       fair-wmt19-ru-en fair-wmt20-en-ta fair-wmt20-ta-en silero_asr_en tacotron_pytorch --input input/tweets.csv