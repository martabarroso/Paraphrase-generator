##!/usr/bin/env bash

# Roundtrip:

# German
echo "german"
python paraphrase.py --method roundtrip --translators fair-wmt19-en-de fair-wmt19-de-en --input input/tweets.csv

# Russian
echo "russian"
python paraphrase.py --method roundtrip --translators fair-wmt19-en-ru fair-wmt19-ru-en --input input/tweets.csv

# Tamil
echo "tamil"
python paraphrase.py --method roundtrip --translators fair-wmt20-en-ta fair-wmt20-ta-en --input input/tweets.csv

# Speech
echo "speech"
python paraphrase.py --method roundtrip --translators silero_asr_en tacotron_pytorch --input input/tweets.csv

# All
echo "all"
python paraphrase.py --method roundtrip --translators fair-wmt19-en-de fair-wmt19-de-en fair-wmt19-en-ru \
       fair-wmt19-ru-en fair-wmt20-en-ta fair-wmt20-ta-en silero_asr_en tacotron_pytorch --input input/tweets.csv

####

# N-cycles:

# All
echo "ncycles"
python paraphrase.py --method ncycles --translators fair-wmt19-en-de fair-wmt19-de-en fair-wmt19-en-ru \
       fair-wmt19-ru-en fair-wmt20-en-ta fair-wmt20-ta-en silero_asr_en tacotron_pytorch --input input/tweets.csv