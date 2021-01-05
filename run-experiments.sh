##!/usr/bin/env bash

# Roundtrip:

# German
echo "german"
python paraphrase.py --method roundtrip --translators fair-wmt19-en-de fair-wmt19-de-en --input input/tweets_.csv

# Russian
echo "russian"
python paraphrase.py --method roundtrip --translators fair-wmt19-en-ru fair-wmt19-ru-en --input input/minitweets.csv

# Tamil
echo "tamil"
python paraphrase.py --method roundtrip --translators fair-wmt20-en-ta fair-wmt20-ta-en --input input/minitweets.csv

# Speech
echo "speech"
python paraphrase.py --method roundtrip --translators silero_asr_en tacotron_pytorch --input input/minitweets.csv

# All
echo "all"
python paraphrase.py --method roundtrip --translators fair-wmt19-en-de fair-wmt19-de-en fair-wmt19-en-ru \
       fair-wmt19-ru-en fair-wmt20-en-ta fair-wmt20-ta-en silero_asr_en tacotron_pytorch --input input/minitweets.csv

####

# N-cycles:

# All
echo "ncycles"
python paraphrase.py --method ncycles --translators fair-wmt19-en-de fair-wmt19-de-en fair-wmt19-en-ru \
       fair-wmt19-ru-en fair-wmt20-en-ta fair-wmt20-ta-en silero_asr_en tacotron_pytorch --input input/minitweets.csv