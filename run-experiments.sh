##!/usr/bin/env bash

# Roundtrip:

# German
echo "german"
python paraphrase.py --method roundtrip --translators marian-en-de marian-de-en --input input/tweets.csv

# Russian
echo "russian"
python paraphrase.py --method roundtrip --translators marian-en-ru marian-ru-en --input input/tweets.csv

# Icelandic
echo "icelandic"
python paraphrase.py --method roundtrip --translators marian-en-is marian-is-en --input input/tweets.csv

# Speech
echo "speech"
python paraphrase.py --method roundtrip --translators silero_asr_en tacotron_pytorch --input input/tweets.csv

# All
echo "all"
python paraphrase.py --method roundtrip --translators marian-en-de marian-de-en marian-en-ru \
       marian-ru-en marian-en-is marian-is-en silero_asr_en tacotron_pytorch --input input/tweets.csv

####

# N-cycles:

# All
echo "ncycles"
python paraphrase.py --method ncycles --translators marian-en-de marian-de-en marian-en-ru \
       marian-ru-en marian-en-is marian-is-en silero_asr_en tacotron_pytorch --input input/tweets.csv