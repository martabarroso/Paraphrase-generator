#!/bin/bash

# Dummy (identity)

python paraphrase.py --method dummy --translators fair-wmt19-en-de fair-wmt19-de-en --input input/example.txt

# Round-trip
python paraphrase.py --method roundtrip --translators fair-wmt19-en-de fair-wmt19-de-en --input input/example.txt