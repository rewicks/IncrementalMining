#!/usr/bin/env bash

echo -e "5 \n10 \n20 \n30 \n40 \n50 \n70 \n100 \n150 \n200 \n250 \n300 \n400 \n500 \n " | \
parallel -j5 -I% --max-args 1 "./main.py --num-iterations % >& err.%"
