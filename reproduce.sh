#!/bin/bash
# Reproduce: Pattern Chain Analysis of a Saturated RNN
# Paper: https://cmpr.ai/hutter/archive/20260206/pattern-chains.pdf
#
# Prerequisites:
#   - gcc
#   - wget or curl
#   - enwik9.zip (will be downloaded if not present)
#
# The trained model (sat_model.bin) is available from the archive:
#   https://cmpr.ai/hutter/archive/20260206/sat_model.bin
#
# This script reproduces all results from the paper.

set -e

echo "=== Reproducing: Pattern Chain Analysis ==="

# 1. Get data
if [ ! -f enwik9 ]; then
    if [ ! -f enwik9.zip ]; then
        echo "Downloading enwik9..."
        wget -q http://mattmahoney.net/dc/enwik9.zip
    fi
    echo "Extracting enwik9..."
    unzip -q enwik9.zip
fi
head -c 1024 enwik9 > enwik_1024.txt
echo "Data: enwik_1024.txt (1024 bytes)"

# 2. Compile
echo "Compiling..."
gcc -O3 -o sat_train sat_train.c -lm
gcc -O3 -o sat_chains sat_chains.c -lm
gcc -O3 -o sat_ngram_um sat_ngram_um.c -lm

# 3. Get or train model
if [ ! -f sat_model.bin ]; then
    if command -v wget &> /dev/null; then
        echo "Downloading pre-trained model..."
        wget -q https://cmpr.ai/hutter/archive/20260206/sat_model.bin
    elif command -v curl &> /dev/null; then
        curl -sO https://cmpr.ai/hutter/archive/20260206/sat_model.bin
    fi
fi

if [ ! -f sat_model.bin ]; then
    echo "Training from scratch (4000 epochs, ~2 min)..."
    ./sat_train enwik_1024.txt sat 4000 500
    cp sat_best.bin sat_model.bin
fi

echo ""
echo "=== Chain Analysis ==="
./sat_chains enwik_1024.txt sat_model.bin 2>&1 | tail -40

echo ""
echo "=== N-gram UM Analysis ==="
./sat_ngram_um enwik_1024.txt 50

echo ""
echo "=== Done ==="
echo "Full paper: https://cmpr.ai/hutter/archive/20260206/pattern-chains.pdf"
