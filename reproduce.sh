#!/bin/bash
# Reproduce SN visibility results for the saturated RNN.
#
# Prerequisites:
#   - gcc with -lm
#   - enwik9 in current directory (or parent)
#   - sat_model.bin (downloaded or symlinked)
#
# This script produces:
#   events.sn, patterns.sn      - Full RNN SN export (2,999 patterns)
#   events_um.sn, patterns_um.sn - N-gram UM SN export (1,915 patterns)
#   calibration.tsv              - Strength calibration (1,029 n-grams)

set -e

# Find enwik9
ENWIK9=""
for p in enwik9 ../enwik9 ../../enwik9; do
    if [ -f "$p" ]; then ENWIK9="$p"; break; fi
done
if [ -z "$ENWIK9" ]; then
    echo "enwik9 not found. Download from http://mattmahoney.net/dc/enwik9.zip"
    exit 1
fi

# Find model
MODEL="sat_model.bin"
if [ ! -f "$MODEL" ]; then
    echo "sat_model.bin not found. Download from https://cmpr.ai/hutter/archive/20260206/sat_model.bin"
    exit 1
fi

# Extract first 1024 bytes
head -c 1024 "$ENWIK9" > enwik_1024.txt

echo "=== Step 1: Full SN Export ==="
gcc -O3 -o sat_sn_full sat_sn_full.c -lm
./sat_sn_full "$MODEL" .

echo ""
echo "=== Step 2: N-gram UM to SN ==="
gcc -O3 -o sat_um_sn sat_um_sn.c -lm
./sat_um_sn enwik_1024.txt 11 .

echo ""
echo "=== Step 3: Strength Calibration ==="
gcc -O3 -o sat_calibrate sat_calibrate.c -lm
./sat_calibrate enwik_1024.txt "$MODEL" calibration.tsv

echo ""
echo "=== Done ==="
echo "Files produced:"
ls -la events.sn patterns.sn events_um.sn patterns_um.sn calibration.tsv
echo ""
echo "View results: open sn-view-sat.html in a browser (needs a local HTTP server)"
