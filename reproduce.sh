#!/bin/bash
# Byte-Level Kneser-Ney Scaling: 1.784 bpc on full enwik9
# Archive: https://cmpr.ai/hutter/archive/20260212/
#
# Reproduces the KN scaling curve from the kn-scaling paper.
# Requires: gcc, enwik9 (1GB), ~3GB RAM for full run.
#
# Results (interpolated KN, 80/20 train/test):
#   10M  KN-6i D=0.8:  2.315 bpc
#   100M KN-6i D=0.8:  2.001 bpc
#   200M KN-6i D=0.8:  1.927 bpc
#   400M KN-6i D=0.8:  1.889 bpc
#   800M KN-6i D=0.8:  1.859 bpc  (HT 97%)
#   1B   KN-6i D=0.9:  1.784 bpc  (HT 100%)
#
# Michaeljohn Clement and Claude, February 2026.

set -e

ENWIK9="${1:-enwik9}"

echo "=== KN Scaling Reproduction ==="
echo ""

# Check dependencies
command -v gcc >/dev/null || { echo "ERROR: gcc required"; exit 1; }

# Check data
if [ ! -f "$ENWIK9" ]; then
    echo "ERROR: $ENWIK9 not found."
    echo "Download enwik9 from https://mattmahoney.net/dc/enwik9.zip"
    echo "Usage: $0 [path_to_enwik9]"
    exit 1
fi

SIZE=$(wc -c < "$ENWIK9")
echo "Data: $ENWIK9 ($SIZE bytes)"
echo ""

# Compile
echo "Compiling byte_kn.c..."
gcc -O2 -o byte_kn byte_kn.c -lm
echo ""

# Run scaling curve
echo "=== Scaling Curve ==="
echo ""

for N in 10000000 100000000 200000000; do
    if [ "$SIZE" -ge "$N" ]; then
        echo "--- N = $((N/1000000))M ---"
        ./byte_kn "$ENWIK9" $N 6 0.8
        echo ""
    fi
done

# 400M and 800M: HT starts to saturate but results are still good
for N in 400000000 800000000; do
    if [ "$SIZE" -ge "$N" ]; then
        echo "--- N = $((N/1000000))M ---"
        ./byte_kn "$ENWIK9" $N 6 0.8
        echo ""
    fi
done

# Full 1B: D=0.9 is slightly better at this scale
if [ "$SIZE" -ge 1000000000 ]; then
    echo "--- N = 1000M (full enwik9) ---"
    ./byte_kn "$ENWIK9" 1000000000 6 0.9
    echo ""
fi

echo "=== Done ==="
echo ""
echo "Expected results (interpolated KN-6, 80/20 split):"
echo "  10M:  ~2.32 bpc (D=0.8)"
echo "  100M: ~2.00 bpc (D=0.8)"
echo "  200M: ~1.93 bpc (D=0.8)"
echo "  400M: ~1.89 bpc (D=0.8, HT 73%)"
echo "  800M: ~1.86 bpc (D=0.8, HT 97%)"
echo "  1B:   ~1.78 bpc (D=0.9, HT 100%)"
echo ""
echo "Note: HT saturation at 800M+ causes some data loss."
echo "A larger hash table (256M+ entries) would improve results"
echo "but requires >4GB RAM."
