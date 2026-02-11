#!/bin/bash
# Weight Construction + Event Space Isomorphism
# Archive: https://cmpr.ai/hutter/archive/20260211/
#
# Model: sat-rnn (128 hidden, tanh, BPTT-50, 0.079 bpc on 1024 bytes)
# Data: first 1024 bytes of enwik9 (weight construction),
#       first 262144 bytes of enwik9 (ES isomorphism)
#
# NOTE: enwik9 must be available. Download from:
#   wget http://mattmahoney.net/dc/enwik9.zip && unzip enwik9.zip

set -e

DATA=enwik_1024.txt
MODEL=sat_model.bin
ENWIK9="${ENWIK9:-enwik9}"  # override with ENWIK9=/path/to/enwik9

echo "=== Weight Construction + Event Space Isomorphism ==="
echo "Archive: https://cmpr.ai/hutter/archive/20260211/"
echo ""

# Check dependencies
command -v gcc >/dev/null || { echo "ERROR: gcc required"; exit 1; }
command -v pdflatex >/dev/null || { echo "WARNING: pdflatex not found, skipping paper build"; NO_PDF=1; }

# Check data files
[ -f "$MODEL" ] || { echo "ERROR: $MODEL missing"; exit 1; }
[ -f "$DATA" ] || { echo "ERROR: $DATA missing"; exit 1; }

T0=$(date +%s)
t() { echo "  ($(( $(date +%s) - T0 ))s elapsed)"; }

# ========== Weight Construction ==========

echo "1. write_weights6: Proper hash + analytic W_y (O(G*8*N*256), ~4min)"
echo "   Key result: ZERO optimization → 1.89 bpc (all 82k params from data stats)"
gcc -O2 -o write_weights6 write_weights6.c -lm
./write_weights6 "$DATA" "$MODEL" 2>&1 | grep -E 'bpc|=== |hash|SUMMARY|Config'
t; echo ""

echo "2. write_weights12: Optimization continuum (O(N*H^2*epochs), ~2min)"
echo "   Key result: closed-form 1.56 → Newton 0.97 → SGD 0.59 → trained 0.079"
gcc -O2 -o write_weights12 write_weights12.c -lm
./write_weights12 "$DATA" "$MODEL" 2>&1 | grep -E 'bpc|=== |SUMMARY|hierarchy|Pseudo|Newton|SGD|Per-offset|FINAL'
t; echo ""

echo "3. write_weights5: Fully analytic survey (O(N*H*256*300), ~3min)"
echo "   5 construction methods, grid search over alpha/scale"
gcc -O2 -o write_weights5 write_weights5.c -lm
./write_weights5 "$DATA" "$MODEL" 2>&1 | grep -E 'bpc|=== |Config|Uniform|Analytic|Naive|Trained|loop'
t; echo ""

# ========== Supporting experiments (reference) ==========

echo "4. write_weights: PMI-based W_y (O(NH), ~10s)"
gcc -O2 -o write_weights_base write_weights.c -lm
timeout 120 ./write_weights_base "$DATA" "$MODEL" 2>&1 | grep -E 'bpc|Summary|corr' || echo "  (partial output)"
t; echo ""

echo "5. write_weights3: Hebbian covariance (O(NH^2), ~10s)"
gcc -O2 -o write_weights3 write_weights3.c -lm
timeout 120 ./write_weights3 "$DATA" "$MODEL" 2>&1 | grep -E 'bpc|corr|Hebbian|blend|sign' | head -n 10 || echo "  (partial output)"
t; echo ""

# ========== Event Space Isomorphism ==========

if [ -f "$ENWIK9" ]; then
    echo "6. ES Isomorphism: SVD partition vs human partition (O(256^2*16*300), ~6s)"
    echo "   Key result: V-side up to 85.7% acc, 0.661 NMI. U-side = refinement."
    gcc -O2 -o es_iso es_iso.c -lm
    ./es_iso "$ENWIK9" 262144 2>&1 | grep -E 'Accuracy|NMI|Centroid|=== |SUMMARY|Mean'
    t; echo ""

    echo "7. ES data generation for viewer (O(256^2*16*300), ~6s)"
    gcc -O2 -o es_data_gen es_data_gen.c -lm
    ./es_data_gen "$ENWIK9" 262144 > es_data.js 2>/dev/null
    echo "  Generated es_data.js ($(wc -c < es_data.js) bytes)"
    rm -f es_data.js
    t; echo ""
else
    echo "6-7. SKIPPED: enwik9 not found. Set ENWIK9=/path/to/enwik9"
    echo ""
fi

echo "8. Cost analysis: FLOP comparison analytic vs SGD (O(1), ~20s)"
if [ -f "$ENWIK9" ]; then
    gcc -O2 -o cost_analysis cost_analysis.c -lm
    ./cost_analysis "$ENWIK9" 262144 2>&1 | grep -E 'FLOP|ratio|OoM|cost|\$|=== '
    t; echo ""
else
    echo "  SKIPPED: enwik9 not found"
fi

# ========== Papers ==========

if [ -z "$NO_PDF" ]; then
    echo "9. Building papers"
    for tex in narrative.tex cost.tex es-isomorphism.tex; do
        pdflatex -interaction=nonstopmode "$tex" > /dev/null 2>&1
        pdflatex -interaction=nonstopmode "$tex" > /dev/null 2>&1
        echo "  built → ${tex%.tex}.pdf"
    done
    t; echo ""
fi

echo "=== Done ==="
echo ""
echo "Key results:"
echo "  FULLY ANALYTIC: 1.89 bpc with ZERO optimization"
echo "    All 82k params from data: hash W_x, shift-register W_h, log-ratio W_y"
echo "    Beats trained model (4.97) by 3.08 bpc"
echo "    Generalizes comparably: test 4.88 vs trained 5.08"
echo "  OPTIMIZATION CONTINUUM: closed-form 1.56 → PI+Newton 0.97 → SGD 0.59"
echo "  ES ISOMORPHISM: V-side 85.7% accuracy (offset 11), 0.661 NMI"
echo "    U-side is a refinement: SVD discovers sub-category structure"
echo "  COST: Analytic = 149 MFLOP, SGD = 5.94 TFLOP. Ratio: 39,800x (4.6 OoM)"
echo ""
echo "Total runtime: $(( $(date +%s) - T0 ))s"
