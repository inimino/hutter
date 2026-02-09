#!/bin/bash
# Pattern Chains: Factor map from RNN neurons to UM skip-patterns
# Archive: https://cmpr.ai/hutter/archive/20260208/
#
# Model: sat-rnn (128 hidden, tanh, BPTT-50, 0.079 bpc on 1024 bytes)
# Data: first 1024 bytes of enwik9

set -e

DATA=enwik_1024.txt
MODEL=sat_model.bin

echo "=== Pattern Chains Reproduction ==="
echo ""

# Check dependencies
command -v gcc >/dev/null || { echo "ERROR: gcc required"; exit 1; }
command -v pdflatex >/dev/null || { echo "WARNING: pdflatex not found, skipping paper build"; NO_PDF=1; }

# Check data files
[ -f "$MODEL" ] || { echo "ERROR: $MODEL missing"; exit 1; }
[ -f "$DATA" ] || { echo "ERROR: $DATA missing"; exit 1; }

echo "1. Binary state analysis"
gcc -O2 -o binary_states binary_states.c -lm
./binary_states "$DATA" "$MODEL"
echo ""

echo "2. Factor map v1: MI + ablation"
gcc -O2 -o factor_map factor_map.c -lm
./factor_map "$DATA" "$MODEL"
echo ""

echo "3. Factor map v2: continuous conditional means, R^2"
gcc -O2 -o factor_map2 factor_map2.c -lm
./factor_map2 "$DATA" "$MODEL"
echo ""

echo "4. Factor map v3: state features (word_len, in_tag, char_class)"
gcc -O2 -o factor_map3 factor_map3.c -lm
./factor_map3 "$DATA" "$MODEL"
echo ""

echo "5. Factor map v4: combined 2-offset + state"
gcc -O2 -o factor_map4 factor_map4.c -lm
./factor_map4 "$DATA" "$MODEL"
echo ""

echo "6. UM learning: greedy offset selection, superset verification"
gcc -O2 -o um_learn um_learn.c -lm
./um_learn "$DATA"
echo ""

echo "7. Jacobian chain trace"
gcc -O2 -o rnn_trace rnn_trace.c -lm
./rnn_trace "$DATA" "$MODEL"
echo ""

echo "8. Numerical trace examples"
gcc -O2 -o trace_example trace_example.c -lm
./trace_example "$DATA" "$MODEL"
echo ""

echo "9. Reverse isomorphism v1: hash-based"
gcc -O2 -o reverse_iso reverse_iso.c -lm
./reverse_iso "$DATA"
echo ""

echo "10. Reverse isomorphism v2: UM conditional probs"
gcc -O2 -o reverse_iso2 reverse_iso2.c -lm
./reverse_iso2 "$DATA"
echo ""

echo "11. Pattern viewer (HTML generation)"
gcc -O2 -o pattern_viewer pattern_viewer.c -lm
./pattern_viewer "$DATA" "$MODEL" > pattern-viewer.html
echo "  generated → pattern-viewer.html ($(wc -c < pattern-viewer.html) bytes)"
echo ""

if [ -z "$NO_PDF" ]; then
    echo "12. Building paper"
    pdflatex -interaction=nonstopmode pattern-chains.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode pattern-chains.tex > /dev/null 2>&1
    echo "  built → pattern-chains.pdf"
    echo ""
fi

echo "=== Done ==="
echo ""
echo "Key results:"
echo "  - 128/128 neurons explained as 2-offset conjunctions (mean R^2=0.837)"
echo "  - 2-offset + word_len + in_tag = 0.43 bpc (92.5% of gain)"
echo "  - UM greedy-6 = 0.000 bpc (perfect on 1024 bytes)"
echo "  - Reverse isomorphism: 0.107 bpc (within 0.03 of trained 0.079)"
echo "  - Gradient-based interpretability fails (chaos); statistical factor map succeeds"
