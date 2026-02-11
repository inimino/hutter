#!/bin/bash
# Total Interpretation: The Sat-RNN as a 128-Bit Boolean Automaton (Q1-Q7)
# Archive: https://cmpr.ai/hutter/archive/20260211/
#
# Model: sat-rnn (128 hidden, tanh, BPTT-50, 0.079 bpc on 1024 bytes)
# Data: first 1024 bytes of enwik9

set -e

DATA=enwik_1024.txt
MODEL=sat_model.bin

echo "=== Total Interpretation: Q1-Q7 ==="
echo "Archive: https://cmpr.ai/hutter/archive/20260211/"
echo ""

# Check dependencies
command -v gcc >/dev/null || { echo "ERROR: gcc required"; exit 1; }
command -v pdflatex >/dev/null || { echo "WARNING: pdflatex not found, skipping paper build"; NO_PDF=1; }

# Check data files
[ -f "$MODEL" ] || { echo "ERROR: $MODEL missing"; exit 1; }
[ -f "$DATA" ] || { echo "ERROR: $DATA missing (extract first 1024 bytes of enwik9)"; exit 1; }

T0=$(date +%s)
t() { echo "  ($(( $(date +%s) - T0 ))s elapsed)"; }

echo "1. Q1: Boolean automaton — sign-only dynamics (O(NH), ~1s)"
gcc -O2 -o q1_boolean q1_boolean.c -lm
./q1_boolean "$DATA" "$MODEL" 2>&1 | grep -E 'bpc|sign flip|entropy|mantissa'
t; echo ""

echo "2. Q1: Margin analysis — mantissa cannot flip signs (O(NH^2), ~1s)"
gcc -O2 -o q1_margins q1_margins.c -lm
./q1_margins "$DATA" "$MODEL" 2>&1 | grep -E 'margin|Safety|Mantissa|Boolean'
t; echo ""

echo "3. Q1: Attractor search — no attractors in 1024 bytes (O(N^2H), ~2s)"
gcc -O2 -o q1_bool_attractor q1_bool_attractor.c -lm
./q1_bool_attractor "$DATA" "$MODEL" 2>&1 | grep -E 'attractor|Unique|entropy'
t; echo ""

echo "4. Q1: Sparsity analysis — gradient mass at depth (O(NHD), ~1s)"
gcc -O2 -o q1_sparsity q1_sparsity.c -lm
./q1_sparsity "$DATA" "$MODEL" 2>&1 | grep -E 'patterns|W_h|median|depth'
t; echo ""

echo "5. Q2: Dominant offsets — MI-greedy offset selection (O(NHK), ~1s)"
gcc -O2 -o q2_offsets q2_offsets.c -lm
./q2_offsets "$DATA" "$MODEL" 2>&1 | grep -E 'd=|greedy|pair'
t; echo ""

echo "6. Q3: Neuron knockout — h28 = 99.7% of compression (O(NH), ~1s)"
gcc -O2 -o q3_neurons q3_neurons.c -lm
./q3_neurons "$DATA" "$MODEL" 2>&1 | grep -E 'h[0-9]|top|beat|bpc'
t; echo ""

echo "7. Q4: Saturation dynamics — all 128 volatile (O(NH), ~1s)"
gcc -O2 -o q4_saturation q4_saturation.c -lm
./q4_saturation "$DATA" "$MODEL" 2>&1 | grep -E 'volatile|frozen|dwell|flips'
t; echo ""

echo "8. Q5: Redux — minimal 15-neuron model (O(NH), ~2s)"
gcc -O2 -o q5_redux q5_redux.c -lm
./q5_redux "$DATA" "$MODEL" 2>&1 | grep -E 'Redux|params|bpc|column'
t; echo ""

echo "9. Q6: Per-prediction justifications (O(NH^2), ~2s)"
gcc -O2 -o q6_justify q6_justify.c -lm
./q6_justify "$DATA" "$MODEL" 2>&1 | grep -E 'Justification|driven|routing'
t; echo ""

echo "10. Q7: PMI alignment — RNN vs data statistics (O(NHD), ~2s)"
gcc -O2 -o q7_algebraic q7_algebraic.c -lm
./q7_algebraic "$DATA" "$MODEL" 2>&1 | grep -E 'Total|Shallow|Deep|align'
t; echo ""

echo "11. Q7: Higher-order PMI (trigram vs bigram) (O(NHD^2), ~20s)"
gcc -O2 -o q7_higher_order q7_higher_order.c -lm
./q7_higher_order "$DATA" "$MODEL" 2>&1 | grep -E 'Conclusion|Total|trigram'
t; echo ""

if [ -z "$NO_PDF" ]; then
    echo "12. Building papers"
    for tex in synthesis.tex boolean-automaton.tex q234-results.tex q6-justifications.tex q1-sparsity.tex; do
        pdflatex -interaction=nonstopmode "$tex" > /dev/null 2>&1
        pdflatex -interaction=nonstopmode "$tex" > /dev/null 2>&1
        echo "  built → ${tex%.tex}.pdf"
    done
    t; echo ""
fi

echo "=== Done ==="
echo ""
echo "Key results:"
echo "  Q1: Sign-only dynamics = 5.69 bpc (BETTER than full f32). Mantissa is noise."
echo "      Mean margin 60.5, safety factor 10^6x. Boolean function IS the computation."
echo "  Q2: Dominant offsets d=18-25 (23% of neurons). MI-greedy captures 9.4%."
echo "  Q3: h28 alone = 99.7% of compression gap. 113 neurons are noise for readout."
echo "  Q4: All 128 neurons volatile. Mean dwell 3.3 steps. Zero frozen."
echo "  Q5: 15 neurons + 36% W_h = 0.15 bpc better than full 128."
echo "  Q6: ~15 weights per prediction. Routing backbone: h54 <- h121 <- h78."
echo "  Q7: 74% RNN-PMI alignment. Shallow (d=1-4): 88%. Deep (d>15): 24-37%."
echo ""
echo "Total runtime: $(( $(date +%s) - T0 ))s"
