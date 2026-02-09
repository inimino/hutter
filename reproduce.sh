#!/bin/bash
set -e

# Pattern Priors and Skip-Patterns
# Reproduces results from pattern-prior.pdf
# Archive: https://cmpr.ai/hutter/archive/20260207/

# Data: enwik_1024.txt is the first 1024 bytes of enwik9
# If not present, extract it:
if [ ! -f enwik_1024.txt ]; then
    if [ ! -f enwik9 ]; then
        echo "Download enwik9 from http://mattmahoney.net/dc/enwik9.zip"
        echo "Or provide enwik_1024.txt (first 1024 bytes)"
        exit 1
    fi
    head -c 1024 enwik9 > enwik_1024.txt
fi

# Second 1024 bytes for test data
if [ ! -f enwik_second_1024.txt ] && [ -f enwik9 ]; then
    dd if=enwik9 bs=1 skip=1024 count=1024 of=enwik_second_1024.txt 2>/dev/null
fi

echo "=== Compiling ==="
gcc -O2 -o backward_trie backward_trie.c -lm
gcc -O2 -o skip2gram skip2gram.c -lm
gcc -O2 -o skip3gram skip3gram.c -lm
gcc -O2 -o skip_kgram skip_kgram.c -lm
gcc -O2 -o offset_viz offset_viz.c -lm
gcc -O2 -o skip2_survival skip2_survival.c -lm
gcc -O2 -o skip2_rnn skip2_rnn.c -lm
gcc -O2 -o compare_wh compare_wh.c -lm
gcc -O2 -o construct_rnn construct_rnn.c -lm
gcc -O2 -o construct_skip construct_skip.c -lm
gcc -O3 -o construct_skip_greedy construct_skip_greedy.c -lm
gcc -O3 -o construct_skip_mlp construct_skip_mlp.c -lm

echo ""
echo "=== Section 2: Backward Trie (MI by offset, atomic patterns) ==="
./backward_trie enwik_1024.txt

echo ""
echo "=== Section 3: Skip-2-grams ==="
./skip2gram enwik_1024.txt 20

echo ""
echo "=== Section 3.2: Survival under DSS doubling ==="
if [ -f enwik9 ]; then
    ./skip2_survival enwik_1024.txt
else
    echo "(Requires enwik9 for second 1024 bytes — skipping)"
fi

echo ""
echo "=== Section 4: Greedy skip-k-grams ==="
./skip_kgram enwik_1024.txt

echo ""
echo "=== Section 5: Skip-patterns in RNN hidden state ==="
if [ -f sat_model.bin ]; then
    ./skip2_rnn enwik_1024.txt sat_model.bin
else
    echo "(Requires sat_model.bin — skipping)"
fi

echo ""
echo "=== Section 5.3: DSS doubling W_h comparison ==="
# Train two models for comparison (requires sat_train.c)
if [ -f sat_train.c ]; then
    gcc -O2 -o sat_train sat_train.c -lm
    echo "Training 1024-byte model (4000 epochs)..."
    ./sat_train enwik_1024.txt sat_1024.bin 4000 2>/dev/null
    if [ -f enwik9 ]; then
        head -c 2048 enwik9 > enwik_2048.txt
        echo "Training 2048-byte model (4000 epochs)..."
        ./sat_train enwik_2048.txt sat_2048.bin 4000 2>/dev/null
        echo "Comparing W_h between models..."
        ./compare_wh sat_1024.bin sat_2048.bin enwik_1024.txt
    fi
else
    echo "(Requires sat_train.c — skipping)"
fi

echo ""
echo "=== Section 7.1: Bigram construction ==="
./construct_rnn enwik_1024.txt bigram_constructed.bin

echo ""
echo "=== Section 7.2: Multi-offset construction ==="
echo "--- Contiguous 4 [1,2,3,4] ---"
./construct_skip_greedy enwik_1024.txt "1,2,3,4"
echo "--- Greedy 4 [1,8,20,3] ---"
./construct_skip_greedy enwik_1024.txt "1,8,20,3"
echo "--- Contiguous 8 [1..8] ---"
./construct_skip_greedy enwik_1024.txt "1,2,3,4,5,6,7,8"
echo "--- Greedy 8 [1,8,20,3,27,2,12,7] ---"
./construct_skip_greedy enwik_1024.txt "1,8,20,3,27,2,12,7"

echo ""
echo "=== Section 7.4: MLP readout ==="
echo "--- Greedy-8, MLP-256 ---"
./construct_skip_mlp enwik_1024.txt "1,8,20,3,27,2,12,7" 256
echo "--- Greedy-4, MLP-256 ---"
./construct_skip_mlp enwik_1024.txt "1,8,20,3" 256

echo ""
echo "=== Done ==="
echo "Key results:"
echo "  Backward trie MI: 2.69 bits at offset 1, decaying to ~1.6 at offset 10"
echo "  Greedy skip-4 [1,8,20,3]: 0.069 bpc (712 patterns)"
echo "  Greedy skip-8: 0.043 bpc (834 patterns)"
echo "  Construction greedy-8 linear: 0.190 bpc"
echo "  Construction greedy-8 MLP-256: 0.137 bpc"
