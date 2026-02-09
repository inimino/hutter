/*
 * backward_trie.c — Build the trie backwards (from output to input).
 *
 * The forward trie asks: "given context a1..ak, what output follows?"
 * The backward trie asks: "given output y, what input patterns predict it?"
 *
 * For each output byte y, we build a trie of the contexts that preceded it.
 * This reveals:
 *   - Atomic patterns (depth 1): single input → output. The elementary
 *     building blocks. These are the same as forward bigrams but viewed
 *     from the output's perspective.
 *   - Compound patterns (depth 2+): sequences of inputs that predict
 *     this output. Each depth adds one more byte of context.
 *
 * Skip-patterns: when we multiply by time (unfold the time dimension),
 * atomic patterns at different offsets become skip-patterns. A pattern
 * "if 'a' appeared 3 steps ago and 'b' appeared 1 step ago → predict 'c'"
 * chains two atomic observations across a time gap.
 *
 * This relates to attention: attention does the same thing at runtime,
 * using learned embeddings to select which positions matter. The backward
 * trie is the static, exhaustive version of what attention learns to do
 * selectively.
 *
 * Usage: backward_trie <data_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_ORDER 12
#define MAX_DATA 2048

/* === Per-output analysis === */

typedef struct {
    int byte;
    int total_count;          /* how many times this byte appears as output */
    int n_contexts[MAX_ORDER + 1]; /* distinct contexts at each depth */
    double entropy[MAX_ORDER + 1]; /* conditional entropy at each depth */
    int atomic_inputs[256];   /* count of each input byte preceding this output */
    int n_atomic;             /* number of distinct atomic inputs */
} OutputProfile;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file>\n", argv[0]);
        return 1;
    }

    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[MAX_DATA];
    int len = fread(data, 1, sizeof(data), df);
    fclose(df);
    printf("Data: %d bytes\n\n", len);

    /* Count output frequencies */
    int out_count[256] = {0};
    for (int t = 0; t < len - 1; t++)
        out_count[data[t + 1]]++;

    /* === 1. Atomic patterns (depth 1): for each output, which inputs precede it? === */
    printf("=== Atomic Patterns (depth 1) ===\n\n");
    printf("For each output byte, which input bytes precede it?\n\n");

    int bigram[256][256] = {{0}};
    for (int t = 0; t < len - 1; t++)
        bigram[data[t]][data[t + 1]]++;

    /* Sort outputs by frequency */
    int oidx[256];
    for (int i = 0; i < 256; i++) oidx[i] = i;
    for (int a = 0; a < 255; a++)
        for (int b = a + 1; b < 256; b++)
            if (out_count[oidx[b]] > out_count[oidx[a]]) {
                int tmp = oidx[a]; oidx[a] = oidx[b]; oidx[b] = tmp;
            }

    printf("output  chr  count  #inputs  H(input|output)  top inputs\n");
    int total_atomic = 0;
    for (int rank = 0; rank < 20 && out_count[oidx[rank]] > 0; rank++) {
        int y = oidx[rank];
        char yc = (y >= 32 && y < 127) ? (char)y : '.';

        /* Count distinct inputs and compute conditional entropy */
        int n_inp = 0;
        double h = 0;
        for (int x = 0; x < 256; x++) {
            if (bigram[x][y] > 0) {
                n_inp++;
                double p = (double)bigram[x][y] / out_count[y];
                h -= p * log2(p);
            }
        }
        total_atomic += n_inp;

        /* Top 3 inputs */
        int top[3] = {-1, -1, -1};
        int top_c[3] = {0, 0, 0};
        for (int x = 0; x < 256; x++) {
            if (bigram[x][y] > top_c[0]) {
                top[2] = top[1]; top_c[2] = top_c[1];
                top[1] = top[0]; top_c[1] = top_c[0];
                top[0] = x; top_c[0] = bigram[x][y];
            } else if (bigram[x][y] > top_c[1]) {
                top[2] = top[1]; top_c[2] = top_c[1];
                top[1] = x; top_c[1] = bigram[x][y];
            } else if (bigram[x][y] > top_c[2]) {
                top[2] = x; top_c[2] = bigram[x][y];
            }
        }

        printf("0x%02X    '%c'  %5d  %7d  %7.3f          ",
               y, yc, out_count[y], n_inp, h);
        for (int i = 0; i < 3 && top[i] >= 0; i++) {
            char tc = (top[i] >= 32 && top[i] < 127) ? (char)top[i] : '.';
            printf("'%c':%d ", tc, top_c[i]);
        }
        printf("\n");
    }

    printf("\nTotal atomic patterns: %d\n", total_atomic);

    /* === 2. Per-output context growth === */
    /* For each output byte, how does the number of distinct contexts grow? */
    printf("\n=== Context Growth per Output ===\n\n");
    printf("How many distinct contexts predict each output at increasing depth?\n\n");

    printf("output  chr  d=1   d=2   d=3   d=4   d=5   d=6\n");
    for (int rank = 0; rank < 10 && out_count[oidx[rank]] > 0; rank++) {
        int y = oidx[rank];
        char yc = (y >= 32 && y < 127) ? (char)y : '.';
        printf("0x%02X    '%c'", y, yc);

        for (int depth = 1; depth <= 6; depth++) {
            /* Count distinct contexts of length 'depth' that precede output y */
            /* Use a hash set (simple: count unique context hashes) */
            int n_ctx = 0;
            unsigned long seen[2048] = {0};  /* simple hash */
            int seen_n = 0;

            for (int t = depth - 1; t < len - 1; t++) {
                if (data[t + 1] != y) continue;

                /* Compute context hash */
                unsigned long h = 0;
                for (int k = 0; k < depth; k++)
                    h = h * 257 + data[t - k] + 1;

                /* Check if seen */
                int found = 0;
                for (int i = 0; i < seen_n; i++) {
                    if (seen[i] == h) { found = 1; break; }
                }
                if (!found && seen_n < 2048) {
                    seen[seen_n++] = h;
                    n_ctx++;
                }
            }
            printf("  %4d", n_ctx);
        }
        printf("\n");
    }

    /* === 3. Skip-pattern analysis === */
    /* For each output, find pairs of inputs at different offsets that
     * co-predict it. This is the "multiply by time" operation. */
    printf("\n=== Skip-Patterns (offset pairs) ===\n\n");
    printf("For the most common output, which (input, offset) pairs co-occur?\n");
    printf("This shows the atomic patterns that combine into skip-patterns.\n\n");

    int top_output = oidx[0];
    char top_c = (top_output >= 32 && top_output < 127) ? (char)top_output : '.';
    printf("Output: '%c' (0x%02X), count=%d\n\n", top_c, top_output, out_count[top_output]);

    /* For each offset (1..6), which input bytes appear before this output? */
    printf("offset  #inputs  top inputs\n");
    for (int offset = 1; offset <= 10; offset++) {
        int inp_at_offset[256] = {0};
        int n_valid = 0;
        for (int t = 0; t < len - 1; t++) {
            if (data[t + 1] != top_output) continue;
            if (t - offset + 1 < 0) continue;
            inp_at_offset[data[t - offset + 1]]++;
            n_valid++;
        }

        int n_inp = 0;
        for (int x = 0; x < 256; x++)
            if (inp_at_offset[x] > 0) n_inp++;

        printf("%6d  %7d  ", offset, n_inp);

        /* Top 3 */
        for (int rank = 0; rank < 3; rank++) {
            int best = -1, best_c = 0;
            for (int x = 0; x < 256; x++) {
                if (inp_at_offset[x] > best_c) {
                    best = x; best_c = inp_at_offset[x];
                }
            }
            if (best >= 0) {
                char bc = (best >= 32 && best < 127) ? (char)best : '.';
                printf("'%c':%d ", bc, best_c);
                inp_at_offset[best] = 0;  /* remove for next rank */
            }
        }
        printf("\n");
    }

    /* === 4. Mutual information between offset pairs === */
    printf("\n=== Mutual Information: I(input@offset; output) ===\n\n");
    printf("How much does knowing the input at each offset tell us about the output?\n\n");

    printf("offset  MI (bits)  MI/H(out)\n");
    /* H(output) */
    double h_out = 0;
    for (int y = 0; y < 256; y++) {
        if (out_count[y] == 0) continue;
        double p = (double)out_count[y] / (len - 1);
        h_out -= p * log2(p);
    }

    for (int offset = 1; offset <= 10; offset++) {
        /* MI(X_{t-offset+1}; Y_{t+1}) = H(Y) - H(Y|X) */
        /* H(Y|X) = sum_x P(x) * H(Y|X=x) */

        int count_x[256] = {0};
        int joint[256][256] = {{0}};  /* joint[x][y] */

        int n_valid = 0;
        for (int t = 0; t < len - 1; t++) {
            int off_pos = t - offset + 1;
            if (off_pos < 0) continue;
            int x = data[off_pos];
            int y = data[t + 1];
            count_x[x]++;
            joint[x][y]++;
            n_valid++;
        }

        double h_y_given_x = 0;
        for (int x = 0; x < 256; x++) {
            if (count_x[x] == 0) continue;
            double px = (double)count_x[x] / n_valid;
            double hx = 0;
            for (int y = 0; y < 256; y++) {
                if (joint[x][y] == 0) continue;
                double py_x = (double)joint[x][y] / count_x[x];
                hx -= py_x * log2(py_x);
            }
            h_y_given_x += px * hx;
        }

        double mi = h_out - h_y_given_x;
        printf("%6d  %9.4f  %7.3f\n", offset, mi, mi / h_out);
    }

    printf("\nH(output) = %.4f bits\n", h_out);

    /* === 5. Attention analogy summary === */
    printf("\n=== Attention Analogy ===\n\n");
    printf("The backward trie shows which inputs predict each output.\n");
    printf("At each offset, inputs contribute different amounts of MI.\n");
    printf("Attention does this at runtime: for each query (output position),\n");
    printf("it learns which keys (input positions/offsets) to attend to.\n");
    printf("\n");
    printf("The pattern-chain UM's explicit skip-patterns are the static,\n");
    printf("exhaustive version of what attention learns to do selectively.\n");
    printf("The backward trie IS the attention map, computed from data.\n");

    return 0;
}
