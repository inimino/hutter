/*
 * reverse_iso.c — Reverse isomorphism: dataset → UM patterns → RNN weights.
 *
 * Pipeline: data → pattern discovery → construct hidden states → optimize W_y
 *
 * Three constructions compared:
 *   1. Bigram (offset 1 only): h[t] = hash(data[t])
 *   2. Skip-8 (8 offsets): h[t] = concat(hash(data[t-d]) for d in offsets)
 *   3. Factor-map guided: h[t] from 12 offset pairs matching trained RNN
 *
 * All share the same procedure: construct deterministic h[t] from data,
 * then optimize W_y via gradient descent. No W_x/W_h training needed.
 *
 * Usage: reverse_iso <data_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 2048

static unsigned char data[MAX_DATA];
static int N;

/* Knuth multiplicative hash for deterministic neuron encoding */
static float hash_sign(int neuron, int byte) {
    unsigned int seed = (unsigned int)(neuron * 65537 + byte);
    seed = seed * 2654435761u;
    seed = (seed >> 16) ^ seed;
    return ((seed & 1) ? 0.95f : -0.95f);
}

/* Optimize W_y given ideal h vectors at each position.
 * Returns final bpc. */
static float optimize_wy(float h[][HIDDEN_SIZE], int start, int end,
                         int n_epochs, int verbose) {
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
    memset(Wy, 0, sizeof(Wy));

    /* Initialize by from unigram */
    int marginal[256] = {0};
    for (int t = 0; t < N; t++) marginal[data[t]]++;
    for (int y = 0; y < OUTPUT_SIZE; y++)
        by[y] = logf((marginal[y] + 0.5f) / (N + 128.0f));

    int n_valid = end - start;
    float bpc = 99.0f;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        float dWy[OUTPUT_SIZE][HIDDEN_SIZE];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));
        double total_loss = 0;

        for (int t = start; t < end; t++) {
            float* hv = h[t];
            int y_true = data[t + 1];

            float logits[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float sum = by[y];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    sum += Wy[y][j] * hv[j];
                logits[y] = sum;
            }

            float maxl = logits[0];
            for (int y = 1; y < OUTPUT_SIZE; y++)
                if (logits[y] > maxl) maxl = logits[y];
            float sum_exp = 0;
            float probs[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                probs[y] = expf(logits[y] - maxl);
                sum_exp += probs[y];
            }
            for (int y = 0; y < OUTPUT_SIZE; y++)
                probs[y] /= sum_exp;

            total_loss -= logf(probs[y_true] + 1e-10f);

            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float err = probs[y] - (y == y_true ? 1.0f : 0.0f);
                dby[y] += err;
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    dWy[y][j] += err * hv[j];
            }
        }

        for (int y = 0; y < OUTPUT_SIZE; y++) {
            by[y] -= 0.5f * dby[y] / n_valid;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                Wy[y][j] -= 0.5f * dWy[y][j] / n_valid;
        }

        bpc = (float)(total_loss / n_valid / log(2.0));
        if (verbose && (epoch < 5 || epoch % 100 == 0 || epoch == n_epochs - 1)) {
            printf("  epoch %4d: %.4f bpc\n", epoch, bpc);
            fflush(stdout);
        }
    }
    return bpc;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file>\n", argv[0]);
        return 1;
    }

    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    N = fread(data, 1, MAX_DATA, df);
    fclose(df);
    printf("Data: %d bytes\n\n", N);
    fflush(stdout);

    /* ===================================================================
     * Construction 1: Bigram (offset 1 only)
     * h[t][j] = hash(j, data[t]) for all 128 neurons
     * =================================================================== */

    printf("=== Construction 1: Bigram (offset 1) ===\n");
    fflush(stdout);

    static float h_big[MAX_DATA][HIDDEN_SIZE];
    for (int t = 0; t < N; t++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            h_big[t][j] = hash_sign(j, data[t]);

    float bpc_bigram = optimize_wy(h_big, 0, N - 1, 500, 1);
    printf("  → Bigram construction: %.4f bpc\n\n", bpc_bigram);
    fflush(stdout);

    /* ===================================================================
     * Construction 2: Skip-8 (8 offsets, each gets 16 neurons)
     * Offsets from greedy selection: [1, 8, 20, 3, 27, 2, 12, 7]
     * =================================================================== */

    printf("=== Construction 2: Skip-8 (8 greedy offsets × 16 neurons) ===\n");
    fflush(stdout);

    int skip8_offsets[] = {1, 8, 20, 3, 27, 2, 12, 7};
    int max_off_8 = 27;

    static float h_skip8[MAX_DATA][HIDDEN_SIZE];
    memset(h_skip8, 0, sizeof(h_skip8));

    for (int t = max_off_8; t < N; t++) {
        for (int g = 0; g < 8; g++) {
            int d = skip8_offsets[g];
            unsigned char byte = data[t - d];
            for (int ni = 0; ni < 16; ni++) {
                int j = g * 16 + ni;
                h_skip8[t][j] = hash_sign(j, byte);
            }
        }
    }

    float bpc_skip8 = optimize_wy(h_skip8, max_off_8, N - 1, 500, 1);
    printf("  → Skip-8 construction: %.4f bpc\n\n", bpc_skip8);
    fflush(stdout);

    /* ===================================================================
     * Construction 3: Factor-map guided (12 offset PAIRS from trained RNN)
     * Each neuron encodes hash(byte_at_d1, byte_at_d2)
     * =================================================================== */

    printf("=== Construction 3: Factor-map guided (12 offset pairs) ===\n");
    fflush(stdout);

    typedef struct { int d1; int d2; int neurons; } Pair;
    Pair pairs[] = {
        {1, 7,  52}, {1, 8,  20}, {8, 2,  18}, {1, 12,  9},
        {2, 7,   8}, {3, 12,  6}, {1, 20,  5}, {2, 12,  4},
        {1, 3,   1}, {3, 7,   1}, {1, 2,   2}, {20, 2,  2},
    };
    int n_pairs = 12;
    int max_off_fm = 20;

    static float h_fm[MAX_DATA][HIDDEN_SIZE];
    memset(h_fm, 0, sizeof(h_fm));

    int neuron_idx = 0;
    for (int p = 0; p < n_pairs; p++) {
        for (int ni = 0; ni < pairs[p].neurons; ni++) {
            int j = neuron_idx++;
            for (int t = max_off_fm; t < N; t++) {
                unsigned char b1 = data[t - pairs[p].d1];
                unsigned char b2 = data[t - pairs[p].d2];
                /* 2-byte hash for conjunction encoding */
                unsigned int seed = (unsigned int)(j * 65537 + b1 * 257 + b2);
                seed = seed * 2654435761u;
                seed = (seed >> 16) ^ seed;
                h_fm[t][j] = ((seed & 1) ? 0.95f : -0.95f);
            }
        }
    }
    printf("  Assigned %d neurons across %d pairs\n", neuron_idx, n_pairs);
    fflush(stdout);

    float bpc_fm = optimize_wy(h_fm, max_off_fm, N - 1, 500, 1);
    printf("  → Factor-map construction: %.4f bpc\n\n", bpc_fm);
    fflush(stdout);

    /* ===================================================================
     * Construction 4: Factor-map + word_len state feature
     * Add word_len as an additional dimension per neuron
     * =================================================================== */

    printf("=== Construction 4: Factor-map + word_len ===\n");
    fflush(stdout);

    static float h_fm_wl[MAX_DATA][HIDDEN_SIZE];
    memset(h_fm_wl, 0, sizeof(h_fm_wl));

    /* Compute word_len at each position */
    int word_len[MAX_DATA];
    word_len[0] = 0;
    for (int t = 1; t < N; t++) {
        if (data[t - 1] == ' ' || data[t - 1] == '\n' || data[t - 1] == '<' || data[t - 1] == '>')
            word_len[t] = 0;
        else
            word_len[t] = (word_len[t - 1] < 15) ? word_len[t - 1] + 1 : 15;
    }

    neuron_idx = 0;
    for (int p = 0; p < n_pairs; p++) {
        for (int ni = 0; ni < pairs[p].neurons; ni++) {
            int j = neuron_idx++;
            for (int t = max_off_fm; t < N; t++) {
                unsigned char b1 = data[t - pairs[p].d1];
                unsigned char b2 = data[t - pairs[p].d2];
                int wl = word_len[t];
                /* 3-feature hash: byte pair + word length */
                unsigned int seed = (unsigned int)(j * 65537 + b1 * 257 + b2 * 17 + wl);
                seed = seed * 2654435761u;
                seed = (seed >> 16) ^ seed;
                h_fm_wl[t][j] = ((seed & 1) ? 0.95f : -0.95f);
            }
        }
    }

    float bpc_fm_wl = optimize_wy(h_fm_wl, max_off_fm, N - 1, 500, 1);
    printf("  → Factor-map + word_len: %.4f bpc\n\n", bpc_fm_wl);
    fflush(stdout);

    /* ===================================================================
     * Summary
     * =================================================================== */

    printf("=== Reverse Isomorphism: Summary ===\n\n");
    printf("Construction                             bpc\n");
    printf("---------------------------------------------------\n");
    printf("Marginal (unigram)                       4.74\n");
    printf("1. Bigram hash (128 neurons, offset 1)   %.3f\n", bpc_bigram);
    printf("2. Skip-8 hash (16n × 8 offsets)         %.3f\n", bpc_skip8);
    printf("3. Factor-map pairs (12 pairs, 128n)     %.3f\n", bpc_fm);
    printf("4. Factor-map + word_len                 %.3f\n", bpc_fm_wl);
    printf("---------------------------------------------------\n");
    printf("Trained RNN (4000 epochs SGD)            0.079\n");
    printf("UM floor (skip-6, perfect)               0.000\n");
    printf("\n");
    printf("All constructions: hash embedding → optimize W_y only.\n");
    printf("No gradient-based W_x/W_h training needed.\n");
    fflush(stdout);

    return 0;
}
