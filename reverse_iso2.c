/*
 * reverse_iso2.c — Improved reverse isomorphism using UM conditional probs.
 *
 * Instead of random hash encodings, use the UM's actual learned conditional
 * distributions as neuron activations. For each offset d and neuron j:
 *   h_j[t] = f(P(y | data[t-d]))
 * where f extracts a specific feature of the conditional distribution.
 *
 * This encodes the UM's learned knowledge directly into h vectors,
 * making the readout's job trivial.
 *
 * Usage: reverse_iso2 <data_file>
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

/* Bigram conditional distributions P(y | x@d) */
static int bg_count[50][256][256];  /* [offset][input_byte][output_byte] */
static int bg_total[50][256];

/* Optimize W_y on given h vectors */
static float optimize_wy(float h[][HIDDEN_SIZE], int start, int end,
                         int n_epochs, int verbose) {
    static float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    static float by_arr[OUTPUT_SIZE];
    memset(Wy, 0, sizeof(Wy));

    int marginal[256] = {0};
    for (int t = 0; t < N; t++) marginal[data[t]]++;
    for (int y = 0; y < OUTPUT_SIZE; y++)
        by_arr[y] = logf((marginal[y] + 0.5f) / (N + 128.0f));

    int n_valid = end - start;
    float bpc = 99.0f;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        static float dWy[OUTPUT_SIZE][HIDDEN_SIZE];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));
        double total_loss = 0;

        for (int t = start; t < end; t++) {
            float* hv = h[t];
            int y_true = data[t + 1];

            float logits[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float sum = by_arr[y];
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
            by_arr[y] -= 0.5f * dby[y] / n_valid;
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

    /* Learn bigram counts at all offsets */
    memset(bg_count, 0, sizeof(bg_count));
    memset(bg_total, 0, sizeof(bg_total));
    for (int d = 1; d <= 30; d++) {
        for (int t = d; t < N - 1; t++) {
            bg_count[d - 1][data[t - d]][data[t + 1]]++;
            bg_total[d - 1][data[t - d]]++;
        }
    }

    /* Marginal distribution */
    int marginal[256] = {0};
    for (int t = 0; t < N; t++) marginal[data[t]]++;
    float marginal_p[256];
    for (int y = 0; y < 256; y++)
        marginal_p[y] = (float)(marginal[y] + 0.001f) / (N + 0.256f);

    /* Find top output bytes (most common) for neuron assignment */
    int sorted_bytes[256];
    for (int i = 0; i < 256; i++) sorted_bytes[i] = i;
    for (int i = 0; i < 255; i++)
        for (int j = i + 1; j < 256; j++)
            if (marginal[sorted_bytes[j]] > marginal[sorted_bytes[i]]) {
                int tmp = sorted_bytes[i]; sorted_bytes[i] = sorted_bytes[j]; sorted_bytes[j] = tmp;
            }

    printf("Top 10 output bytes: ");
    for (int i = 0; i < 10; i++) {
        int b = sorted_bytes[i];
        char safe = (b >= 32 && b < 127) ? b : '.';
        printf("'%c'(%d) ", safe, marginal[b]);
    }
    printf("\n\n");
    fflush(stdout);

    int skip8_offsets[] = {1, 8, 20, 3, 27, 2, 12, 7};

    /* ===================================================================
     * Construction A: Log-prob encoding
     * For each neuron j assigned to (offset g, output byte y_j):
     *   h_j[t] = log P(y_j | data[t - offset_g]) - log P(y_j)
     * This is the log-likelihood ratio — the UM's learned information.
     * =================================================================== */

    printf("=== Construction A: Log-prob encoding (8 offsets × 16 neurons) ===\n");
    fflush(stdout);

    static float h_logp[MAX_DATA][HIDDEN_SIZE];
    memset(h_logp, 0, sizeof(h_logp));

    int max_off = 27;

    /* Assign neurons: 16 per offset, each tracking a different output byte */
    for (int g = 0; g < 8; g++) {
        int d = skip8_offsets[g];
        for (int ni = 0; ni < 16; ni++) {
            int j = g * 16 + ni;
            int y_target = sorted_bytes[ni]; /* track the ni-th most common byte */

            for (int t = max_off; t < N; t++) {
                int x = data[t - d];
                float p_cond;
                if (bg_total[d - 1][x] > 0)
                    p_cond = (bg_count[d - 1][x][y_target] + 0.001f) /
                             (bg_total[d - 1][x] + 0.256f);
                else
                    p_cond = marginal_p[y_target];

                /* Log-likelihood ratio, clamped to [-3, 3] */
                float llr = logf(p_cond / marginal_p[y_target]);
                if (llr > 3.0f) llr = 3.0f;
                if (llr < -3.0f) llr = -3.0f;
                h_logp[t][j] = llr;
            }
        }
    }

    float bpc_logp = optimize_wy(h_logp, max_off, N - 1, 500, 1);
    printf("  → Log-prob encoding: %.4f bpc\n\n", bpc_logp);
    fflush(stdout);

    /* ===================================================================
     * Construction B: Full conditional vector encoding
     * For each offset, encode ALL conditional probs P(y|x@d) for the
     * input byte at that offset. Use SVD-like dimensionality reduction.
     * Simpler: just use the conditional probs for the top-16 bytes.
     * =================================================================== */

    printf("=== Construction B: Conditional prob vector (8 off × 16 neurons) ===\n");
    fflush(stdout);

    static float h_cond[MAX_DATA][HIDDEN_SIZE];
    memset(h_cond, 0, sizeof(h_cond));

    for (int g = 0; g < 8; g++) {
        int d = skip8_offsets[g];
        for (int ni = 0; ni < 16; ni++) {
            int j = g * 16 + ni;
            int y_target = sorted_bytes[ni];

            for (int t = max_off; t < N; t++) {
                int x = data[t - d];
                float p_cond;
                if (bg_total[d - 1][x] > 0)
                    p_cond = (float)(bg_count[d - 1][x][y_target] + 0.001f) /
                             (bg_total[d - 1][x] + 0.256f);
                else
                    p_cond = marginal_p[y_target];

                /* Store the conditional probability directly, centered */
                h_cond[t][j] = (p_cond - marginal_p[y_target]) * 10.0f;
            }
        }
    }

    float bpc_cond = optimize_wy(h_cond, max_off, N - 1, 500, 1);
    printf("  → Conditional prob encoding: %.4f bpc\n\n", bpc_cond);
    fflush(stdout);

    /* ===================================================================
     * Construction C: Direct count encoding (log-stochastic)
     * Each neuron stores floor(log2(count(x@d, y_j))) — the UM weight.
     * =================================================================== */

    printf("=== Construction C: Log-count encoding (8 off × 16 neurons) ===\n");
    fflush(stdout);

    static float h_logc[MAX_DATA][HIDDEN_SIZE];
    memset(h_logc, 0, sizeof(h_logc));

    for (int g = 0; g < 8; g++) {
        int d = skip8_offsets[g];
        for (int ni = 0; ni < 16; ni++) {
            int j = g * 16 + ni;
            int y_target = sorted_bytes[ni];

            for (int t = max_off; t < N; t++) {
                int x = data[t - d];
                int count = bg_count[d - 1][x][y_target];
                /* Log-stochastic: floor(log2(count)) */
                float ls = 0;
                if (count > 0) {
                    int c = count;
                    while (c > 1) { c >>= 1; ls += 1.0f; }
                }
                h_logc[t][j] = ls - 3.0f; /* center around typical value */
            }
        }
    }

    float bpc_logc = optimize_wy(h_logc, max_off, N - 1, 500, 1);
    printf("  → Log-count encoding: %.4f bpc\n\n", bpc_logc);
    fflush(stdout);

    /* ===================================================================
     * Construction D: Mixed encoding (top-k probs as direct features)
     * For each offset d, store the conditional probs P(y|x@d) for
     * the most common output bytes as direct neuron activations.
     * This is like the UM's learned distribution projected onto neurons.
     * =================================================================== */

    printf("=== Construction D: Top-8 probs × 4 offsets (32 features) ===\n");
    printf("  Plus log-prob for remaining 96 neurons\n");
    fflush(stdout);

    static float h_mix[MAX_DATA][HIDDEN_SIZE];
    memset(h_mix, 0, sizeof(h_mix));

    /* First 32 neurons: direct probs for top-8 bytes at offsets 1,8,2,7 */
    int key_offsets[] = {1, 8, 2, 7};
    for (int g = 0; g < 4; g++) {
        int d = key_offsets[g];
        for (int ni = 0; ni < 8; ni++) {
            int j = g * 8 + ni;
            int y_target = sorted_bytes[ni];

            for (int t = max_off; t < N; t++) {
                int x = data[t - d];
                float p_cond;
                if (bg_total[d - 1][x] > 0)
                    p_cond = (float)(bg_count[d - 1][x][y_target] + 0.001f) /
                             (bg_total[d - 1][x] + 0.256f);
                else
                    p_cond = marginal_p[y_target];
                h_mix[t][j] = (p_cond - marginal_p[y_target]) * 10.0f;
            }
        }
    }

    /* Remaining 96 neurons: log-prob encoding at all 8 offsets × 12 neurons */
    for (int g = 0; g < 8; g++) {
        int d = skip8_offsets[g];
        for (int ni = 0; ni < 12; ni++) {
            int j = 32 + g * 12 + ni;
            int y_target = sorted_bytes[ni];

            for (int t = max_off; t < N; t++) {
                int x = data[t - d];
                float p_cond;
                if (bg_total[d - 1][x] > 0)
                    p_cond = (bg_count[d - 1][x][y_target] + 0.001f) /
                             (bg_total[d - 1][x] + 0.256f);
                else
                    p_cond = marginal_p[y_target];
                float llr = logf(p_cond / marginal_p[y_target]);
                if (llr > 3.0f) llr = 3.0f;
                if (llr < -3.0f) llr = -3.0f;
                h_mix[t][j] = llr;
            }
        }
    }

    float bpc_mix = optimize_wy(h_mix, max_off, N - 1, 500, 1);
    printf("  → Mixed encoding: %.4f bpc\n\n", bpc_mix);
    fflush(stdout);

    /* ===================================================================
     * Construction E: ORACLE — use the actual P(y|context) as features
     * For each position, h[t] = the UM's predicted distribution.
     * This is the best possible with UM knowledge, showing the ceiling.
     * =================================================================== */

    printf("=== Construction E: Oracle (UM predicted probs as features) ===\n");
    fflush(stdout);

    /* At each position, for each of 128 output bytes, store P(y|x@1) */
    static float h_oracle[MAX_DATA][HIDDEN_SIZE];
    memset(h_oracle, 0, sizeof(h_oracle));

    for (int t = 0; t < N; t++) {
        int x = data[t];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            int y_target = sorted_bytes[j];
            float p_cond;
            if (bg_total[0][x] > 0)
                p_cond = (float)(bg_count[0][x][y_target] + 0.001f) /
                         (bg_total[0][x] + 0.256f);
            else
                p_cond = marginal_p[y_target];
            /* Log-odds encoding */
            float lo = logf(p_cond / (1.0f - p_cond + 1e-10f));
            if (lo > 5.0f) lo = 5.0f;
            if (lo < -5.0f) lo = -5.0f;
            h_oracle[t][j] = lo;
        }
    }

    float bpc_oracle = optimize_wy(h_oracle, 0, N - 1, 500, 1);
    printf("  → Oracle (bigram probs): %.4f bpc\n\n", bpc_oracle);
    fflush(stdout);

    /* ===================================================================
     * Summary
     * =================================================================== */

    printf("=== Reverse Isomorphism v2: Summary ===\n\n");
    printf("Construction                             bpc     Encoding\n");
    printf("---------------------------------------------------------------\n");
    printf("Marginal (unigram)                       4.74    none\n");
    printf("A. Log-prob ratio (8off × 16n)           %.3f   log P(y|x@d)/P(y)\n", bpc_logp);
    printf("B. Cond. prob centered (8off × 16n)      %.3f   P(y|x@d) - P(y)\n", bpc_cond);
    printf("C. Log-count (8off × 16n)                %.3f   floor(log2(cnt))\n", bpc_logc);
    printf("D. Mixed (32 probs + 96 log-prob)        %.3f   hybrid\n", bpc_mix);
    printf("E. Oracle (128 bigram probs)             %.3f   log-odds P(y|x@1)\n", bpc_oracle);
    printf("---------------------------------------------------------------\n");
    printf("Random hash (skip-8, from v1)            0.911   hash sign\n");
    printf("Trained RNN (4000 epochs SGD)            0.079   learned\n");
    printf("UM floor (skip-6, perfect)               0.000   counting\n");
    fflush(stdout);

    return 0;
}
