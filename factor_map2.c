/*
 * factor_map2.c — Improved factor map with continuous-value prediction.
 *
 * The first factor_map.c showed that all 128 neurons are explained as
 * 2-offset conjunctions (90-97% sign accuracy), but the binary sign
 * prediction failed for verification (5.0 bpc vs 0.08 actual).
 *
 * This version uses conditional means instead of binary signs:
 *   h_predicted[j][t] = E[h_j | data[t-d1], data[t-d2]]
 * where (d1, d2) is the neuron's best conjunction pair.
 *
 * This preserves the continuous magnitudes that carry most of the
 * predictive information (since mean |h| = 0.655, not deeply saturated).
 *
 * Usage: factor_map2 <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 1100
#define N_OFFSETS 8

typedef struct {
    float Wx[HIDDEN_SIZE][INPUT_SIZE];
    float Wh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bh[HIDDEN_SIZE];
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
    float h[HIDDEN_SIZE];
} RNN;

void load_model(RNN* rnn, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("load_model"); exit(1); }
    fread(rnn->Wx, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
    fread(rnn->Wh, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    fread(rnn->bh, sizeof(float), HIDDEN_SIZE, f);
    fread(rnn->Wy, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(rnn->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

void rnn_step(RNN* rnn, unsigned char x_t) {
    float h_new[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = rnn->bh[i] + rnn->Wx[i][x_t];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += rnn->Wh[i][j] * rnn->h[j];
        h_new[i] = tanhf(sum);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++)
        rnn->h[i] = h_new[i];
}

void softmax(float* logits, float* probs, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - maxv);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

static int OFFSETS[N_OFFSETS] = {1, 8, 20, 3, 27, 2, 12, 7};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data_file> <model_file>\n", argv[0]);
        return 1;
    }

    /* Load data */
    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[MAX_DATA];
    int N = fread(data, 1, MAX_DATA, df);
    fclose(df);
    printf("Data: %d bytes\n\n", N);

    /* Load model */
    RNN rnn;
    load_model(&rnn, argv[2]);
    memset(rnn.h, 0, sizeof(rnn.h));

    /* Run forward pass */
    float h_states[MAX_DATA][HIDDEN_SIZE];
    for (int t = 0; t < N; t++) {
        rnn_step(&rnn, data[t]);
        memcpy(h_states[t], rnn.h, sizeof(rnn.h));
    }

    int max_off = 0;
    for (int d = 0; d < N_OFFSETS; d++)
        if (OFFSETS[d] > max_off) max_off = OFFSETS[d];
    int T_start = max_off;
    int T_count = N - T_start;

    /* ===================================================================
     * Find best conjunction (d1, d2) per neuron — same as factor_map.c
     * but using R² (variance explained) instead of MI for continuous h
     * =================================================================== */

    printf("=== Finding best 2-offset predictors per neuron ===\n\n");

    /* Per neuron: best pair of offsets for predicting h_j (continuous) */
    int best_d1[HIDDEN_SIZE], best_d2[HIDDEN_SIZE];
    double best_r2[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        /* Total variance of h_j */
        double mean_h = 0;
        for (int t = T_start; t < N; t++) mean_h += h_states[t][j];
        mean_h /= T_count;
        double var_total = 0;
        for (int t = T_start; t < N; t++) {
            double d = h_states[t][j] - mean_h;
            var_total += d * d;
        }

        double best = -1;
        int bd1 = 0, bd2 = 1;

        for (int d1i = 0; d1i < N_OFFSETS; d1i++) {
            for (int d2i = d1i + 1; d2i < N_OFFSETS; d2i++) {
                int d1 = OFFSETS[d1i], d2 = OFFSETS[d2i];

                /* Compute conditional means E[h_j | data[t-d1], data[t-d2]] */
                /* Using a 256x256 table — sum and count per cell */
                double sum_table[256][256];
                int count_table[256][256];
                memset(sum_table, 0, sizeof(sum_table));
                memset(count_table, 0, sizeof(count_table));

                for (int t = T_start; t < N; t++) {
                    int b1 = data[t - d1];
                    int b2 = data[t - d2];
                    sum_table[b1][b2] += h_states[t][j];
                    count_table[b1][b2]++;
                }

                /* Compute residual variance (h_j - conditional_mean)^2 */
                double var_residual = 0;
                for (int t = T_start; t < N; t++) {
                    int b1 = data[t - d1];
                    int b2 = data[t - d2];
                    double cond_mean = sum_table[b1][b2] / count_table[b1][b2];
                    double residual = h_states[t][j] - cond_mean;
                    var_residual += residual * residual;
                }

                double r2 = 1.0 - var_residual / var_total;
                if (r2 > best) {
                    best = r2;
                    bd1 = d1i;
                    bd2 = d2i;
                }
            }
        }

        best_d1[j] = bd1;
        best_d2[j] = bd2;
        best_r2[j] = best;
    }

    /* Also find best single-offset R² for comparison */
    double best_r2_single[HIDDEN_SIZE];
    int best_d_single[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double mean_h = 0;
        for (int t = T_start; t < N; t++) mean_h += h_states[t][j];
        mean_h /= T_count;
        double var_total = 0;
        for (int t = T_start; t < N; t++) {
            double d = h_states[t][j] - mean_h;
            var_total += d * d;
        }

        double best = -1;
        int bd = 0;
        for (int di = 0; di < N_OFFSETS; di++) {
            int d = OFFSETS[di];
            double sum_tab[256] = {0};
            int count_tab[256] = {0};
            for (int t = T_start; t < N; t++) {
                int b = data[t - d];
                sum_tab[b] += h_states[t][j];
                count_tab[b]++;
            }
            double var_res = 0;
            for (int t = T_start; t < N; t++) {
                int b = data[t - d];
                double cm = sum_tab[b] / count_tab[b];
                double r = h_states[t][j] - cm;
                var_res += r * r;
            }
            double r2 = 1.0 - var_res / var_total;
            if (r2 > best) { best = r2; bd = di; }
        }
        best_r2_single[j] = best;
        best_d_single[j] = bd;
    }

    /* Print sorted by R² */
    int order[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) order[j] = j;
    for (int i = 0; i < HIDDEN_SIZE - 1; i++)
        for (int j = i + 1; j < HIDDEN_SIZE; j++)
            if (best_r2[order[j]] > best_r2[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }

    printf("Top 30 neurons by R² (variance explained by 2-offset conjunction):\n");
    printf("%-6s  single(off, R²)      conj(off1,off2, R²)     gain\n", "h_j");
    for (int r = 0; r < 30; r++) {
        int j = order[r];
        printf("h%-5d  off=%-3d R²=%.3f    off=(%d,%d) R²=%.3f    +%.3f\n",
               j, OFFSETS[best_d_single[j]], best_r2_single[j],
               OFFSETS[best_d1[j]], OFFSETS[best_d2[j]], best_r2[j],
               best_r2[j] - best_r2_single[j]);
    }

    /* Summary stats */
    double mean_r2_s = 0, mean_r2_c = 0;
    int n_above90 = 0, n_above80 = 0, n_above70 = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        mean_r2_s += best_r2_single[j];
        mean_r2_c += best_r2[j];
        if (best_r2[j] >= 0.90) n_above90++;
        if (best_r2[j] >= 0.80) n_above80++;
        if (best_r2[j] >= 0.70) n_above70++;
    }
    printf("\nR² summary:\n");
    printf("  Mean single-offset R²:  %.3f\n", mean_r2_s / HIDDEN_SIZE);
    printf("  Mean 2-offset R²:       %.3f\n", mean_r2_c / HIDDEN_SIZE);
    printf("  Neurons with R² >= 0.90: %d\n", n_above90);
    printf("  Neurons with R² >= 0.80: %d\n", n_above80);
    printf("  Neurons with R² >= 0.70: %d\n", n_above70);

    /* ===================================================================
     * Verification: predict h from conditional means, compute bpc
     * =================================================================== */

    printf("\n=== Verification: BPC from Conditional Mean Predictions ===\n\n");

    /* Build conditional mean tables for each neuron using its best pair */
    /* cond_mean_table[j][b1][b2] = mean h_j when data[t-d1]=b1, data[t-d2]=b2 */
    /* Too large for stack (128 * 256 * 256 * 4 = 4GB!) — use per-neuron lookup */

    /* For verification, compute bpc at several levels:
     * 1. Actual RNN
     * 2. Single-offset conditional mean
     * 3. 2-offset conditional mean
     * 4. 3-offset conditional mean (greedy extend)
     */

    double bpc_actual = 0, bpc_single = 0, bpc_conj = 0;
    int bpc_count = 0;

    for (int t = T_start; t < N - 1; t++) {
        int y = data[t + 1];

        /* Actual RNN */
        {
            float logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                logits[o] = rnn.by[o];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    logits[o] += rnn.Wy[o][j] * h_states[t][j];
            }
            float probs[OUTPUT_SIZE];
            softmax(logits, probs, OUTPUT_SIZE);
            bpc_actual -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        /* Single-offset conditional mean */
        {
            float h_pred[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                int d = OFFSETS[best_d_single[j]];
                int b = data[t - d];
                /* Need conditional mean — recompute on the fly (slow but correct) */
                double sum = 0; int cnt = 0;
                for (int s = T_start; s < N; s++) {
                    if (data[s - d] == b) {
                        sum += h_states[s][j];
                        cnt++;
                    }
                }
                h_pred[j] = (cnt > 0) ? sum / cnt : 0;
            }
            float logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                logits[o] = rnn.by[o];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    logits[o] += rnn.Wy[o][j] * h_pred[j];
            }
            float probs[OUTPUT_SIZE];
            softmax(logits, probs, OUTPUT_SIZE);
            bpc_single -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        /* 2-offset conditional mean */
        {
            float h_pred[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                int d1 = OFFSETS[best_d1[j]];
                int d2 = OFFSETS[best_d2[j]];
                int b1 = data[t - d1];
                int b2 = data[t - d2];
                double sum = 0; int cnt = 0;
                for (int s = T_start; s < N; s++) {
                    if (data[s - d1] == b1 && data[s - d2] == b2) {
                        sum += h_states[s][j];
                        cnt++;
                    }
                }
                h_pred[j] = (cnt > 0) ? sum / cnt : 0;
            }
            float logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                logits[o] = rnn.by[o];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    logits[o] += rnn.Wy[o][j] * h_pred[j];
            }
            float probs[OUTPUT_SIZE];
            softmax(logits, probs, OUTPUT_SIZE);
            bpc_conj -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        bpc_count++;
        if (bpc_count % 100 == 0)
            fprintf(stderr, "  position %d/%d\r", bpc_count, T_count - 1);
    }

    bpc_actual /= bpc_count;
    bpc_single /= bpc_count;
    bpc_conj /= bpc_count;

    printf("Positions evaluated: %d\n", bpc_count);
    printf("Actual RNN:                    %.4f bpc\n", bpc_actual);
    printf("Single-offset cond. mean:      %.4f bpc\n", bpc_single);
    printf("2-offset conjunction cond. mean: %.4f bpc\n", bpc_conj);
    printf("Marginal (no context):         ~4.74 bpc\n");
    printf("UM floor (skip-8):             0.043 bpc\n\n");

    double gain_actual = 4.74 - bpc_actual;
    double gain_single = 4.74 - bpc_single;
    double gain_conj   = 4.74 - bpc_conj;
    printf("BPC gain captured (single): %.3f / %.3f = %.1f%%\n",
           gain_single, gain_actual, 100.0 * gain_single / gain_actual);
    printf("BPC gain captured (conj):   %.3f / %.3f = %.1f%%\n",
           gain_conj, gain_actual, 100.0 * gain_conj / gain_actual);

    /* ===================================================================
     * Per-offset pair distribution
     * =================================================================== */

    printf("\n=== Offset Pair Distribution ===\n\n");
    printf("Pair     neurons  mean_R²\n");

    for (int d1i = 0; d1i < N_OFFSETS; d1i++) {
        for (int d2i = d1i + 1; d2i < N_OFFSETS; d2i++) {
            int cnt = 0;
            double sum_r2 = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                if (best_d1[j] == d1i && best_d2[j] == d2i) {
                    cnt++;
                    sum_r2 += best_r2[j];
                }
            }
            if (cnt > 0)
                printf("(%d,%d)%*s  %3d      %.3f\n",
                       OFFSETS[d1i], OFFSETS[d2i],
                       8 - (int)log10(OFFSETS[d1i]+1) - (int)log10(OFFSETS[d2i]+1), "",
                       cnt, sum_r2 / cnt);
        }
    }

    return 0;
}
