/*
 * factor_map4.c — Combined factor map: fixed offsets + state features.
 *
 * Combines the insights from factor_map2 (2-offset conditional means,
 * R²=0.837, 0.677 bpc) and factor_map3 (word_len is universally best
 * state feature).
 *
 * Tests progressively richer predictors:
 *   Level 1: data[t-1] + data[t-d2] (2-offset, reproduces factor_map2)
 *   Level 2: data[t-1] + data[t-d2] + word_len (add timing state)
 *   Level 3: data[t-1] + word_len + in_tag (fewer byte features, more state)
 *   Level 4: data[t-1] + data[t-d2] + word_len + in_tag
 *
 * Usage: factor_map4 <data_file> <model_file>
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

    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[MAX_DATA];
    int N = fread(data, 1, MAX_DATA, df);
    fclose(df);
    printf("Data: %d bytes\n\n", N);

    RNN rnn;
    load_model(&rnn, argv[2]);
    memset(rnn.h, 0, sizeof(rnn.h));

    float h_states[MAX_DATA][HIDDEN_SIZE];
    for (int t = 0; t < N; t++) {
        rnn_step(&rnn, data[t]);
        memcpy(h_states[t], rnn.h, sizeof(rnn.h));
    }

    /* Compute state features */
    int word_len[MAX_DATA];
    int in_tag[MAX_DATA];
    {
        int wl = 0, it = 0;
        for (int t = 0; t < N; t++) {
            if (data[t] == '<') it = 1;
            if (data[t] == '>') it = 0;
            if (data[t] == ' ' || data[t] == '\n' || data[t] == '\t') wl = 0;
            else wl++;
            word_len[t] = (wl > 15) ? 15 : wl;
            in_tag[t] = it;
        }
    }

    int max_off = 0;
    for (int d = 0; d < N_OFFSETS; d++)
        if (OFFSETS[d] > max_off) max_off = OFFSETS[d];
    int T_start = max_off;
    int T_count = N - T_start;

    /* Find best second offset for each neuron (from factor_map2 logic) */
    int best_d2_idx[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double mean_h = 0;
        for (int t = T_start; t < N; t++) mean_h += h_states[t][j];
        mean_h /= T_count;
        double var_total = 0;
        for (int t = T_start; t < N; t++) {
            double d = h_states[t][j] - mean_h;
            var_total += d * d;
        }

        double best_r2 = -1;
        int best_di = 1;  /* default to offset index 1 (offset 8) */
        for (int di = 1; di < N_OFFSETS; di++) {  /* skip offset 1 since we always use it */
            int d2 = OFFSETS[di];
            double sum_tab[256][256];
            int count_tab[256][256];
            memset(sum_tab, 0, sizeof(sum_tab));
            memset(count_tab, 0, sizeof(count_tab));
            for (int t = T_start; t < N; t++) {
                sum_tab[data[t-1]][data[t-d2]] += h_states[t][j];
                count_tab[data[t-1]][data[t-d2]]++;
            }
            double var_res = 0;
            for (int t = T_start; t < N; t++) {
                double cm = (count_tab[data[t-1]][data[t-d2]] > 0) ?
                    sum_tab[data[t-1]][data[t-d2]] / count_tab[data[t-1]][data[t-d2]] : 0;
                double r = h_states[t][j] - cm;
                var_res += r * r;
            }
            double r2 = 1.0 - var_res / var_total;
            if (r2 > best_r2) { best_r2 = r2; best_di = di; }
        }
        best_d2_idx[j] = best_di;
    }

    /* ===================================================================
     * Verification BPC at multiple levels
     * =================================================================== */

    printf("=== BPC Verification at Multiple Context Levels ===\n\n");

    double bpc_actual = 0;
    double bpc_2off = 0;     /* data[t-1] + data[t-d2] */
    double bpc_2off_wl = 0;  /* data[t-1] + data[t-d2] + word_len */
    double bpc_2off_wl_tag = 0; /* data[t-1] + data[t-d2] + word_len + in_tag */
    int bpc_count = 0;

    for (int t = T_start; t < N - 1; t++) {
        int y = data[t + 1];

        /* Actual RNN */
        {
            float logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                logits[o] = rnn.by[o];
                for (int jj = 0; jj < HIDDEN_SIZE; jj++)
                    logits[o] += rnn.Wy[o][jj] * h_states[t][jj];
            }
            float probs[OUTPUT_SIZE];
            softmax(logits, probs, OUTPUT_SIZE);
            bpc_actual -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        /* Level 1: data[t-1] + data[t-d2] */
        {
            float h_pred[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                int d2 = OFFSETS[best_d2_idx[j]];
                int b1 = data[t - 1], b2 = data[t - d2];
                double sum = 0; int cnt = 0;
                for (int s = T_start; s < N; s++) {
                    if (data[s-1] == b1 && data[s - d2] == b2) {
                        sum += h_states[s][j]; cnt++;
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
            bpc_2off -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        /* Level 2: data[t-1] + data[t-d2] + word_len */
        {
            float h_pred[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                int d2 = OFFSETS[best_d2_idx[j]];
                int b1 = data[t - 1], b2 = data[t - d2];
                int wl = word_len[t];
                double sum = 0; int cnt = 0;
                for (int s = T_start; s < N; s++) {
                    if (data[s-1] == b1 && data[s - d2] == b2 && word_len[s] == wl) {
                        sum += h_states[s][j]; cnt++;
                    }
                }
                /* Fall back to 2-offset if no match with word_len */
                if (cnt == 0) {
                    for (int s = T_start; s < N; s++) {
                        if (data[s-1] == b1 && data[s - d2] == b2) {
                            sum += h_states[s][j]; cnt++;
                        }
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
            bpc_2off_wl -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        /* Level 3: data[t-1] + data[t-d2] + word_len + in_tag */
        {
            float h_pred[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                int d2 = OFFSETS[best_d2_idx[j]];
                int b1 = data[t - 1], b2 = data[t - d2];
                int wl = word_len[t], it = in_tag[t];
                double sum = 0; int cnt = 0;
                for (int s = T_start; s < N; s++) {
                    if (data[s-1] == b1 && data[s - d2] == b2
                        && word_len[s] == wl && in_tag[s] == it) {
                        sum += h_states[s][j]; cnt++;
                    }
                }
                /* Fall back to 2-offset + word_len */
                if (cnt == 0) {
                    for (int s = T_start; s < N; s++) {
                        if (data[s-1] == b1 && data[s - d2] == b2 && word_len[s] == wl) {
                            sum += h_states[s][j]; cnt++;
                        }
                    }
                }
                /* Fall back to 2-offset only */
                if (cnt == 0) {
                    for (int s = T_start; s < N; s++) {
                        if (data[s-1] == b1 && data[s - d2] == b2) {
                            sum += h_states[s][j]; cnt++;
                        }
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
            bpc_2off_wl_tag -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        bpc_count++;
        if (bpc_count % 50 == 0)
            fprintf(stderr, "  position %d/%d\r", bpc_count, T_count - 1);
    }

    bpc_actual /= bpc_count;
    bpc_2off /= bpc_count;
    bpc_2off_wl /= bpc_count;
    bpc_2off_wl_tag /= bpc_count;

    double gain_actual = 4.74 - bpc_actual;

    printf("Positions: %d\n\n", bpc_count);
    printf("%-45s  bpc      gain%%\n", "Method");
    printf("%-45s  %.4f   -\n", "Marginal (no context)", 4.74);
    printf("%-45s  %.4f   %.1f%%\n", "data[t-1] + data[t-d2]",
           bpc_2off, 100.0 * (4.74 - bpc_2off) / gain_actual);
    printf("%-45s  %.4f   %.1f%%\n", "data[t-1] + data[t-d2] + word_len",
           bpc_2off_wl, 100.0 * (4.74 - bpc_2off_wl) / gain_actual);
    printf("%-45s  %.4f   %.1f%%\n", "data[t-1] + data[t-d2] + word_len + in_tag",
           bpc_2off_wl_tag, 100.0 * (4.74 - bpc_2off_wl_tag) / gain_actual);
    printf("%-45s  %.4f   100.0%%\n", "Actual RNN", bpc_actual);
    printf("%-45s  0.0430   -\n", "UM floor (skip-8)");

    return 0;
}
