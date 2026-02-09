/*
 * factor_map3.c — State-based factor map: extend beyond fixed offsets.
 *
 * Instead of h_j ≈ f(data[t-d1], data[t-d2]), use state features:
 *   - data[t-1] (immediate input, always available via Wx)
 *   - Character class of recent history (alpha, digit, space, punct, XML)
 *   - "in word" / "between words" (flip-flop on space)
 *   - "in XML tag" (flip-flop on < and >)
 *   - Word length (how many non-space chars since last space)
 *   - Recent character class histogram
 *
 * These state features represent what the RNN hidden layer CAN encode
 * via its recurrence — not fixed offsets, but accumulated state.
 *
 * We then measure R² and bpc for predicting h_j from:
 *   Level 1: data[t-1] alone (Wx-driven)
 *   Level 2: data[t-1] + one state feature
 *   Level 3: data[t-1] + best pair of state features
 *
 * Usage: factor_map3 <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 1100

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

/* State features — things the RNN hidden layer could plausibly track */

/* Character class: 0=space/ctrl, 1=alpha, 2=digit, 3=punct, 4=xml(<>), 5=other */
int char_class(unsigned char c) {
    if (c <= 32) return 0;  /* space, newline, ctrl */
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
    if (c >= '0' && c <= '9') return 2;
    if (c == '<' || c == '>') return 4;
    if (c == '.' || c == ',' || c == ';' || c == ':' || c == '!'
        || c == '?' || c == '\'' || c == '"' || c == '-' || c == '/'
        || c == '=' || c == '&') return 3;
    return 5;
}

#define N_STATE_FEATURES 8

/* Compute state features for each position */
void compute_state_features(unsigned char *data, int N, int features[MAX_DATA][N_STATE_FEATURES]) {
    int in_tag = 0;       /* 1 if inside XML tag (between < and >) */
    int word_len = 0;     /* chars since last space */
    int tag_depth = 0;    /* nesting depth of XML tags */

    for (int t = 0; t < N; t++) {
        unsigned char c = data[t];

        /* Update state */
        if (c == '<') { in_tag = 1; tag_depth++; }
        if (c == '>') { in_tag = 0; }
        if (c == ' ' || c == '\n' || c == '\t') { word_len = 0; }
        else { word_len++; }

        /* Feature 0: char class of data[t] (0-5) */
        features[t][0] = char_class(c);

        /* Feature 1: in_tag (binary) */
        features[t][1] = in_tag;

        /* Feature 2: word_len capped at 15 (0-15) */
        features[t][2] = (word_len > 15) ? 15 : word_len;

        /* Feature 3: char class of data[t-1] (if available) */
        features[t][3] = (t > 0) ? char_class(data[t-1]) : 0;

        /* Feature 4: "just saw space" = (data[t-1] was space-like) */
        features[t][4] = (t > 0 && (data[t-1] == ' ' || data[t-1] == '\n')) ? 1 : 0;

        /* Feature 5: tag_depth capped at 7 */
        features[t][5] = (tag_depth > 7) ? 7 : tag_depth;

        /* Feature 6: last non-space char class */
        {
            int lns = 0;
            for (int s = t - 1; s >= 0; s--) {
                if (data[s] != ' ' && data[s] != '\n') {
                    lns = char_class(data[s]);
                    break;
                }
            }
            features[t][6] = lns;
        }

        /* Feature 7: position mod 8 (crude timing) */
        features[t][7] = t % 8;
    }
}

const char *feature_names[N_STATE_FEATURES] = {
    "char_class", "in_tag", "word_len", "prev_class",
    "after_space", "tag_depth", "last_nonsp_class", "pos_mod8"
};

int feature_max[N_STATE_FEATURES] = {
    6, 2, 16, 6, 2, 8, 6, 8
};

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

    /* Compute state features */
    int features[MAX_DATA][N_STATE_FEATURES];
    compute_state_features(data, N, features);

    int T_start = 1;  /* need at least 1 step for prev_class */
    int T_count = N - T_start;

    /* ===================================================================
     * 1. Single-feature R² for each neuron
     * =================================================================== */

    printf("=== State Feature R² for Each Neuron ===\n\n");

    double r2_byte[HIDDEN_SIZE];      /* R² from data[t] alone (256 classes) */
    double r2_feat[HIDDEN_SIZE][N_STATE_FEATURES];  /* R² from each state feature */

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        /* Total variance */
        double mean_h = 0;
        for (int t = T_start; t < N; t++) mean_h += h_states[t][j];
        mean_h /= T_count;
        double var_total = 0;
        for (int t = T_start; t < N; t++) {
            double d = h_states[t][j] - mean_h;
            var_total += d * d;
        }

        /* R² from data[t] */
        {
            double sum_tab[256] = {0};
            int count_tab[256] = {0};
            for (int t = T_start; t < N; t++) {
                sum_tab[data[t]] += h_states[t][j];
                count_tab[data[t]]++;
            }
            double var_res = 0;
            for (int t = T_start; t < N; t++) {
                double cm = sum_tab[data[t]] / count_tab[data[t]];
                double r = h_states[t][j] - cm;
                var_res += r * r;
            }
            r2_byte[j] = 1.0 - var_res / var_total;
        }

        /* R² from each state feature */
        for (int fi = 0; fi < N_STATE_FEATURES; fi++) {
            int max_val = feature_max[fi];
            double sum_tab[16] = {0};
            int count_tab[16] = {0};
            for (int t = T_start; t < N; t++) {
                int fv = features[t][fi];
                sum_tab[fv] += h_states[t][j];
                count_tab[fv]++;
            }
            double var_res = 0;
            for (int t = T_start; t < N; t++) {
                int fv = features[t][fi];
                double cm = (count_tab[fv] > 0) ? sum_tab[fv] / count_tab[fv] : 0;
                double r = h_states[t][j] - cm;
                var_res += r * r;
            }
            r2_feat[j][fi] = 1.0 - var_res / var_total;
            (void)max_val;
        }
    }

    /* Summary: average R² per feature across all neurons */
    printf("Average R² per feature (across 128 neurons):\n");
    printf("  %-20s  R²=%.4f\n", "data[t] (256 vals)", 0.0);
    {
        double sum = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) sum += r2_byte[j];
        printf("  %-20s  R²=%.4f\n", "data[t] (byte)", sum / HIDDEN_SIZE);
    }
    for (int fi = 0; fi < N_STATE_FEATURES; fi++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) sum += r2_feat[j][fi];
        printf("  %-20s  R²=%.4f\n", feature_names[fi], sum / HIDDEN_SIZE);
    }

    /* ===================================================================
     * 2. data[t] + one state feature: R² for each neuron
     *
     * For each neuron, find which state feature adds the most R² on top
     * of data[t] alone.
     * =================================================================== */

    printf("\n=== data[t] + Best State Feature ===\n\n");

    double r2_byte_plus_feat[HIDDEN_SIZE][N_STATE_FEATURES];
    int best_feat[HIDDEN_SIZE];
    double best_r2_combined[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double mean_h = 0;
        for (int t = T_start; t < N; t++) mean_h += h_states[t][j];
        mean_h /= T_count;
        double var_total = 0;
        for (int t = T_start; t < N; t++) {
            double d = h_states[t][j] - mean_h;
            var_total += d * d;
        }

        best_feat[j] = 0;
        best_r2_combined[j] = 0;

        for (int fi = 0; fi < N_STATE_FEATURES; fi++) {
            /* Joint context: (data[t], feature[fi][t]) */
            /* Use hash: byte * max_feature_val + feature_val */
            int max_fv = feature_max[fi];
            int table_size = 256 * max_fv;

            /* Dynamic allocation for tables */
            double *sum_tab = calloc(table_size, sizeof(double));
            int *count_tab = calloc(table_size, sizeof(int));

            for (int t = T_start; t < N; t++) {
                int key = data[t] * max_fv + features[t][fi];
                sum_tab[key] += h_states[t][j];
                count_tab[key]++;
            }

            double var_res = 0;
            for (int t = T_start; t < N; t++) {
                int key = data[t] * max_fv + features[t][fi];
                double cm = (count_tab[key] > 0) ? sum_tab[key] / count_tab[key] : 0;
                double r = h_states[t][j] - cm;
                var_res += r * r;
            }

            r2_byte_plus_feat[j][fi] = 1.0 - var_res / var_total;
            if (r2_byte_plus_feat[j][fi] > best_r2_combined[j]) {
                best_r2_combined[j] = r2_byte_plus_feat[j][fi];
                best_feat[j] = fi;
            }

            free(sum_tab);
            free(count_tab);
        }
    }

    /* Print top 30 neurons sorted by best combined R² */
    int order[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) order[j] = j;
    for (int i = 0; i < HIDDEN_SIZE - 1; i++)
        for (int j = i + 1; j < HIDDEN_SIZE; j++)
            if (best_r2_combined[order[j]] > best_r2_combined[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }

    printf("Top 30 neurons by data[t] + best state feature R²:\n");
    printf("%-6s  byte_R²  +feature         combined_R²  gain\n", "h_j");
    for (int r = 0; r < 30; r++) {
        int j = order[r];
        printf("h%-5d  %.3f    +%-15s  %.3f        +%.3f\n",
               j, r2_byte[j], feature_names[best_feat[j]],
               best_r2_combined[j], best_r2_combined[j] - r2_byte[j]);
    }

    /* Summary */
    double mean_byte = 0, mean_combined = 0;
    int feat_count[N_STATE_FEATURES] = {0};
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        mean_byte += r2_byte[j];
        mean_combined += best_r2_combined[j];
        feat_count[best_feat[j]]++;
    }
    printf("\nSummary:\n");
    printf("  Mean R² (data[t] alone): %.4f\n", mean_byte / HIDDEN_SIZE);
    printf("  Mean R² (data[t] + best feature): %.4f\n", mean_combined / HIDDEN_SIZE);
    printf("  Mean gain from state feature: +%.4f\n",
           (mean_combined - mean_byte) / HIDDEN_SIZE);
    printf("\nBest state feature distribution:\n");
    for (int fi = 0; fi < N_STATE_FEATURES; fi++)
        if (feat_count[fi] > 0)
            printf("  %-20s  %d neurons\n", feature_names[fi], feat_count[fi]);

    /* ===================================================================
     * 3. Compare with fixed 2-offset R²
     *
     * factor_map2 showed mean R²=0.837 for best 2-offset pair.
     * How does data[t] + state feature compare?
     * =================================================================== */

    printf("\n=== Comparison with Fixed 2-Offset Model ===\n\n");

    /* Also compute data[t-1]+data[t] for comparison (2 recent bytes) */
    double mean_r2_byte_pair = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double mean_h = 0;
        for (int t = T_start; t < N; t++) mean_h += h_states[t][j];
        mean_h /= T_count;
        double var_total = 0;
        for (int t = T_start; t < N; t++) {
            double d = h_states[t][j] - mean_h;
            var_total += d * d;
        }

        /* Joint (data[t-1], data[t]) */
        double sum_tab[256][256];
        int count_tab[256][256];
        memset(sum_tab, 0, sizeof(sum_tab));
        memset(count_tab, 0, sizeof(count_tab));
        for (int t = T_start; t < N; t++) {
            sum_tab[data[t-1]][data[t]] += h_states[t][j];
            count_tab[data[t-1]][data[t]]++;
        }
        double var_res = 0;
        for (int t = T_start; t < N; t++) {
            double cm = sum_tab[data[t-1]][data[t]] / count_tab[data[t-1]][data[t]];
            double r = h_states[t][j] - cm;
            var_res += r * r;
        }
        double r2_pair = 1.0 - var_res / var_total;
        mean_r2_byte_pair += r2_pair;
    }
    mean_r2_byte_pair /= HIDDEN_SIZE;

    printf("Comparison of R² approaches (mean across 128 neurons):\n");
    printf("  data[t] alone:               %.4f\n", mean_byte / HIDDEN_SIZE);
    printf("  data[t] + best state feat:    %.4f\n", mean_combined / HIDDEN_SIZE);
    printf("  data[t-1] + data[t] (bigram): %.4f\n", mean_r2_byte_pair);
    printf("  Best 2-offset (factor_map2):  0.837  (reference)\n");

    /* ===================================================================
     * 4. Verification: predict h from data[t] + best state feature
     * =================================================================== */

    printf("\n=== Verification: BPC from State-Based Predictions ===\n\n");

    double bpc_actual = 0, bpc_state = 0;
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

        /* State-based prediction */
        {
            float h_pred[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                int fi = best_feat[j];
                int max_fv = feature_max[fi];
                int key = data[t] * max_fv + features[t][fi];

                /* Compute conditional mean on-the-fly */
                double sum = 0; int cnt = 0;
                for (int s = T_start; s < N; s++) {
                    int skey = data[s] * max_fv + features[s][fi];
                    if (skey == key) {
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
            bpc_state -= log2(probs[y] > 1e-10 ? probs[y] : 1e-10);
        }

        bpc_count++;
        if (bpc_count % 100 == 0)
            fprintf(stderr, "  position %d/%d\r", bpc_count, T_count - 1);
    }

    bpc_actual /= bpc_count;
    bpc_state /= bpc_count;

    printf("Positions evaluated: %d\n", bpc_count);
    printf("Actual RNN:                     %.4f bpc\n", bpc_actual);
    printf("State-based (byte + feature):   %.4f bpc\n", bpc_state);
    printf("Fixed 2-offset (factor_map2):   0.677 bpc (reference)\n");
    printf("Marginal (no context):          ~4.74 bpc\n\n");

    double gain_actual = 4.74 - bpc_actual;
    double gain_state  = 4.74 - bpc_state;
    printf("BPC gain captured: %.3f / %.3f = %.1f%%\n",
           gain_state, gain_actual, 100.0 * gain_state / gain_actual);

    return 0;
}
