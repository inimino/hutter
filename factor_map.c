/*
 * factor_map.c — Map UM patterns onto RNN hidden neurons.
 *
 * For each position t in the data, we know which backward-looking UM patterns
 * are active (from the n-gram and skip-k-gram inventories). For each hidden
 * neuron j, we find which UM features best predict sign(h_j[t]).
 *
 * This gives the factor map φ: architecture-natural (128 neurons) →
 * domain-natural (I→O skip-patterns). The map answers: what has the RNN
 * actually learned, expressed in terms of data patterns?
 *
 * Approach:
 *   1. Run RNN forward pass, record h[t][j] for all t, j
 *   2. For each offset d in {1,2,3,4,8,12,20,27}, build feature:
 *      f_d(t) = data[t-d] (the byte d steps back from position t)
 *   3. For each neuron j, for each offset d, for each byte value v:
 *      count how often sign(h_j) = +1 when data[t-d] = v
 *      → gives MI(sign(h_j), data[t-d])
 *   4. Find the best single-offset predictor for each neuron
 *   5. Find the best 2-offset conjunction predictor
 *   6. Verification: use factor map to predict h, run through Wy, measure bpc
 *
 * Usage: factor_map <data_file> <model_file>
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

/* Greedy skip-k-gram offsets from pattern-prior analysis */
static int OFFSETS[N_OFFSETS] = {1, 8, 20, 3, 27, 2, 12, 7};

/* Per-neuron factor map entry */
typedef struct {
    int neuron;
    /* Best single-offset predictor */
    int best_offset;          /* which offset index */
    double best_mi;           /* MI(sign(h_j), data[t-offset]) */
    double best_accuracy;     /* accuracy of best single predictor */
    /* Best 2-offset conjunction */
    int best_off1, best_off2;
    double conj_mi;
    double conj_accuracy;
    /* Wy importance: sum_o |Wy[o][j]| */
    double wy_importance;
    /* Mean activation and flip rate */
    double mean_h;
    double flip_rate;
    /* Classification */
    char class;  /* 'W' = Wx-driven (offset 1), 'H' = Wh-carried (skip), 'C' = conjunction, '?' = unexplained */
} FactorEntry;

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

    /* Run forward pass, record hidden states */
    float h_states[MAX_DATA][HIDDEN_SIZE];
    memset(rnn.h, 0, sizeof(rnn.h));
    for (int t = 0; t < N; t++) {
        rnn_step(&rnn, data[t]);
        memcpy(h_states[t], rnn.h, sizeof(rnn.h));
    }

    /* ===================================================================
     * 1. Per-neuron single-offset MI
     *
     * For each neuron j, offset d: compute MI(sign(h_j[t]), data[t-d])
     * MI = H(sign) + H(byte) - H(sign, byte)
     * where H(sign) = entropy of sign(h_j) across positions,
     *       H(byte) = entropy of data[t-d] across positions,
     *       H(sign, byte) = joint entropy
     * =================================================================== */

    printf("=== Single-Offset MI: MI(sign(h_j), data[t-d]) ===\n\n");
    printf("Greedy offsets: [");
    for (int d = 0; d < N_OFFSETS; d++) printf("%s%d", d?",":"", OFFSETS[d]);
    printf("]\n\n");

    /* For usable positions: t must be >= max_offset */
    int max_off = 0;
    for (int d = 0; d < N_OFFSETS; d++)
        if (OFFSETS[d] > max_off) max_off = OFFSETS[d];
    int T_start = max_off;  /* first usable position */
    int T_count = N - T_start;

    printf("Usable positions: %d (from t=%d to t=%d)\n\n", T_count, T_start, N-1);

    /* sign(h_j[t]) for each position */
    signed char h_sign[MAX_DATA][HIDDEN_SIZE];
    for (int t = 0; t < N; t++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            h_sign[t][j] = (h_states[t][j] >= 0) ? 1 : 0;  /* 0/1 for easier counting */

    /* MI computation for each (neuron, offset) pair */
    double mi_table[HIDDEN_SIZE][N_OFFSETS];

    /* Also track best single predictor per neuron:
     * For each neuron j, offset d: find the set of byte values v where
     * P(sign=+1 | data[t-d]=v) > 0.5, and use that as the predictor.
     * Accuracy = fraction of positions correctly predicted. */
    double acc_table[HIDDEN_SIZE][N_OFFSETS];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        /* Count sign(h_j) distribution */
        int sign_count[2] = {0, 0};
        for (int t = T_start; t < N; t++)
            sign_count[h_sign[t][j]]++;

        double p_sign[2];
        p_sign[0] = (double)sign_count[0] / T_count;
        p_sign[1] = (double)sign_count[1] / T_count;
        double H_sign = 0;
        for (int s = 0; s < 2; s++)
            if (p_sign[s] > 0) H_sign -= p_sign[s] * log2(p_sign[s]);

        for (int di = 0; di < N_OFFSETS; di++) {
            int d = OFFSETS[di];

            /* Joint counts: joint[byte][sign] */
            int joint[256][2];
            memset(joint, 0, sizeof(joint));
            int byte_count[256];
            memset(byte_count, 0, sizeof(byte_count));

            for (int t = T_start; t < N; t++) {
                int b = data[t - d];
                int s = h_sign[t][j];
                joint[b][s]++;
                byte_count[b]++;
            }

            /* H(byte) */
            double H_byte = 0;
            for (int b = 0; b < 256; b++) {
                if (byte_count[b] == 0) continue;
                double p = (double)byte_count[b] / T_count;
                H_byte -= p * log2(p);
            }

            /* H(sign, byte) */
            double H_joint = 0;
            for (int b = 0; b < 256; b++)
                for (int s = 0; s < 2; s++) {
                    if (joint[b][s] == 0) continue;
                    double p = (double)joint[b][s] / T_count;
                    H_joint -= p * log2(p);
                }

            mi_table[j][di] = H_sign + H_byte - H_joint;

            /* Accuracy: for each byte value, predict the majority sign */
            int correct = 0;
            for (int b = 0; b < 256; b++) {
                if (byte_count[b] == 0) continue;
                int majority = (joint[b][1] >= joint[b][0]) ? 1 : 0;
                correct += joint[b][majority];
            }
            acc_table[j][di] = (double)correct / T_count;
        }
    }

    /* Print MI table header */
    printf("%-6s", "h_j");
    for (int di = 0; di < N_OFFSETS; di++)
        printf("  off=%-3d", OFFSETS[di]);
    printf("  best_off  best_MI  best_acc\n");

    /* For each neuron, find best single offset */
    FactorEntry fmap[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        fmap[j].neuron = j;
        fmap[j].best_offset = 0;
        fmap[j].best_mi = mi_table[j][0];
        fmap[j].best_accuracy = acc_table[j][0];

        for (int di = 1; di < N_OFFSETS; di++) {
            if (mi_table[j][di] > fmap[j].best_mi) {
                fmap[j].best_mi = mi_table[j][di];
                fmap[j].best_offset = di;
                fmap[j].best_accuracy = acc_table[j][di];
            }
        }

        /* Wy importance */
        double wy_imp = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++)
            wy_imp += fabsf(rnn.Wy[o][j]);
        fmap[j].wy_importance = wy_imp;

        /* Mean activation and flip rate */
        double sum_h = 0;
        int flips = 0;
        for (int t = T_start; t < N; t++) {
            sum_h += h_states[t][j];
            if (t > T_start && h_sign[t][j] != h_sign[t-1][j]) flips++;
        }
        fmap[j].mean_h = sum_h / T_count;
        fmap[j].flip_rate = (double)flips / (T_count - 1);

        /* Classification */
        if (fmap[j].best_accuracy < 0.60)
            fmap[j].class = '?';
        else if (OFFSETS[fmap[j].best_offset] == 1)
            fmap[j].class = 'W';  /* Wx-driven */
        else
            fmap[j].class = 'H';  /* Wh-carried (skip) */
    }

    /* Print per-neuron MI table (sorted by best MI descending) */
    int order[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) order[j] = j;
    for (int i = 0; i < HIDDEN_SIZE - 1; i++)
        for (int j = i + 1; j < HIDDEN_SIZE; j++)
            if (fmap[order[j]].best_mi > fmap[order[i]].best_mi) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }

    printf("\nTop 30 neurons by MI with best single offset:\n");
    printf("%-6s", "h_j");
    for (int di = 0; di < N_OFFSETS; di++)
        printf("  off=%-3d", OFFSETS[di]);
    printf("  best_off  best_MI  acc%%  class  Wy_imp\n");

    for (int r = 0; r < 30 && r < HIDDEN_SIZE; r++) {
        int j = order[r];
        printf("h%-5d", j);
        for (int di = 0; di < N_OFFSETS; di++)
            printf("  %.4f ", mi_table[j][di]);
        printf("  off=%-3d  %.4f  %5.1f   %c     %.1f\n",
               OFFSETS[fmap[j].best_offset], fmap[j].best_mi,
               100.0 * fmap[j].best_accuracy, fmap[j].class,
               fmap[j].wy_importance);
    }

    /* ===================================================================
     * 2. Two-offset conjunction MI
     *
     * For each neuron j, for each pair (d1, d2): compute
     * MI(sign(h_j), (data[t-d1], data[t-d2]))
     * This captures neurons that respond to skip-patterns.
     * =================================================================== */

    printf("\n=== Two-Offset Conjunction MI ===\n\n");

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        int sign_count[2] = {0, 0};
        for (int t = T_start; t < N; t++)
            sign_count[h_sign[t][j]]++;

        double p_sign[2];
        p_sign[0] = (double)sign_count[0] / T_count;
        p_sign[1] = (double)sign_count[1] / T_count;
        double H_sign = 0;
        for (int s = 0; s < 2; s++)
            if (p_sign[s] > 0) H_sign -= p_sign[s] * log2(p_sign[s]);

        double best_conj_mi = 0;
        int best_d1 = 0, best_d2 = 1;
        double best_conj_acc = 0;

        for (int d1i = 0; d1i < N_OFFSETS; d1i++) {
            for (int d2i = d1i + 1; d2i < N_OFFSETS; d2i++) {
                int d1 = OFFSETS[d1i], d2 = OFFSETS[d2i];

                /* Joint counts: context_hash → [sign0_count, sign1_count]
                 * Use a simple 256*256 table (byte-pair context) */
                /* For memory: use a hash table since 256*256*2 = 128K ints is fine */
                int joint[256][256][2];  /* stack-allocated, 256KB — OK for this */
                memset(joint, 0, sizeof(joint));
                int ctx_count[256][256];
                memset(ctx_count, 0, sizeof(ctx_count));

                for (int t = T_start; t < N; t++) {
                    int b1 = data[t - d1];
                    int b2 = data[t - d2];
                    int s = h_sign[t][j];
                    joint[b1][b2][s]++;
                    ctx_count[b1][b2]++;
                }

                /* H(ctx) */
                double H_ctx = 0;
                for (int b1 = 0; b1 < 256; b1++)
                    for (int b2 = 0; b2 < 256; b2++) {
                        if (ctx_count[b1][b2] == 0) continue;
                        double p = (double)ctx_count[b1][b2] / T_count;
                        H_ctx -= p * log2(p);
                    }

                /* H(sign, ctx) */
                double H_joint = 0;
                for (int b1 = 0; b1 < 256; b1++)
                    for (int b2 = 0; b2 < 256; b2++)
                        for (int s = 0; s < 2; s++) {
                            if (joint[b1][b2][s] == 0) continue;
                            double p = (double)joint[b1][b2][s] / T_count;
                            H_joint -= p * log2(p);
                        }

                double mi = H_sign + H_ctx - H_joint;

                /* Accuracy: majority vote per context */
                int correct = 0;
                for (int b1 = 0; b1 < 256; b1++)
                    for (int b2 = 0; b2 < 256; b2++) {
                        if (ctx_count[b1][b2] == 0) continue;
                        int majority = (joint[b1][b2][1] >= joint[b1][b2][0]) ? 1 : 0;
                        correct += joint[b1][b2][majority];
                    }
                double acc = (double)correct / T_count;

                if (mi > best_conj_mi) {
                    best_conj_mi = mi;
                    best_d1 = d1i;
                    best_d2 = d2i;
                    best_conj_acc = acc;
                }
            }
        }

        fmap[j].best_off1 = best_d1;
        fmap[j].best_off2 = best_d2;
        fmap[j].conj_mi = best_conj_mi;
        fmap[j].conj_accuracy = best_conj_acc;

        /* Reclassify if conjunction is significantly better than single */
        if (best_conj_acc > fmap[j].best_accuracy + 0.05 && best_conj_acc >= 0.70)
            fmap[j].class = 'C';
    }

    /* Print conjunction results for top neurons */
    printf("Top 30 neurons by conjunction MI:\n");
    printf("%-6s  single(off,MI,acc)         conj(off1,off2,MI,acc)     class  gain\n", "h_j");

    /* Re-sort by conjunction MI */
    for (int i = 0; i < HIDDEN_SIZE - 1; i++)
        for (int j = i + 1; j < HIDDEN_SIZE; j++)
            if (fmap[order[j]].conj_mi > fmap[order[i]].conj_mi) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }

    for (int r = 0; r < 30 && r < HIDDEN_SIZE; r++) {
        int j = order[r];
        printf("h%-5d  off=%d  MI=%.3f acc=%.1f%%   off=(%d,%d) MI=%.3f acc=%.1f%%  %c   %+.1f%%\n",
               j,
               OFFSETS[fmap[j].best_offset], fmap[j].best_mi,
               100.0 * fmap[j].best_accuracy,
               OFFSETS[fmap[j].best_off1], OFFSETS[fmap[j].best_off2],
               fmap[j].conj_mi, 100.0 * fmap[j].conj_accuracy,
               fmap[j].class,
               100.0 * (fmap[j].conj_accuracy - fmap[j].best_accuracy));
    }

    /* ===================================================================
     * 3. Factor map summary
     * =================================================================== */

    printf("\n=== Factor Map Summary ===\n\n");

    int n_wx = 0, n_wh = 0, n_conj = 0, n_unknown = 0;
    double total_mi = 0, total_wy = 0;
    double explained_wy = 0;

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        total_wy += fmap[j].wy_importance;
        double best_acc = fmap[j].best_accuracy;
        if (fmap[j].conj_accuracy > best_acc) best_acc = fmap[j].conj_accuracy;

        switch (fmap[j].class) {
            case 'W': n_wx++;  explained_wy += fmap[j].wy_importance; break;
            case 'H': n_wh++;  explained_wy += fmap[j].wy_importance; break;
            case 'C': n_conj++; explained_wy += fmap[j].wy_importance; break;
            default:  n_unknown++; break;
        }
        total_mi += fmap[j].best_mi;
    }

    printf("Neuron classification:\n");
    printf("  W (Wx-driven, offset 1):     %3d neurons\n", n_wx);
    printf("  H (Wh-carried, skip offset): %3d neurons\n", n_wh);
    printf("  C (conjunction of 2 offsets): %3d neurons\n", n_conj);
    printf("  ? (unexplained, acc < 60%%):   %3d neurons\n", n_unknown);
    printf("\n");
    printf("Total MI across all neurons: %.3f bits\n", total_mi);
    printf("Explained Wy importance: %.1f / %.1f (%.1f%%)\n",
           explained_wy, total_wy, 100.0 * explained_wy / total_wy);

    /* Per-offset distribution */
    printf("\nPer-offset breakdown (best single):\n");
    printf("  offset  neurons  avg_MI  avg_acc\n");
    for (int di = 0; di < N_OFFSETS; di++) {
        int cnt = 0;
        double sum_mi = 0, sum_acc = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (fmap[j].best_offset == di) {
                cnt++;
                sum_mi += fmap[j].best_mi;
                sum_acc += fmap[j].best_accuracy;
            }
        }
        if (cnt > 0)
            printf("  %-6d  %3d      %.4f  %.1f%%\n",
                   OFFSETS[di], cnt, sum_mi/cnt, 100.0*sum_acc/cnt);
    }

    /* ===================================================================
     * 4. Verification: predict h from factor map, run Wy, measure bpc
     *
     * For each position t:
     *   - Use the factor map to predict sign(h_j[t]) for each neuron
     *   - Set h_predicted[j] = predicted_sign * mean(|h_j|)
     *   - Compute logits via Wy, softmax, -log2(p[y])
     * =================================================================== */

    printf("\n=== Verification: BPC from Factor Map Predictions ===\n\n");

    /* Precompute: for each neuron j, build the lookup table for its best predictor.
     * best_sign[j][byte_value] = predicted sign for neuron j given data[t-best_offset]=byte_value */
    signed char best_sign_table[HIDDEN_SIZE][256];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        int di = fmap[j].best_offset;
        int d = OFFSETS[di];

        /* Count sign distribution per byte value at offset d */
        int joint[256][2];
        memset(joint, 0, sizeof(joint));
        for (int t = T_start; t < N; t++) {
            int b = data[t - d];
            int s = h_sign[t][j];
            joint[b][s]++;
        }

        for (int b = 0; b < 256; b++)
            best_sign_table[j][b] = (joint[b][1] >= joint[b][0]) ? 1 : 0;
    }

    /* Precompute mean |h_j| for scaling predictions */
    float mean_abs_h[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double sum = 0;
        for (int t = T_start; t < N; t++)
            sum += fabsf(h_states[t][j]);
        mean_abs_h[j] = sum / T_count;
    }

    /* Compute BPC: actual RNN, single-offset prediction, and conjunction prediction */
    double bpc_actual = 0, bpc_single = 0;
    int bpc_count = 0;

    for (int t = T_start; t < N - 1; t++) {
        int y = data[t + 1];  /* next byte to predict */

        /* Actual RNN BPC */
        {
            float logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                logits[o] = rnn.by[o];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    logits[o] += rnn.Wy[o][j] * h_states[t][j];
            }
            float probs[OUTPUT_SIZE];
            softmax(logits, probs, OUTPUT_SIZE);
            if (probs[y] > 1e-10)
                bpc_actual -= log2f(probs[y]);
            else
                bpc_actual += 30;
        }

        /* Single-offset factor map prediction */
        {
            float h_pred[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                int d = OFFSETS[fmap[j].best_offset];
                int b = data[t - d];
                int predicted_sign = best_sign_table[j][b] ? 1 : -1;
                h_pred[j] = predicted_sign * mean_abs_h[j];
            }

            float logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                logits[o] = rnn.by[o];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    logits[o] += rnn.Wy[o][j] * h_pred[j];
            }
            float probs[OUTPUT_SIZE];
            softmax(logits, probs, OUTPUT_SIZE);
            if (probs[y] > 1e-10)
                bpc_single -= log2f(probs[y]);
            else
                bpc_single += 30;
        }

        bpc_count++;
    }

    bpc_actual /= bpc_count;
    bpc_single /= bpc_count;

    printf("Positions evaluated: %d\n", bpc_count);
    printf("Actual RNN:                %.4f bpc\n", bpc_actual);
    printf("Factor map (single-off):   %.4f bpc\n", bpc_single);
    printf("Marginal (no context):     ~4.74 bpc\n");
    printf("UM floor (skip-8):         0.043 bpc\n\n");

    /* Also measure: how much of the bpc gain does the factor map capture? */
    double gain_actual = 4.74 - bpc_actual;
    double gain_fmap   = 4.74 - bpc_single;
    printf("BPC gain captured: %.3f / %.3f = %.1f%%\n",
           gain_fmap, gain_actual, 100.0 * gain_fmap / gain_actual);

    /* ===================================================================
     * 5. Ablation: remove one neuron at a time, measure bpc change
     *
     * For each neuron j, set h_j = 0 and recompute bpc.
     * The bpc increase = importance of that neuron.
     * =================================================================== */

    printf("\n=== Ablation: Per-Neuron Importance ===\n\n");

    typedef struct { int neuron; double bpc_without; double delta_bpc; } Ablation;
    Ablation ablations[HIDDEN_SIZE];

    for (int j_abl = 0; j_abl < HIDDEN_SIZE; j_abl++) {
        double bpc_abl = 0;

        for (int t = T_start; t < N - 1; t++) {
            int y = data[t + 1];
            float logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                logits[o] = rnn.by[o];
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    if (j == j_abl) continue;  /* ablate this neuron */
                    logits[o] += rnn.Wy[o][j] * h_states[t][j];
                }
            }
            float probs[OUTPUT_SIZE];
            softmax(logits, probs, OUTPUT_SIZE);
            if (probs[y] > 1e-10)
                bpc_abl -= log2f(probs[y]);
            else
                bpc_abl += 30;
        }
        bpc_abl /= bpc_count;

        ablations[j_abl].neuron = j_abl;
        ablations[j_abl].bpc_without = bpc_abl;
        ablations[j_abl].delta_bpc = bpc_abl - bpc_actual;
    }

    /* Sort by delta_bpc descending (most important first) */
    for (int i = 0; i < HIDDEN_SIZE - 1; i++)
        for (int j = i + 1; j < HIDDEN_SIZE; j++)
            if (ablations[j].delta_bpc > ablations[i].delta_bpc) {
                Ablation tmp = ablations[i]; ablations[i] = ablations[j]; ablations[j] = tmp;
            }

    printf("Top 30 most important neurons (by ablation delta bpc):\n");
    printf("%-6s  bpc_without  delta_bpc  best_off  MI     acc%%   class\n", "h_j");
    for (int r = 0; r < 30 && r < HIDDEN_SIZE; r++) {
        int j = ablations[r].neuron;
        printf("h%-5d  %.4f      %+.4f     off=%-3d   %.3f  %5.1f   %c\n",
               j, ablations[r].bpc_without, ablations[r].delta_bpc,
               OFFSETS[fmap[j].best_offset], fmap[j].best_mi,
               100.0 * fmap[j].best_accuracy, fmap[j].class);
    }

    /* Cross-tabulate: are the most important neurons also the best explained? */
    printf("\n=== Importance × Explainability Cross-Tab ===\n\n");

    /* Top 20 by ablation: how many are explained (W/H/C)? */
    int top20_explained = 0;
    double top20_delta_sum = 0;
    for (int r = 0; r < 20; r++) {
        int j = ablations[r].neuron;
        top20_delta_sum += ablations[r].delta_bpc;
        if (fmap[j].class != '?') top20_explained++;
    }
    printf("Top 20 most important neurons: %d/20 explained by factor map\n", top20_explained);
    printf("Sum of ablation deltas (top 20): %.4f bpc\n", top20_delta_sum);

    /* Explained neurons: total ablation delta vs unexplained */
    double delta_explained = 0, delta_unexplained = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        if (fmap[j].class != '?')
            delta_explained += (ablations[j].bpc_without - bpc_actual);
        else
            delta_unexplained += (ablations[j].bpc_without - bpc_actual);
    }
    /* Note: ablations array is sorted, need to use fmap index directly */
    delta_explained = 0; delta_unexplained = 0;
    for (int r = 0; r < HIDDEN_SIZE; r++) {
        int j = ablations[r].neuron;
        if (fmap[j].class != '?')
            delta_explained += ablations[r].delta_bpc;
        else
            delta_unexplained += ablations[r].delta_bpc;
    }
    printf("Total ablation delta (explained neurons): %.4f bpc\n", delta_explained);
    printf("Total ablation delta (unexplained neurons): %.4f bpc\n", delta_unexplained);
    printf("Interpretability coverage: %.1f%%\n",
           100.0 * delta_explained / (delta_explained + delta_unexplained));

    /* ===================================================================
     * 6. Full factor map table (sorted by ablation importance)
     * =================================================================== */

    printf("\n=== Full Factor Map (sorted by importance) ===\n\n");
    printf("rank  h_j   class  best_off  MI     acc%%  conj(off1,off2)  conj_acc%%  delta_bpc  Wy_imp  flip%%\n");

    for (int r = 0; r < HIDDEN_SIZE; r++) {
        int j = ablations[r].neuron;
        printf("%3d   h%-4d  %c      off=%-3d   %.3f  %5.1f  (%d,%d)%10s  %5.1f      %+.4f     %5.1f   %5.1f\n",
               r+1, j, fmap[j].class,
               OFFSETS[fmap[j].best_offset], fmap[j].best_mi,
               100.0 * fmap[j].best_accuracy,
               OFFSETS[fmap[j].best_off1], OFFSETS[fmap[j].best_off2], "",
               100.0 * fmap[j].conj_accuracy,
               ablations[r].delta_bpc,
               fmap[j].wy_importance,
               100.0 * fmap[j].flip_rate);
    }

    return 0;
}
