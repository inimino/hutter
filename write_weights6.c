/*
 * write_weights6.c — Fully analytic construction with PROPER hash.
 *
 * write_weights5 discovered that the hash function was degenerate:
 * only 2 distinct patterns out of 256 bytes (lowest bit depends only on x&1).
 * Despite this, analytic W_y achieved 3.92 bpc (beating trained 4.97).
 *
 * This version uses a proper hash that's actually injective (256 distinct
 * 8-bit patterns for 256 bytes). With full byte identity encoded in
 * each group, the analytic W_y should be even better.
 *
 * Also tests: what happens when we increase group_size to encode more
 * information per offset?
 *
 * Usage: write_weights6 <data_file> [trained_model_for_comparison]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define H HIDDEN_SIZE

typedef struct {
    float Wx[H][INPUT_SIZE];
    float Wh[H][H];
    float bh[H];
    float Wy[OUTPUT_SIZE][H];
    float by[OUTPUT_SIZE];
} Model;

void load_model(Model* m, const char* path) {
    FILE* f = fopen(path, "rb");
    fread(m->Wx, sizeof(float), H*INPUT_SIZE, f);
    fread(m->Wh, sizeof(float), H*H, f);
    fread(m->bh, sizeof(float), H, f);
    fread(m->Wy, sizeof(float), OUTPUT_SIZE*H, f);
    fread(m->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

void rnn_step(float* out, float* in, int x, Model* m) {
    for (int i = 0; i < H; i++) {
        float z = m->bh[i] + m->Wx[i][x];
        for (int j = 0; j < H; j++) z += m->Wh[i][j]*in[j];
        out[i] = tanhf(z);
    }
}

double eval_bpc(unsigned char* data, int n, Model* m) {
    float h[H]; memset(h, 0, sizeof(h));
    double total = 0;
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], m);
        memcpy(h, hn, sizeof(h));
        double P[OUTPUT_SIZE], max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m->by[o];
            for (int j = 0; j < H; j++) s += m->Wy[o][j]*h[j];
            P[o] = s; if (s > max_l) max_l = s;
        }
        double se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(P[o]-max_l); se += P[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;
        int y = data[t+1];
        total += -log2(P[y] > 1e-30 ? P[y] : 1e-30);
    }
    return total / (n-1);
}

/* PROPER hash: mix bits thoroughly so each neuron gets a distinct partition */
int hash_sign(int x, int j) {
    unsigned h = (unsigned)x;
    h = ((h >> 4) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h);
    h += (unsigned)j * 2654435761u;
    h = ((h >> 4) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h);
    return (h & 1) ? 1 : -1;
}

/* Check injectivity of hash for group_size bits */
int check_injectivity(int group_size) {
    int seen[256]; memset(seen, 0, sizeof(seen));
    int distinct = 0;
    for (int x = 0; x < 256; x++) {
        int pattern = 0;
        for (int j = 0; j < group_size && j < 8; j++)
            if (hash_sign(x, j) == 1) pattern |= (1 << j);
        if (!seen[pattern]) { distinct++; seen[pattern] = 1; }
    }
    return distinct;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <data> [model]\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;

    printf("=== Write Weights v6: Proper Hash + Analytic W_y ===\n");
    printf("Data: %d bytes\n\n", n);

    /* Check hash quality */
    printf("Hash injectivity test:\n");
    for (int gs = 1; gs <= 8; gs++)
        printf("  %d bits: %d distinct / 256 bytes\n", gs, check_injectivity(gs));
    printf("\n");

    /* Compute data statistics */
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));
    for (int t = 0; t < n; t++) byte_count[data[t]]++;

    double marginal[256], marginal_total = 0;
    for (int o = 0; o < 256; o++) {
        marginal[o] = byte_count[o] + 0.5;
        marginal_total += marginal[o];
    }
    for (int o = 0; o < 256; o++) marginal[o] /= marginal_total;

    /* Skip-bigram counts */
    static double skip_bigram[16][256][256];
    memset(skip_bigram, 0, sizeof(skip_bigram));
    for (int d = 0; d < 16; d++)
        for (int t = d; t < n-1; t++)
            skip_bigram[d][data[t-d]][data[t+1]] += 1.0;

    /* ========== Architecture: 16 groups of 8 ========== */
    int n_groups = 16;
    int group_size = H / n_groups;

    Model model;
    memset(&model, 0, sizeof(Model));

    float scale_wx = 10.0;
    float carry_weight = 5.0;

    /* W_x: hash encoding for group 0 */
    for (int j = 0; j < group_size; j++)
        for (int x = 0; x < INPUT_SIZE; x++)
            model.Wx[j][x] = scale_wx * hash_sign(x, j);

    /* W_h: shift register */
    for (int g = 1; g < n_groups; g++)
        for (int j = 0; j < group_size; j++)
            model.Wh[g*group_size + j][(g-1)*group_size + j] = carry_weight;

    /* b_y from marginal */
    for (int o = 0; o < OUTPUT_SIZE; o++)
        model.by[o] = logf((byte_count[o] + 0.5f) / (n + 128.0f));

    /* ========== Verify encoding ========== */
    static float h_traj[1025][H];
    float h[H]; memset(h, 0, sizeof(h));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &model);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
    }
    int T = n - 1;

    int correct = 0, checks = 0;
    for (int t = n_groups; t < T; t++) {
        for (int g = 0; g < n_groups; g++) {
            int match = 1;
            for (int j = 0; j < group_size; j++) {
                int expected = hash_sign(data[t-g], j);
                int actual = (h_traj[t][g*group_size + j] > 0) ? 1 : -1;
                if (expected != actual) { match = 0; break; }
            }
            if (match) correct++;
            checks++;
        }
    }
    printf("Shift-register encoding accuracy: %d/%d = %.1f%%\n\n",
           correct, checks, 100.0*correct/checks);

    /* ========== Analytic W_y: Log-ratio approach ========== */
    printf("=== Analytic W_y (log-ratio) ===\n");

    /* For neuron j_local in group g:
     * S+ = {x : hash_sign(x, j_local) = +1}
     * W_y[o][g*8+j] = scale * (log P(o | data[t-g] ∈ S+) - log P(o | data[t-g] ∈ S-))
     */

    double best_bpc = 999, best_alpha = 1.0, best_scale = 1.0;
    static float orig_Wy[OUTPUT_SIZE][H];

    for (double alpha = 0.5; alpha <= 5.0; alpha += 0.5) {
        for (int g = 0; g < n_groups; g++) {
            int d = g;
            for (int j_local = 0; j_local < group_size; j_local++) {
                int is_positive[256];
                for (int x = 0; x < 256; x++)
                    is_positive[x] = (hash_sign(x, j_local) == 1);

                double count_pos[256], count_neg[256];
                double total_pos = alpha * 256, total_neg = alpha * 256;
                for (int o = 0; o < 256; o++) { count_pos[o] = alpha; count_neg[o] = alpha; }

                for (int t = d; t < n-1; t++) {
                    int src = data[t - d], dst = data[t + 1];
                    if (is_positive[src]) {
                        count_pos[dst] += 1.0; total_pos += 1.0;
                    } else {
                        count_neg[dst] += 1.0; total_neg += 1.0;
                    }
                }

                int ni = g * group_size + j_local;
                for (int o = 0; o < 256; o++) {
                    double lr = log(count_pos[o] / total_pos) - log(count_neg[o] / total_neg);
                    model.Wy[o][ni] = (float)lr;
                }
            }
        }

        memcpy(orig_Wy, model.Wy, sizeof(orig_Wy));
        for (float scale = 0.05; scale <= 3.0; scale += 0.05) {
            for (int o = 0; o < OUTPUT_SIZE; o++)
                for (int j = 0; j < H; j++)
                    model.Wy[o][j] = scale * orig_Wy[o][j];
            double bpc = eval_bpc(data, n, &model);
            if (bpc < best_bpc) {
                best_bpc = bpc; best_alpha = alpha; best_scale = scale;
            }
        }
    }
    printf("  Best: alpha=%.1f, scale=%.2f → %.4f bpc\n", best_alpha, best_scale, best_bpc);

    /* ========== Fourier/least-squares W_y ========== */
    printf("\n=== Fourier least-squares W_y ===\n");

    /* For each group g, each output o:
     * w_j = (1/N) * sum_x count(x at offset g) * f_o(x) * sign_j(x)
     * where f_o(x) = log P(o | data[t-g]=x) - log P(o)
     */
    for (int g = 0; g < n_groups; g++) {
        int d = g;
        double byte_at[256]; memset(byte_at, 0, sizeof(byte_at));
        double total_at = 0;
        for (int t = d; t < n-1; t++) {
            byte_at[data[t-d]] += 1.0; total_at += 1.0;
        }

        for (int o = 0; o < 256; o++) {
            for (int j_local = 0; j_local < group_size; j_local++) {
                double sum = 0, norm = 0;
                for (int x = 0; x < 256; x++) {
                    if (byte_at[x] < 0.5) continue;
                    double total_x = 0;
                    for (int o2 = 0; o2 < 256; o2++)
                        total_x += skip_bigram[d][x][o2] + 0.01;
                    double p_cond = (skip_bigram[d][x][o] + 0.01) / total_x;
                    double f_val = log(p_cond) - log(marginal[o]);
                    int s = hash_sign(x, j_local);
                    sum += byte_at[x] * f_val * s;
                    norm += byte_at[x];
                }
                model.Wy[o][g*group_size + j_local] = (float)(sum / (norm > 0 ? norm : 1));
            }
        }
    }

    /* Scale search */
    memcpy(orig_Wy, model.Wy, sizeof(orig_Wy));
    double best_fourier = 999; float best_fscale = 1.0;
    for (float scale = 0.05; scale <= 10.0; scale += 0.05) {
        for (int o = 0; o < OUTPUT_SIZE; o++)
            for (int j = 0; j < H; j++)
                model.Wy[o][j] = scale * orig_Wy[o][j];
        double bpc = eval_bpc(data, n, &model);
        if (bpc < best_fourier) { best_fourier = bpc; best_fscale = scale; }
    }
    printf("  Best: scale=%.2f → %.4f bpc\n", best_fscale, best_fourier);

    /* ========== Naive Bayes with exact byte identity ========== */
    printf("\n=== Naive Bayes (exact byte identity, 16 offsets) ===\n");

    double nb_best = 999; float nb_best_temp = 1.0;
    for (float temp = 0.001; temp <= 1.0; temp += 0.001) {
        double nb_total = 0; int nb_count = 0;
        for (int t = 16; t < n-1; t++) {
            double logits[256];
            for (int o = 0; o < 256; o++)
                logits[o] = log(marginal[o]);
            for (int d = 0; d < 16; d++) {
                int src = data[t - d];
                double total_d = 0;
                for (int o = 0; o < 256; o++)
                    total_d += skip_bigram[d][src][o] + 0.5;
                for (int o = 0; o < 256; o++) {
                    double p_cond = (skip_bigram[d][src][o] + 0.5) / total_d;
                    logits[o] += temp * (log(p_cond) - log(marginal[o]));
                }
            }
            double max_l = -1e30;
            for (int o = 0; o < 256; o++) if (logits[o] > max_l) max_l = logits[o];
            double se = 0;
            for (int o = 0; o < 256; o++) se += exp(logits[o] - max_l);
            double p = exp(logits[data[t+1]] - max_l) / se;
            nb_total += -log2(p > 1e-30 ? p : 1e-30);
            nb_count++;
        }
        double bpc = nb_total / nb_count;
        if (bpc < nb_best) { nb_best = bpc; nb_best_temp = temp; }
    }
    printf("  Naive Bayes: temp=%.3f → %.4f bpc\n", nb_best_temp, nb_best);

    /* ========== Comparison: optimized W_y (for reference) ========== */
    printf("\n=== For reference: optimized W_y (gradient descent) ===\n");

    /* Reset W_y */
    memset(model.Wy, 0, sizeof(model.Wy));
    for (int o = 0; o < OUTPUT_SIZE; o++)
        model.by[o] = logf((byte_count[o] + 0.5f) / (n + 128.0f));

    float lr = 0.5;
    for (int epoch = 0; epoch < 1000; epoch++) {
        static float dWy[OUTPUT_SIZE][H];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));
        double total_loss = 0;

        for (int t = n_groups; t < T-1; t++) {
            float* ht = h_traj[t];
            int y = data[t+1];
            double logits[OUTPUT_SIZE], max_l = -1e30;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                double s = model.by[o];
                for (int j = 0; j < H; j++) s += model.Wy[o][j]*ht[j];
                logits[o] = s; if (s > max_l) max_l = s;
            }
            double P[OUTPUT_SIZE], se = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(logits[o]-max_l); se += P[o]; }
            for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;
            total_loss += -log2(P[y] > 1e-30 ? P[y] : 1e-30);

            for (int o = 0; o < OUTPUT_SIZE; o++) {
                float err = (float)(P[o] - (o == y ? 1.0 : 0.0));
                dby[o] += err;
                for (int j = 0; j < H; j++)
                    dWy[o][j] += err * ht[j];
            }
        }

        float sc = lr / (T - 1 - n_groups);
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            model.by[o] -= sc * dby[o];
            for (int j = 0; j < H; j++)
                model.Wy[o][j] -= sc * dWy[o][j];
        }

        if (epoch == 300) lr *= 0.3;
        if (epoch == 600) lr *= 0.3;
        if (epoch == 800) lr *= 0.3;
    }

    double bpc_opt = eval_bpc(data, n, &model);
    printf("  Optimized W_y: %.4f bpc\n", bpc_opt);

    /* ========== Summary ========== */
    printf("\n========================================\n");
    printf("=== SUMMARY: PROPER HASH ===\n");
    printf("========================================\n");
    printf("Configuration                              bpc\n");
    printf("----------------------------------------------\n");
    printf("Uniform                                    8.000\n");
    printf("Analytic log-ratio W_y                     %.4f\n", best_bpc);
    printf("Fourier/LS W_y                             %.4f\n", best_fourier);
    printf("Naive Bayes (exact, 16 offsets)            %.4f\n", nb_best);
    printf("Optimized W_y (gradient, 1000 epochs)      %.4f\n", bpc_opt);
    printf("----------------------------------------------\n");

    if (argc >= 3) {
        Model trained; load_model(&trained, argv[2]);
        double trained_bpc = eval_bpc(data, n, &trained);
        printf("Trained model                              %.4f\n", trained_bpc);
    }

    printf("\nPrevious (degenerate hash): analytic 3.92, opt 3.57\n");
    printf("If proper hash helps: analytic should improve substantially.\n");

    return 0;
}
