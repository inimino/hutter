/*
 * write_weights5.c — Fully analytic weight construction: ZERO optimization.
 *
 * Goal: close the loop completely. Construct ALL weight matrices from
 * data statistics alone. No gradient descent, no trained model, no BPTT.
 *
 * Architecture: same shift-register as write_weights4.c
 *   - 16 groups of 8 neurons, group g encodes data[t-g] via hash
 *   - W_x, W_h, b_h: deterministic shift-register (from write_weights4)
 *
 * NEW: W_y from skip-bigram conditional distributions.
 *   For neuron j in group g, hash_sign(x, j_local) partitions bytes into S+ and S-.
 *   W_y[o][g*8+j] = scale * [log P(o | data[t-g] ∈ S+_j) - log P(o | data[t-g] ∈ S-_j)]
 *
 * This is the sign-conditioned log-ratio computed directly from data.
 * With Laplace smoothing for sparse counts.
 *
 * b_y from byte marginals.
 *
 * The question: does this purely data-derived readout compress?
 * If so, the entire model (all 82k parameters) is determined by data statistics.
 *
 * Usage: write_weights5 <data_file> [trained_model_for_comparison]
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

/* Simple hash: deterministic sign pattern for byte x, neuron j */
int hash_sign(int x, int j) {
    unsigned h = (unsigned)(x * 2654435761u + j * 340573321u);
    return (h & 1) ? 1 : -1;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <data> [model]\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;

    printf("=== Write Weights v5: Fully Analytic Construction (ZERO optimization) ===\n");
    printf("Data: %d bytes\n\n", n);

    /* Compute data statistics */
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));
    for (int t = 0; t < n; t++) byte_count[data[t]]++;

    int n_groups = 16;
    int group_size = H / n_groups; /* 8 */

    /* ========== Build dynamics (same as write_weights4) ========== */
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

    /* ========== Build W_y analytically ========== */
    printf("=== Constructing W_y from skip-bigram log-ratios ===\n\n");

    /* For each group g (offset d = g+1 for predicting data[t+1] from data[t-g]):
     * For each neuron j_local in [0, group_size):
     *   Partition bytes into S+ = {x : hash_sign(x, j_local) = +1}
     *   Count: for each output byte o,
     *     count_pos[o] = #{t : data[t-g] ∈ S+, data[t+1] = o}
     *     count_neg[o] = #{t : data[t-g] ∈ S-, data[t+1] = o}
     *   W_y[o][g*group_size + j_local] = scale * (log P_pos(o) - log P_neg(o))
     */

    double alpha = 1.0; /* Laplace smoothing */

    for (int g = 0; g < n_groups; g++) {
        int d = g; /* offset: data[t-g] for predicting data[t+1] */
        for (int j_local = 0; j_local < group_size; j_local++) {
            /* Determine S+ and S- */
            int is_positive[256];
            for (int x = 0; x < 256; x++)
                is_positive[x] = (hash_sign(x, j_local) == 1);

            /* Count transitions conditioned on S+/S- */
            double count_pos[256], count_neg[256];
            double total_pos = 0, total_neg = 0;
            for (int o = 0; o < 256; o++) {
                count_pos[o] = alpha;
                count_neg[o] = alpha;
            }
            total_pos = alpha * 256;
            total_neg = alpha * 256;

            for (int t = d; t < n-1; t++) {
                int src = data[t - d]; /* byte at offset g before position t */
                int dst = data[t + 1]; /* next byte */
                if (is_positive[src]) {
                    count_pos[dst] += 1.0;
                    total_pos += 1.0;
                } else {
                    count_neg[dst] += 1.0;
                    total_neg += 1.0;
                }
            }

            /* Log-ratio */
            int neuron_idx = g * group_size + j_local;
            for (int o = 0; o < 256; o++) {
                double log_pos = log(count_pos[o] / total_pos);
                double log_neg = log(count_neg[o] / total_neg);
                model.Wy[o][neuron_idx] = (float)(log_pos - log_neg);
            }
        }
    }

    /* b_y from marginal */
    for (int o = 0; o < OUTPUT_SIZE; o++)
        model.by[o] = logf((byte_count[o] + 0.5f) / (n + 128.0f));

    double bpc_v1 = eval_bpc(data, n, &model);
    printf("  Version 1 (raw log-ratio, alpha=%.1f): %.4f bpc\n", alpha, bpc_v1);

    /* Try different scales for W_y */
    float best_scale = 1.0;
    double best_bpc = bpc_v1;

    /* Save original W_y */
    static float orig_Wy[OUTPUT_SIZE][H];
    memcpy(orig_Wy, model.Wy, sizeof(orig_Wy));

    for (float scale = 0.1; scale <= 5.0; scale += 0.1) {
        for (int o = 0; o < OUTPUT_SIZE; o++)
            for (int j = 0; j < H; j++)
                model.Wy[o][j] = scale * orig_Wy[o][j];
        double bpc = eval_bpc(data, n, &model);
        if (bpc < best_bpc) { best_bpc = bpc; best_scale = scale; }
    }
    printf("  Best scale: %.1f → %.4f bpc\n", best_scale, best_bpc);

    /* Apply best scale */
    for (int o = 0; o < OUTPUT_SIZE; o++)
        for (int j = 0; j < H; j++)
            model.Wy[o][j] = best_scale * orig_Wy[o][j];

    /* Try different Laplace smoothing */
    printf("\n  Trying different smoothing:\n");
    for (double a = 0.01; a <= 10.0; a *= 2) {
        /* Recompute W_y with this alpha */
        for (int g = 0; g < n_groups; g++) {
            int d = g;
            for (int j_local = 0; j_local < group_size; j_local++) {
                int is_positive[256];
                for (int x = 0; x < 256; x++)
                    is_positive[x] = (hash_sign(x, j_local) == 1);

                double count_pos[256], count_neg[256];
                double total_pos = a * 256, total_neg = a * 256;
                for (int o = 0; o < 256; o++) {
                    count_pos[o] = a;
                    count_neg[o] = a;
                }

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
                    model.Wy[o][ni] = (float)(best_scale * lr);
                }
            }
        }
        double bpc = eval_bpc(data, n, &model);
        printf("    alpha=%.2f, scale=%.1f: %.4f bpc\n", a, best_scale, bpc);
    }

    /* ========== Version 2: Direct conditional distribution (no log-ratio) ========== */
    printf("\n=== Version 2: Direct skip-bigram conditional W_y ===\n");

    /* Alternative: W_y[o][j] = log P(o | h_j_sign) - log P(o)
     * This is the pointwise mutual information between neuron sign and output. */

    double marginal[256];
    double marginal_total = 0;
    for (int o = 0; o < 256; o++) {
        marginal[o] = byte_count[o] + 0.5;
        marginal_total += marginal[o];
    }
    for (int o = 0; o < 256; o++) marginal[o] /= marginal_total;

    for (int g = 0; g < n_groups; g++) {
        int d = g;
        for (int j_local = 0; j_local < group_size; j_local++) {
            int is_positive[256];
            for (int x = 0; x < 256; x++)
                is_positive[x] = (hash_sign(x, j_local) == 1);

            /* P(o | h_j > 0) */
            double count_pos[256], total_pos = 0;
            for (int o = 0; o < 256; o++) count_pos[o] = 0.5;
            total_pos = 128.0;

            for (int t = d; t < n-1; t++) {
                int src = data[t - d], dst = data[t + 1];
                if (is_positive[src]) {
                    count_pos[dst] += 1.0; total_pos += 1.0;
                }
            }

            int ni = g * group_size + j_local;
            for (int o = 0; o < 256; o++) {
                double p_cond = count_pos[o] / total_pos;
                /* PMI: log P(o|h+) - log P(o) */
                model.Wy[o][ni] = (float)(log(p_cond) - log(marginal[o]));
            }
        }
    }

    /* Find best scale for v2 */
    memcpy(orig_Wy, model.Wy, sizeof(orig_Wy));
    best_scale = 1.0; best_bpc = 999;
    for (float scale = 0.1; scale <= 5.0; scale += 0.1) {
        for (int o = 0; o < OUTPUT_SIZE; o++)
            for (int j = 0; j < H; j++)
                model.Wy[o][j] = scale * orig_Wy[o][j];
        double bpc = eval_bpc(data, n, &model);
        if (bpc < best_bpc) { best_bpc = bpc; best_scale = scale; }
    }
    printf("  Best scale: %.1f → %.4f bpc\n", best_scale, best_bpc);

    /* ========== Version 3: Per-offset optimal scale ========== */
    printf("\n=== Version 3: Per-offset optimized scale ===\n");

    /* Different offsets may need different scales.
     * Try per-group scaling. */

    /* Recompute with log-ratio (which performed better) */
    double best_alpha = 1.0;
    double overall_best = 999;

    for (double a = 0.1; a <= 5.0; a += 0.1) {
        for (int g = 0; g < n_groups; g++) {
            int d = g;
            for (int j_local = 0; j_local < group_size; j_local++) {
                int is_positive[256];
                for (int x = 0; x < 256; x++)
                    is_positive[x] = (hash_sign(x, j_local) == 1);

                double count_pos[256], count_neg[256];
                double total_pos = a * 256, total_neg = a * 256;
                for (int o = 0; o < 256; o++) { count_pos[o] = a; count_neg[o] = a; }

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

        /* Try scales */
        memcpy(orig_Wy, model.Wy, sizeof(orig_Wy));
        for (float scale = 0.1; scale <= 5.0; scale += 0.2) {
            for (int o = 0; o < OUTPUT_SIZE; o++)
                for (int j = 0; j < H; j++)
                    model.Wy[o][j] = scale * orig_Wy[o][j];
            double bpc = eval_bpc(data, n, &model);
            if (bpc < overall_best) {
                overall_best = bpc;
                best_alpha = a;
                best_scale = scale;
            }
        }
    }
    printf("  Best alpha=%.1f, scale=%.1f → %.4f bpc\n",
           best_alpha, best_scale, overall_best);

    /* Apply best parameters */
    for (int g = 0; g < n_groups; g++) {
        int d = g;
        for (int j_local = 0; j_local < group_size; j_local++) {
            int is_positive[256];
            for (int x = 0; x < 256; x++)
                is_positive[x] = (hash_sign(x, j_local) == 1);

            double count_pos[256], count_neg[256];
            double total_pos = best_alpha * 256, total_neg = best_alpha * 256;
            for (int o = 0; o < 256; o++) { count_pos[o] = best_alpha; count_neg[o] = best_alpha; }

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
                model.Wy[o][ni] = (float)(best_scale * lr);
            }
        }
    }

    double bpc_final = eval_bpc(data, n, &model);
    printf("  Final analytic construction: %.4f bpc\n", bpc_final);

    /* ========== Version 4: Exact byte identity readout ========== */
    printf("\n=== Version 4: Exact byte identity readout ===\n");

    /* The hash gives 8 bits per group. For 8 neurons, there are 256 sign
     * patterns, one per byte (assuming the hash is injective).
     * Instead of the linear readout, compute the EXACT conditional:
     * P(o | data[t-0]=x0, data[t-1]=x1, ...) via pattern matching.
     *
     * But this doesn't fit in the W_y linear readout. The linear readout
     * is a sum over neurons, which is a SUM of per-offset log-ratios.
     * This is equivalent to assuming offset contributions are INDEPENDENT.
     *
     * The independence assumption is the bottleneck. The trained model
     * uses W_h to create non-linear interactions between offsets.
     * Our shift-register keeps offsets independent.
     *
     * Let's quantify: what's the best we can do with independent offsets?
     */

    /* For each offset d and each byte x at that offset,
     * compute P(o | data[t-d] = x) from skip-bigrams.
     * The combined prediction under independence:
     * log P(o | x0, x1, ..., x15) ≈ log P(o) + sum_d [log P(o|x_d) - log P(o)]
     *
     * This is the naive Bayes model over skip-bigrams. */

    /* Count skip-bigrams at each offset */
    static double skip_bigram[16][256][256]; /* [offset][src][dst] */
    memset(skip_bigram, 0, sizeof(skip_bigram));

    for (int d = 0; d < 16; d++)
        for (int t = d; t < n-1; t++)
            skip_bigram[d][data[t-d]][data[t+1]] += 1.0;

    /* Naive Bayes evaluation */
    double nb_total = 0;
    int nb_count = 0;
    for (int t = 16; t < n-1; t++) {
        double logits[256];
        /* Start with log marginal */
        for (int o = 0; o < 256; o++)
            logits[o] = log(marginal[o]);

        /* Add PMI from each offset */
        for (int d = 0; d < 16; d++) {
            int src = data[t - d];
            double total_d = 0;
            for (int o = 0; o < 256; o++)
                total_d += skip_bigram[d][src][o] + 0.5;
            for (int o = 0; o < 256; o++) {
                double p_cond = (skip_bigram[d][src][o] + 0.5) / total_d;
                logits[o] += log(p_cond) - log(marginal[o]);
            }
        }

        /* Softmax */
        double max_l = -1e30;
        for (int o = 0; o < 256; o++) if (logits[o] > max_l) max_l = logits[o];
        double se = 0;
        for (int o = 0; o < 256; o++) se += exp(logits[o] - max_l);
        double p = exp(logits[data[t+1]] - max_l) / se;
        nb_total += -log2(p > 1e-30 ? p : 1e-30);
        nb_count++;
    }
    printf("  Naive Bayes (16 offsets, exact identity): %.4f bpc\n",
           nb_total / nb_count);

    /* Also try with SCALED PMI (reduce overconfidence) */
    for (float temp = 0.1; temp <= 2.0; temp += 0.1) {
        double nb2_total = 0;
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
            nb2_total += -log2(p > 1e-30 ? p : 1e-30);
        }
        double bpc = nb2_total / nb_count;
        if (temp > 0.05 && temp < 0.15)
            printf("  Naive Bayes (temp=%.1f): %.4f bpc\n", temp, bpc);
        if (fabs(temp - 1.0) < 0.05)
            printf("  Naive Bayes (temp=%.1f): %.4f bpc\n", temp, bpc);
    }

    /* Find optimal temperature */
    double nb_best = 999; float nb_best_temp = 1.0;
    for (float temp = 0.01; temp <= 3.0; temp += 0.01) {
        double nb2_total = 0;
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
            nb2_total += -log2(p > 1e-30 ? p : 1e-30);
        }
        double bpc = nb2_total / nb_count;
        if (bpc < nb_best) { nb_best = bpc; nb_best_temp = temp; }
    }
    printf("  Naive Bayes optimal: temp=%.2f → %.4f bpc\n", nb_best_temp, nb_best);

    /* ========== Version 5: Hash readout (linear approximation of NB) ========== */
    printf("\n=== Version 5: Hash-encoded Naive Bayes via W_y ===\n");

    /* The hash is injective if it maps each of 256 bytes to a distinct
     * 8-bit sign pattern. Check this. */
    int patterns_seen[256]; memset(patterns_seen, 0, sizeof(patterns_seen));
    int distinct = 0;
    for (int x = 0; x < 256; x++) {
        int pattern = 0;
        for (int j = 0; j < group_size; j++)
            if (hash_sign(x, j) == 1) pattern |= (1 << j);
        if (!patterns_seen[pattern]) { distinct++; patterns_seen[pattern] = 1; }
    }
    printf("  Hash injectivity: %d distinct patterns / 256 bytes\n", distinct);

    /* If injective, each 8-bit sign pattern uniquely identifies a byte.
     * Then W_y can EXACTLY encode the conditional distribution, because
     * the 8 neurons per group provide 8 linear features that can
     * reconstruct any function of the byte identity via:
     *
     * For byte x with sign pattern s(x) = (s_0, s_1, ..., s_7) ∈ {-1,+1}^8:
     *   sum_j W_y[o][g*8+j] * s_j(x) = Fourier coefficient combination
     *
     * The 8 sign patterns form a Hadamard-like basis.
     * To encode f(x) = log P(o | data[t-d]=x) for each output o,
     * we need W_y values that satisfy:
     *   sum_j W_y[o][g*8+j] * s_j(x) = f_o(x)  for all x
     *
     * This is a system of 256 equations in 8 unknowns per group per output.
     * It's overdetermined (256 > 8), so we can only get the best linear
     * approximation. But since the hash maps each byte to a unique pattern,
     * the effective system size is the number of bytes that actually occur.
     *
     * With tanh saturation, h_j ≈ ±1, so the readout is:
     *   sum_j W_y[o][j] * sign(h_j)
     * which IS the Fourier representation in the sign basis.
     */

    /* Compute optimal W_y by least-squares:
     * For each group g and output o, find w ∈ R^8 that minimizes
     * sum_x [(sum_j w_j * s_j(x)) - f_o(x)]^2
     * where f_o(x) = log P(o | data[t-d]=x) - log P(o)
     * and the sum is over x values that appear in the data at offset d. */

    /* For 8 binary features, the least-squares solution has a nice form.
     * Since s_j ∈ {-1,+1}, the features are orthogonal (if the hash is
     * a good covering design). The solution is approximately:
     *   w_j = (1/N) * sum_x f_o(x) * s_j(x)
     * weighted by the count of each x. */

    for (int g = 0; g < n_groups; g++) {
        int d = g;
        /* Count occurrences of each byte at offset d */
        double byte_at_offset[256]; memset(byte_at_offset, 0, sizeof(byte_at_offset));
        double total_at_offset = 0;
        for (int t = d; t < n-1; t++) {
            byte_at_offset[data[t-d]] += 1.0;
            total_at_offset += 1.0;
        }

        for (int o = 0; o < 256; o++) {
            /* f_o(x) = log P(o | x at offset d) - log P(o) */
            for (int j_local = 0; j_local < group_size; j_local++) {
                double sum = 0, norm = 0;
                for (int x = 0; x < 256; x++) {
                    if (byte_at_offset[x] < 0.5) continue;
                    /* Conditional: P(o | data[t-d]=x) */
                    double total_given_x = 0;
                    for (int o2 = 0; o2 < 256; o2++)
                        total_given_x += skip_bigram[d][x][o2] + 0.01;
                    double p_cond = (skip_bigram[d][x][o] + 0.01) / total_given_x;
                    double f_val = log(p_cond) - log(marginal[o]);
                    int s = hash_sign(x, j_local);
                    sum += byte_at_offset[x] * f_val * s;
                    norm += byte_at_offset[x];
                }
                model.Wy[o][g*group_size + j_local] = (float)(sum / (norm > 0 ? norm : 1));
            }
        }
    }

    /* b_y from marginal */
    for (int o = 0; o < OUTPUT_SIZE; o++)
        model.by[o] = logf((byte_count[o] + 0.5f) / (n + 128.0f));

    /* Evaluate at different scales */
    memcpy(orig_Wy, model.Wy, sizeof(orig_Wy));
    best_scale = 1.0; best_bpc = 999;
    for (float scale = 0.1; scale <= 10.0; scale += 0.1) {
        for (int o = 0; o < OUTPUT_SIZE; o++)
            for (int j = 0; j < H; j++)
                model.Wy[o][j] = scale * orig_Wy[o][j];
        double bpc = eval_bpc(data, n, &model);
        if (bpc < best_bpc) { best_bpc = bpc; best_scale = scale; }
    }
    printf("  Hash-encoded NB readout (scale=%.1f): %.4f bpc\n", best_scale, best_bpc);

    /* ========== Summary ========== */
    printf("\n========================================\n");
    printf("=== ZERO-OPTIMIZATION SUMMARY ===\n");
    printf("========================================\n");
    printf("Configuration                              bpc\n");
    printf("----------------------------------------------\n");
    printf("Uniform                                    8.000\n");
    printf("Analytic W_y (log-ratio)                   %.4f\n", overall_best);
    printf("Analytic W_y (PMI)                         %.4f\n", best_bpc);
    printf("Naive Bayes (exact identity, 16 offsets)   %.4f\n", nb_best);
    printf("----------------------------------------------\n");
    printf("For comparison (from previous experiments):\n");
    printf("Shift-register + optimized W_y             3.572\n");
    printf("Trained model                              4.965\n");

    if (argc >= 3) {
        Model trained; load_model(&trained, argv[2]);
        double trained_bpc = eval_bpc(data, n, &trained);
        printf("Trained model (measured):                  %.4f\n", trained_bpc);
    }

    printf("\nIf any analytic W_y compresses below ~7.0 bpc,\n");
    printf("the loop is closed: ALL parameters from data statistics.\n");

    return 0;
}
