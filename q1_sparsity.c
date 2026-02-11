/*
 * q1_sparsity.c — Q1: How sparse is the explanation?
 *
 * For each prediction of the sat-rnn, count how many UM patterns
 * participate in the backward attribution chain above threshold tau.
 *
 * Protocol A (f32): backward trace through the trained RNN.
 *
 * Usage: q1_sparsity <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 1100
#define D_MAX 50

typedef struct {
    float Wx[HIDDEN_SIZE][INPUT_SIZE];
    float Wh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bh[HIDDEN_SIZE];
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
} Model;

void load_model(Model* m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("load_model"); exit(1); }
    fread(m->Wx, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
    fread(m->Wh, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    fread(m->bh, sizeof(float), HIDDEN_SIZE, f);
    fread(m->Wy, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(m->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

/* Pattern tracking.
 * We identify each pattern by type + indices:
 *   Wx: (byte, neuron) -> id = byte * 128 + neuron          [0, 32767]
 *   Wh: (j, k)         -> id = 32768 + j * 128 + k          [32768, 49151]
 *   Wy: (neuron, byte)  -> id = 49152 + neuron * 256 + byte  [49152, 81919]
 * Total: 81920 possible patterns.
 * We track max attribution per pattern per position.
 */
#define NUM_WX (INPUT_SIZE * HIDDEN_SIZE)
#define NUM_WH (HIDDEN_SIZE * HIDDEN_SIZE)
#define NUM_WY (HIDDEN_SIZE * OUTPUT_SIZE)
#define NUM_PATTERNS (NUM_WX + NUM_WH + NUM_WY)

#define WX_ID(byte, neuron) ((byte) * HIDDEN_SIZE + (neuron))
#define WH_ID(j, k) (NUM_WX + (j) * HIDDEN_SIZE + (k))
#define WY_ID(neuron, byte) (NUM_WX + NUM_WH + (neuron) * OUTPUT_SIZE + (byte))

/* Significance threshold for counting "existing" patterns */
#define SIG_EPSILON 0.01f

#define NUM_THRESHOLDS 7

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data_file> <model_file>\n", argv[0]);
        return 1;
    }

    /* Load data */
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("data"); return 1; }
    unsigned char data[MAX_DATA];
    int n = fread(data, 1, MAX_DATA, f);
    fclose(f);
    if (n > 1024) n = 1024;

    /* Load model */
    Model m;
    load_model(&m, argv[2]);

    /* Count significant patterns */
    int sig_wx = 0, sig_wh = 0, sig_wy = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++)
        for (int b = 0; b < INPUT_SIZE; b++)
            if (fabsf(m.Wx[j][b]) > SIG_EPSILON) sig_wx++;
    for (int j = 0; j < HIDDEN_SIZE; j++)
        for (int k = 0; k < HIDDEN_SIZE; k++)
            if (fabsf(m.Wh[k][j]) > SIG_EPSILON) sig_wh++;
    for (int j = 0; j < HIDDEN_SIZE; j++)
        for (int b = 0; b < OUTPUT_SIZE; b++)
            if (fabsf(m.Wy[b][j]) > SIG_EPSILON) sig_wy++;
    int sig_total = sig_wx + sig_wh + sig_wy;
    printf("Significant patterns (|w| > %.3f): Wx=%d Wh=%d Wy=%d total=%d\n\n",
           SIG_EPSILON, sig_wx, sig_wh, sig_wy, sig_total);

    /* Forward pass: store all h_t and P_t */
    float h_all[MAX_DATA][HIDDEN_SIZE];
    double P_all[MAX_DATA][OUTPUT_SIZE];
    float h[HIDDEN_SIZE];
    memset(h, 0, sizeof(h));

    double total_bpc = 0;
    for (int t = 0; t < n; t++) {
        float h_new[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = m.bh[i] + m.Wx[i][data[t]];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += m.Wh[i][j] * h[j];
            h_new[i] = tanhf(sum);
        }
        memcpy(h, h_new, sizeof(h));
        memcpy(h_all[t], h_new, sizeof(h_new));

        /* Compute softmax */
        if (t < n - 1) {
            double max_logit = -1e30;
            double logits[OUTPUT_SIZE];
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                double s = m.by[o];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    s += m.Wy[o][j] * h_new[j];
                logits[o] = s;
                if (s > max_logit) max_logit = s;
            }
            double sum_exp = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                P_all[t][o] = exp(logits[o] - max_logit);
                sum_exp += P_all[t][o];
            }
            for (int o = 0; o < OUTPUT_SIZE; o++)
                P_all[t][o] /= sum_exp;

            int y = data[t + 1];
            total_bpc -= log(P_all[t][y]);
        }
    }
    total_bpc /= (n - 1) * log(2);
    printf("Forward pass: %d positions, %.4f bpc\n\n", n, total_bpc);

    /* Threshold values */
    float thresholds[NUM_THRESHOLDS] = {1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.0f, 10.0f};

    /* Per-threshold aggregates */
    long long sum_nx[NUM_THRESHOLDS] = {0};
    long long sum_nh[NUM_THRESHOLDS] = {0};
    long long sum_ny[NUM_THRESHOLDS] = {0};
    long long sum_n[NUM_THRESHOLDS] = {0};
    int min_n[NUM_THRESHOLDS], max_n[NUM_THRESHOLDS];
    for (int i = 0; i < NUM_THRESHOLDS; i++) {
        min_n[i] = NUM_PATTERNS;
        max_n[i] = 0;
    }

    /* Per-position counts for median */
    int pos_counts[1024][NUM_THRESHOLDS];

    /* Depth profile: total attribution mass by offset */
    double depth_mass[D_MAX + 1] = {0};
    double depth_count[D_MAX + 1] = {0};

    /* Pattern-level: how many positions is each pattern active at? */
    /* (for "never active" count) */
    int pattern_ever_active[NUM_PATTERNS];
    memset(pattern_ever_active, 0, sizeof(pattern_ever_active));

    /* Per-pattern attribution for current position */
    float pat_attr[NUM_PATTERNS];

    /* Main loop */
    for (int t = 0; t < n - 1; t++) {
        int y = data[t + 1];

        memset(pat_attr, 0, sizeof(pat_attr));

        /* Output gradient g_t */
        float g[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double gj = m.Wy[y][j];
            for (int o = 0; o < OUTPUT_SIZE; o++)
                gj -= m.Wy[o][j] * P_all[t][o];
            g[j] = (float)gj;
        }

        /* W_y attributions: |g_t[j]| for each neuron j.
         * This scores neuron j's contribution to predicting y.
         * The specific pattern is (j, y). */
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float a = fabsf(g[j]);
            int id = WY_ID(j, y);
            if (a > pat_attr[id]) pat_attr[id] = a;
        }

        /* Depth 0 mass (output layer) */
        float d0_mass = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            d0_mass += fabsf(g[j]);
        depth_mass[0] += d0_mass;
        depth_count[0] += 1;

        /* Backward trace */
        float back_g[HIDDEN_SIZE];
        memcpy(back_g, g, sizeof(g));

        for (int d = 1; d <= D_MAX && t - d >= 0; d++) {
            int s = t - d + 1; /* Jacobian at step s */

            /* J_s^T back_g: first scale by saturation gate */
            float scaled[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                scaled[j] = (1.0f - h_all[s][j] * h_all[s][j]) * back_g[j];

            /* W_h attributions at this step:
             * pattern (j, k) has attribution |J_s[k,j] * back_g_prev[k]|
             * = |(1 - h_j(s)^2) * W_h[k,j] * back_g_prev[k]|
             * Wait — need to be careful about indexing.
             *
             * J_s[k,j] = (1 - h_j(s)^2) * W_h[k,j]
             * The pattern (j,k) means neuron j -> neuron k.
             * Its attribution = |J_s[k,j] * g_{arriving at k}|
             *
             * But back_g is the gradient that has arrived at step s+1.
             * After scaling by the gate, scaled[j] = gate_j * back_g[j].
             * The W_h matmul then distributes: new_back[j] = sum_k W_h[k,j] * scaled[k]
             *
             * So the attribution of pattern (k -> j) at this step is:
             * |W_h[j,k] * scaled[k]| = |W_h[j,k] * (1-h_k(s)^2) * back_g[k]|
             *
             * Hmm, let's be precise. m.Wh[i][j] means W_h[i,j].
             * In the RNN: h_new[i] = tanh(... + sum_j W_h[i][j] * h[j])
             * So W_h[i,j] connects h_j -> h_i (j influences i).
             * Pattern (j, i) = "neuron j excites/inhibits neuron i".
             * J_s[i,j] = (1 - h_j(s)^2) * W_h[i,j]
             * No wait: J = diag(1-h^2) * W_h
             * J[i,j] = (1 - h_i(s)^2) * W_h[i,j]
             * (the diagonal scales the rows)
             *
             * So J[i,j] = gate_i * W_h[i,j].
             * Pattern: j -> i (column j, row i).
             * In the transpose: J^T[j,i] = J[i,j] = gate_i * W_h[i,j].
             * back_g_new[j] = sum_i J^T[j,i] * back_g[i] = sum_i gate_i * W_h[i,j] * back_g[i]
             *               = sum_i scaled[i] * W_h[i,j]  (where scaled[i] = gate_i * back_g[i])
             * Wait, that's not matching the existing code. Let me check:
             * sparse_diff.c line 200: sum += m.Wh[i][j] * scaled[i]
             * So new_back[j] = sum_i m.Wh[i][j] * scaled[i]
             * And scaled[i] = (1 - h[s][i]^2) * back_g[i]
             * This means: new_back[j] = sum_i (1-h_i^2) * W_h[i,j] * back_g[i]
             *
             * The per-pattern attribution of W_h pattern (j -> i), i.e. m.Wh[i][j]:
             * it contributes |m.Wh[i][j] * scaled[i]| to new_back[j].
             * But this is only meaningful as part of the chain.
             * We use: a_Wh(j->i, d) = |scaled[i] * m.Wh[i][j]|
             * = |(1-h_i(s)^2) * back_g[i] * W_h[i,j]|
             */
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                if (fabsf(scaled[i]) < 1e-8f) continue; /* gate killed */
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    float a = fabsf(scaled[i] * m.Wh[i][j]);
                    /* Pattern: j -> i, so ID uses (j, i) */
                    int id = WH_ID(j, i);
                    if (a > pat_attr[id]) pat_attr[id] = a;
                }
            }

            /* Propagate: new_back[j] = sum_i W_h[i][j] * scaled[i] */
            float new_back[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float sum = 0;
                for (int i = 0; i < HIDDEN_SIZE; i++)
                    sum += m.Wh[i][j] * scaled[i];
                new_back[j] = sum;
            }
            memcpy(back_g, new_back, sizeof(new_back));

            /* W_x attributions at offset d */
            int input_pos = t - d;
            if (input_pos >= 0) {
                int b = data[input_pos];
                float d_mass = 0;
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    float a = fabsf(m.Wx[j][b] * back_g[j]);
                    int id = WX_ID(b, j);
                    if (a > pat_attr[id]) pat_attr[id] = a;
                    d_mass += a;
                }
                depth_mass[d] += d_mass;
                depth_count[d] += 1;
            }
        }

        /* Count active patterns at each threshold */
        for (int ti = 0; ti < NUM_THRESHOLDS; ti++) {
            float tau = thresholds[ti];
            int nx = 0, nh = 0, ny = 0;

            for (int id = 0; id < NUM_WX; id++)
                if (pat_attr[id] > tau) nx++;
            for (int id = NUM_WX; id < NUM_WX + NUM_WH; id++)
                if (pat_attr[id] > tau) nh++;
            for (int id = NUM_WX + NUM_WH; id < NUM_PATTERNS; id++)
                if (pat_attr[id] > tau) ny++;

            int total = nx + nh + ny;
            sum_nx[ti] += nx;
            sum_nh[ti] += nh;
            sum_ny[ti] += ny;
            sum_n[ti] += total;
            if (total < min_n[ti]) min_n[ti] = total;
            if (total > max_n[ti]) max_n[ti] = total;
            pos_counts[t][ti] = total;
        }

        /* Track which patterns are ever active (at tau = 0.01) */
        for (int id = 0; id < NUM_PATTERNS; id++)
            if (pat_attr[id] > 0.01f) pattern_ever_active[id] = 1;
    }

    /* ============================================================
     * Report: Sparsity distribution
     * ============================================================ */
    printf("=== Sparsity Distribution ===\n\n");
    printf("  tau       mean_n   median   min    max    n/sig\n");
    for (int ti = 0; ti < NUM_THRESHOLDS; ti++) {
        /* Compute median */
        int counts_sorted[1024];
        int np = n - 1;
        memcpy(counts_sorted, pos_counts, np * sizeof(int));
        /* sort for median — just column ti */
        for (int a = 0; a < np; a++)
            counts_sorted[a] = pos_counts[a][ti];
        for (int a = 0; a < np - 1; a++)
            for (int b = a + 1; b < np; b++)
                if (counts_sorted[b] < counts_sorted[a]) {
                    int tmp = counts_sorted[a];
                    counts_sorted[a] = counts_sorted[b];
                    counts_sorted[b] = tmp;
                }
        int median = counts_sorted[np / 2];

        float mean = (float)sum_n[ti] / (n - 1);
        printf("  %-8.0e  %6.1f   %5d   %5d  %5d  %5.3f\n",
               thresholds[ti], mean, median, min_n[ti], max_n[ti],
               mean / sig_total);
    }

    /* ============================================================
     * Report: Breakdown by pattern class
     * ============================================================ */
    printf("\n=== Breakdown by Pattern Class ===\n\n");
    printf("  tau       mean_Wx  mean_Wh  mean_Wy\n");
    for (int ti = 0; ti < NUM_THRESHOLDS; ti++) {
        printf("  %-8.0e  %6.1f   %6.1f   %6.1f\n",
               thresholds[ti],
               (float)sum_nx[ti] / (n - 1),
               (float)sum_nh[ti] / (n - 1),
               (float)sum_ny[ti] / (n - 1));
    }

    /* ============================================================
     * Report: Never-active patterns
     * ============================================================ */
    int never_wx = 0, never_wh = 0, never_wy = 0;
    for (int id = 0; id < NUM_WX; id++)
        if (!pattern_ever_active[id]) never_wx++;
    for (int id = NUM_WX; id < NUM_WX + NUM_WH; id++)
        if (!pattern_ever_active[id]) never_wh++;
    for (int id = NUM_WX + NUM_WH; id < NUM_PATTERNS; id++)
        if (!pattern_ever_active[id]) never_wy++;
    printf("\n=== Never-Active Patterns (tau=0.01) ===\n");
    printf("  Wx: %d/%d never active\n", never_wx, NUM_WX);
    printf("  Wh: %d/%d never active\n", never_wh, NUM_WH);
    printf("  Wy: %d/%d never active\n", never_wy, NUM_WY);
    printf("  Total: %d/%d never active\n",
           never_wx + never_wh + never_wy, NUM_PATTERNS);

    /* ============================================================
     * Report: Depth profile
     * ============================================================ */
    printf("\n=== Depth Profile: Mean Attribution Mass by Offset ===\n\n");
    printf("  d    mean_mass    fraction_of_d0\n");
    double d0_mean = (depth_count[0] > 0) ? depth_mass[0] / depth_count[0] : 1;
    for (int d = 0; d <= D_MAX && depth_count[d] > 0; d++) {
        double dm = depth_mass[d] / depth_count[d];
        printf("  %-3d  %10.4f    %.4f\n", d, dm, dm / d0_mean);
    }

    return 0;
}
