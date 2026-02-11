/*
 * q1_bool_attractor.c — Attractor landscape of the Boolean dynamics.
 *
 * The sat-rnn is effectively a 128-bit Boolean automaton:
 *   sigma_{t+1} = f(sigma_t, x_t)
 *
 * We analyze:
 * 1. Influence graph: for each (j,i), does flipping sigma_j change sigma_i?
 *    Average over all positions to get a weighted influence matrix.
 * 2. Neuron communities: cluster neurons by influence pattern.
 * 3. Convergence: start from random states, run with fixed input, measure basin.
 * 4. Boolean attribution: for each prediction, trace which sign bits matter
 *    through W_y, then which sign bits at t-1 influenced those through W_h.
 *
 * Usage: q1_bool_attractor <data_file> <model_file>
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

/* Boolean step: state sigma -> sigma' given input byte x */
void bool_step(int* out, int* in, int x, Model* m) {
    for (int i = 0; i < H; i++) {
        float z = m->bh[i] + m->Wx[i][x];
        for (int j = 0; j < H; j++)
            z += m->Wh[i][j] * (in[j] ? 1.0f : -1.0f);
        out[i] = (z >= 0) ? 1 : 0;
    }
}

/* Full RNN step with tanh */
void rnn_step(float* out, float* in, int x, Model* m) {
    for (int i = 0; i < H; i++) {
        float z = m->bh[i] + m->Wx[i][x];
        for (int j = 0; j < H; j++) z += m->Wh[i][j]*in[j];
        out[i] = tanhf(z);
    }
}

int hamming(int* a, int* b) {
    int d = 0;
    for (int j = 0; j < H; j++) if (a[j] != b[j]) d++;
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    /* ===== 1. Run full RNN to get ground-truth sign trajectories ===== */
    printf("=== Ground-Truth Sign Trajectory (from full RNN) ===\n");
    float h_full[H]; memset(h_full, 0, sizeof(h_full));
    int sigma_traj[1024][H]; /* sign vectors at each position */

    for (int t = 0; t < n - 1; t++) {
        float hn[H];
        rnn_step(hn, h_full, data[t], &m);
        memcpy(h_full, hn, sizeof(h_full));
        for (int j = 0; j < H; j++)
            sigma_traj[t][j] = (h_full[j] >= 0) ? 1 : 0;
    }

    /* ===== 2. Influence graph: average over positions ===== */
    printf("\n=== Influence Graph (averaged over positions) ===\n");
    /* influence[j][i] = fraction of positions where flipping sigma_j changes sigma'_i */
    int influence_count[H][H]; memset(influence_count, 0, sizeof(influence_count));
    int num_positions_used = 0;

    for (int t = 10; t < n - 2; t += 3) { /* sample positions, skip early transient */
        int* sigma = sigma_traj[t];
        int s_base[H];
        bool_step(s_base, sigma, data[t+1], &m);

        for (int j_flip = 0; j_flip < H; j_flip++) {
            int sigma_flip[H];
            memcpy(sigma_flip, sigma, sizeof(int)*H);
            sigma_flip[j_flip] ^= 1;

            int s_flip[H];
            bool_step(s_flip, sigma_flip, data[t+1], &m);

            for (int i = 0; i < H; i++)
                if (s_base[i] != s_flip[i])
                    influence_count[j_flip][i]++;
        }
        num_positions_used++;
    }

    /* Find top influence edges */
    printf("Top 20 influence edges (source -> target, fraction):\n");
    typedef struct { int j; int i; float frac; } Edge;
    Edge edges[H*H];
    int ne = 0;
    float total_influence = 0;
    for (int j = 0; j < H; j++)
        for (int i = 0; i < H; i++) {
            edges[ne].j = j; edges[ne].i = i;
            edges[ne].frac = (float)influence_count[j][i] / num_positions_used;
            total_influence += edges[ne].frac;
            ne++;
        }
    /* Sort by fraction descending */
    for (int a = 0; a < 20; a++)
        for (int b = a+1; b < ne; b++)
            if (edges[b].frac > edges[a].frac) { Edge t = edges[a]; edges[a] = edges[b]; edges[b] = t; }

    for (int a = 0; a < 20 && a < ne; a++)
        printf("  h%-3d -> h%-3d : %.3f\n", edges[a].j, edges[a].i, edges[a].frac);

    printf("Mean influence per edge: %.4f\n", total_influence / (H*H));

    /* Per-neuron: out-degree and in-degree */
    printf("\nPer-neuron influence (out-degree = how many neurons I affect, in-degree = how many affect me):\n");
    printf("  j  out_deg  in_deg  self_influence\n");
    float out_deg[H], in_deg[H];
    for (int j = 0; j < H; j++) {
        out_deg[j] = 0; in_deg[j] = 0;
        for (int i = 0; i < H; i++) {
            out_deg[j] += (float)influence_count[j][i] / num_positions_used;
            in_deg[j] += (float)influence_count[i][j] / num_positions_used;
        }
    }
    for (int j = 0; j < H; j++) {
        float self = (float)influence_count[j][j] / num_positions_used;
        if (j < 10 || out_deg[j] > 10 || in_deg[j] > 10)
            printf("  %-3d  %6.2f   %6.2f   %.3f\n", j, out_deg[j], in_deg[j], self);
    }

    /* ===== 3. Convergence from random states ===== */
    printf("\n=== Convergence from Random States ===\n");
    printf("Fix input to data[42] = '%c' (0x%02x), start from 50 random states, run 100 steps\n",
           data[42] >= 32 && data[42] < 127 ? data[42] : '.', data[42]);

    srand(42);
    int final_states[50][H];
    int converge_step[50];

    for (int trial = 0; trial < 50; trial++) {
        int sigma[H];
        for (int j = 0; j < H; j++) sigma[j] = rand() & 1;

        int prev_d = 999;
        converge_step[trial] = 100;
        for (int step = 0; step < 100; step++) {
            int s_new[H];
            bool_step(s_new, sigma, data[42], &m);
            int d = hamming(s_new, sigma);
            if (d == 0 && step < converge_step[trial]) converge_step[trial] = step;
            memcpy(sigma, s_new, sizeof(sigma));
        }
        memcpy(final_states[trial], sigma, sizeof(sigma));
    }

    /* How many unique final states? */
    int unique_finals = 0;
    int basin_sizes[50]; memset(basin_sizes, 0, sizeof(basin_sizes));
    int final_class[50]; memset(final_class, -1, sizeof(final_class));

    for (int i = 0; i < 50; i++) {
        if (final_class[i] >= 0) continue;
        final_class[i] = unique_finals;
        basin_sizes[unique_finals] = 1;
        for (int j = i+1; j < 50; j++) {
            if (final_class[j] >= 0) continue;
            if (hamming(final_states[i], final_states[j]) == 0) {
                final_class[j] = unique_finals;
                basin_sizes[unique_finals]++;
            }
        }
        unique_finals++;
    }

    printf("Unique final states: %d / 50 trials\n", unique_finals);
    for (int c = 0; c < unique_finals && c < 10; c++)
        printf("  Attractor %d: %d trials (%.0f%%)\n", c, basin_sizes[c], 100.0*basin_sizes[c]/50);

    /* Convergence speed */
    int sum_conv = 0;
    for (int i = 0; i < 50; i++) sum_conv += converge_step[i];
    printf("Mean convergence step: %.1f (100 = didn't converge)\n", (float)sum_conv/50);

    /* Also: fixed-input cycles? Run one state for 200 steps, look for period */
    printf("\n=== Cycle Detection (fixed input) ===\n");
    {
        int sigma[H];
        for (int j = 0; j < H; j++) sigma[j] = rand() & 1;
        /* Warm up */
        for (int step = 0; step < 100; step++) {
            int s_new[H]; bool_step(s_new, sigma, data[42], &m);
            memcpy(sigma, s_new, sizeof(sigma));
        }
        /* Record trajectory */
        int traj[200][H];
        for (int step = 0; step < 200; step++) {
            int s_new[H]; bool_step(s_new, sigma, data[42], &m);
            memcpy(sigma, s_new, sizeof(sigma));
            memcpy(traj[step], sigma, sizeof(sigma));
        }
        /* Check for period */
        int period = 0;
        for (int p = 1; p <= 100; p++) {
            if (hamming(traj[199], traj[199-p]) == 0) { period = p; break; }
        }
        printf("Cycle period (from warm start): %d %s\n", period, period ? "" : "(no cycle ≤ 100)");
    }

    /* ===== 4. Boolean attribution for one prediction ===== */
    printf("\n=== Boolean Attribution at t=42 ===\n");

    /* Compute output distribution from sign vector at t=42 */
    int* sigma42 = sigma_traj[42];
    float h_sign[H];
    for (int j = 0; j < H; j++) h_sign[j] = sigma42[j] ? 1.0f : -1.0f;

    double P_base[OUTPUT_SIZE], max_l = -1e30;
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double s = m.by[o];
        for (int j = 0; j < H; j++) s += m.Wy[o][j]*h_sign[j];
        P_base[o] = s; if (s > max_l) max_l = s;
    }
    double se = 0;
    for (int o = 0; o < OUTPUT_SIZE; o++) { P_base[o] = exp(P_base[o]-max_l); se += P_base[o]; }
    for (int o = 0; o < OUTPUT_SIZE; o++) P_base[o] /= se;

    int y = data[43]; /* true next byte */
    printf("True next byte: '%c' (0x%02x), P(y) = %.4f, bpc = %.3f\n",
           y >= 32 && y < 127 ? y : '.', y, P_base[y], -log2(P_base[y] > 1e-30 ? P_base[y] : 1e-30));

    /* Attribution: which sign bits matter for this prediction? */
    /* Flip each bit and measure delta log-prob */
    printf("\nPer-neuron attribution (delta bpc when flipping sign bit j):\n");
    printf("  j  delta_bpc  Wy_contrib  sign  W_h_top_source\n");

    typedef struct { int j; float delta; float wy_contrib; } Attrib;
    Attrib attr[H];

    for (int j = 0; j < H; j++) {
        float h_flip[H];
        memcpy(h_flip, h_sign, sizeof(h_sign));
        h_flip[j] = -h_flip[j]; /* flip one sign bit */

        double P_flip[OUTPUT_SIZE]; max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m.by[o];
            for (int k = 0; k < H; k++) s += m.Wy[o][k]*h_flip[k];
            P_flip[o] = s; if (s > max_l) max_l = s;
        }
        se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P_flip[o] = exp(P_flip[o]-max_l); se += P_flip[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P_flip[o] /= se;

        double bpc_flip = -log2(P_flip[y] > 1e-30 ? P_flip[y] : 1e-30);
        double bpc_base = -log2(P_base[y] > 1e-30 ? P_base[y] : 1e-30);

        attr[j].j = j;
        attr[j].delta = bpc_flip - bpc_base;
        attr[j].wy_contrib = m.Wy[y][j] * h_sign[j]; /* direct contribution to logit */
    }

    /* Sort by |delta| descending */
    for (int a = 0; a < H-1; a++)
        for (int b = a+1; b < H; b++)
            if (fabsf(attr[b].delta) > fabsf(attr[a].delta)) { Attrib t = attr[a]; attr[a] = attr[b]; attr[b] = t; }

    for (int a = 0; a < 20; a++) {
        int j = attr[a].j;
        /* Find: which neuron at t-1 most influenced this neuron's sign? */
        /* z_j = bh + Wx + sum(Wh*sigma_{t-1}) */
        /* The biggest contributor is the Wh[j][k]*sigma[k] with largest magnitude */
        int* sigma41 = sigma_traj[41];
        int top_k = -1; float top_contrib = 0;
        for (int k = 0; k < H; k++) {
            float c = m.Wh[j][k] * (sigma41[k] ? 1.0f : -1.0f);
            if (fabsf(c) > fabsf(top_contrib)) { top_contrib = c; top_k = k; }
        }
        printf("  %-3d  %+.4f    %+.4f    %+d     h%-3d (%+.2f)\n",
               j, attr[a].delta, attr[a].wy_contrib, sigma42[j] ? 1 : -1, top_k, top_contrib);
    }

    /* ===== 5. Backward chain: top neurons at t=42 -> their sources at t=41 -> t=40 ===== */
    printf("\n=== Backward Attribution Chain (depth 3) ===\n");
    printf("Tracing top 5 neurons backward through Boolean dynamics:\n\n");

    for (int rank = 0; rank < 5; rank++) {
        int j0 = attr[rank].j;
        printf("--- Chain %d: h%d (delta_bpc=%+.4f) ---\n", rank, j0, attr[rank].delta);

        /* t=42: which neurons at t=41 determined h_j0's sign? */
        printf("  t=42 h%-3d ← ", j0);

        /* Compute the pre-activation to find which inputs matter */
        int* sigma41 = sigma_traj[41];
        float z = m.bh[j0] + m.Wx[j0][data[42]];
        float wh_contribs[H];
        for (int k = 0; k < H; k++) {
            wh_contribs[k] = m.Wh[j0][k] * (sigma41[k] ? 1.0f : -1.0f);
            z += wh_contribs[k];
        }
        /* Sort contributions by magnitude */
        int sorted[H]; for (int k = 0; k < H; k++) sorted[k] = k;
        for (int a = 0; a < 3; a++)
            for (int b = a+1; b < H; b++)
                if (fabsf(wh_contribs[sorted[b]]) > fabsf(wh_contribs[sorted[a]]))
                    { int t = sorted[a]; sorted[a] = sorted[b]; sorted[b] = t; }

        printf("z=%.2f, bias=%.2f, Wx=%.2f, top sources: ", z, m.bh[j0], m.Wx[j0][data[42]]);
        for (int a = 0; a < 3; a++)
            printf("h%d(%+.2f) ", sorted[a], wh_contribs[sorted[a]]);
        printf("\n");

        /* t=41: which neurons at t=40 determined the top source's sign? */
        int j1 = sorted[0]; /* most influential at t=41 */
        int* sigma40 = sigma_traj[40];
        float z1 = m.bh[j1] + m.Wx[j1][data[41]];
        printf("  t=41 h%-3d ← ", j1);
        float wh1[H];
        for (int k = 0; k < H; k++) {
            wh1[k] = m.Wh[j1][k] * (sigma40[k] ? 1.0f : -1.0f);
            z1 += wh1[k];
        }
        int sorted1[H]; for (int k = 0; k < H; k++) sorted1[k] = k;
        for (int a = 0; a < 3; a++)
            for (int b = a+1; b < H; b++)
                if (fabsf(wh1[sorted1[b]]) > fabsf(wh1[sorted1[a]]))
                    { int t = sorted1[a]; sorted1[a] = sorted1[b]; sorted1[b] = t; }
        printf("z=%.2f, bias=%.2f, Wx=%.2f, top sources: ", z1, m.bh[j1], m.Wx[j1][data[41]]);
        for (int a = 0; a < 3; a++)
            printf("h%d(%+.2f) ", sorted1[a], wh1[sorted1[a]]);
        printf("\n");

        /* t=40: one more level */
        int j2 = sorted1[0];
        int* sigma39 = sigma_traj[39];
        float z2 = m.bh[j2] + m.Wx[j2][data[40]];
        printf("  t=40 h%-3d ← ", j2);
        float wh2[H];
        for (int k = 0; k < H; k++) {
            wh2[k] = m.Wh[j2][k] * (sigma39[k] ? 1.0f : -1.0f);
            z2 += wh2[k];
        }
        int sorted2[H]; for (int k = 0; k < H; k++) sorted2[k] = k;
        for (int a = 0; a < 3; a++)
            for (int b = a+1; b < H; b++)
                if (fabsf(wh2[sorted2[b]]) > fabsf(wh2[sorted2[a]]))
                    { int t = sorted2[a]; sorted2[a] = sorted2[b]; sorted2[b] = t; }
        printf("z=%.2f, bias=%.2f, Wx=%.2f, top sources: ", z2, m.bh[j2], m.Wx[j2][data[40]]);
        for (int a = 0; a < 3; a++)
            printf("h%d(%+.2f) ", sorted2[a], wh2[sorted2[a]]);
        printf("\n\n");
    }

    /* ===== 6. Sign flip causality: which input bytes cause most flips? ===== */
    printf("=== Input Byte -> Sign Flip Causality ===\n");
    int byte_flip_count[256]; memset(byte_flip_count, 0, sizeof(byte_flip_count));
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));

    for (int t = 1; t < n - 2; t++) {
        int d = hamming(sigma_traj[t], sigma_traj[t-1]);
        byte_flip_count[data[t]] += d;
        byte_count[data[t]]++;
    }

    printf("Top 20 bytes by mean sign flips:\n");
    typedef struct { int byte; float mean_flips; int count; } ByteFlip;
    ByteFlip bf[256];
    for (int x = 0; x < 256; x++) {
        bf[x].byte = x;
        bf[x].count = byte_count[x];
        bf[x].mean_flips = byte_count[x] ? (float)byte_flip_count[x] / byte_count[x] : 0;
    }
    for (int a = 0; a < 20; a++)
        for (int b = a+1; b < 256; b++)
            if (bf[b].mean_flips > bf[a].mean_flips) { ByteFlip t = bf[a]; bf[a] = bf[b]; bf[b] = t; }

    for (int a = 0; a < 20; a++) {
        char c = bf[a].byte;
        printf("  0x%02x '%c'  mean_flips: %.1f  (n=%d)\n",
               bf[a].byte, c >= 32 && c < 127 ? c : '.', bf[a].mean_flips, bf[a].count);
    }

    return 0;
}
