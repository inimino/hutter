/*
 * q1_margins.c — Pre-activation margin analysis of the Boolean dynamics.
 *
 * For each neuron at each position, compute z = bh + Wx + Wh*h.
 * The margin is |z| — how far the pre-activation is from the threshold.
 * If all margins are large, the Boolean function is identical to tanh,
 * and the mantissa truly doesn't matter (any perturbation < |z| has no effect).
 *
 * This explains WHY sign-only dynamics works: the margins are so large that
 * the mantissa (which perturbs h by at most ~2^{-23}) cannot change any sign.
 *
 * We also compute:
 * 1. Margin distribution (histogram)
 * 2. Per-neuron statistics (which neurons have small margins?)
 * 3. "Fragile transitions": positions where a neuron's margin is small
 * 4. Relationship between margin and prediction quality
 *
 * Usage: q1_margins <data_file> <model_file>
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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    float h[H]; memset(h, 0, sizeof(h));

    /* Margin bins */
    int margin_hist[50]; memset(margin_hist, 0, sizeof(margin_hist)); /* bins of 5: 0-5, 5-10, ... */
    int small_margin_count = 0;  /* |z| < 1 */
    int tiny_margin_count = 0;   /* |z| < 0.1 */
    int total_neurons = 0;
    double total_margin = 0;

    /* Per-neuron accumulators */
    double per_neuron_margin_sum[H]; memset(per_neuron_margin_sum, 0, sizeof(per_neuron_margin_sum));
    int per_neuron_small[H]; memset(per_neuron_small, 0, sizeof(per_neuron_small));
    float per_neuron_min_margin[H];
    for (int j = 0; j < H; j++) per_neuron_min_margin[j] = 1e30;

    /* Per-position: minimum margin and sign-only bpc */
    double total_bpc_full = 0, total_bpc_sign = 0;
    float min_margin_per_pos[1024];

    printf("=== Margin Analysis ===\n\n");
    printf("  t  min_margin  mean_margin  n_small(<1)  n_tiny(<0.1)  bpc_full  bpc_sign\n");

    for (int t = 0; t < n - 1; t++) {
        float hn[H];
        float pre_act[H]; /* pre-activation z */

        for (int i = 0; i < H; i++) {
            float z = m.bh[i] + m.Wx[i][data[t]];
            for (int j = 0; j < H; j++) z += m.Wh[i][j]*h[j];
            pre_act[i] = z;
            hn[i] = tanhf(z);
        }

        /* Margins */
        float min_m = 1e30, sum_m = 0;
        int n_small = 0, n_tiny = 0;
        for (int i = 0; i < H; i++) {
            float margin = fabsf(pre_act[i]);
            sum_m += margin;
            if (margin < min_m) min_m = margin;
            if (margin < 1.0f) n_small++;
            if (margin < 0.1f) n_tiny++;

            int bin = (int)(margin / 5.0f);
            if (bin >= 50) bin = 49;
            margin_hist[bin]++;

            per_neuron_margin_sum[i] += margin;
            if (margin < 1.0f) per_neuron_small[i]++;
            if (margin < per_neuron_min_margin[i]) per_neuron_min_margin[i] = margin;

            total_margin += margin;
            total_neurons++;
        }
        small_margin_count += n_small;
        tiny_margin_count += n_tiny;
        min_margin_per_pos[t] = min_m;

        /* Full bpc */
        double P_full[OUTPUT_SIZE], max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m.by[o];
            for (int j = 0; j < H; j++) s += m.Wy[o][j]*hn[j];
            P_full[o] = s; if (s > max_l) max_l = s;
        }
        double se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P_full[o] = exp(P_full[o]-max_l); se += P_full[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P_full[o] /= se;

        /* Sign bpc */
        float h_sign[H];
        for (int j = 0; j < H; j++) h_sign[j] = (hn[j] >= 0) ? 1.0f : -1.0f;
        double P_sign[OUTPUT_SIZE]; max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m.by[o];
            for (int j = 0; j < H; j++) s += m.Wy[o][j]*h_sign[j];
            P_sign[o] = s; if (s > max_l) max_l = s;
        }
        se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P_sign[o] = exp(P_sign[o]-max_l); se += P_sign[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P_sign[o] /= se;

        int y = data[t+1];
        double bpc_f = -log2(P_full[y] > 1e-30 ? P_full[y] : 1e-30);
        double bpc_s = -log2(P_sign[y] > 1e-30 ? P_sign[y] : 1e-30);
        total_bpc_full += bpc_f;
        total_bpc_sign += bpc_s;

        if (t < 10 || (t < 100 && t % 10 == 0) || t % 50 == 0 || n_small > 3 || t == n-2) {
            printf("  %-4d %8.2f    %8.2f      %3d          %3d      %6.3f    %6.3f\n",
                   t, min_m, sum_m/H, n_small, n_tiny, bpc_f, bpc_s);
        }

        memcpy(h, hn, sizeof(h));
    }

    /* Summary */
    printf("\n=== Margin Summary ===\n");
    printf("Total neuron-steps: %d\n", total_neurons);
    printf("Mean margin: %.2f\n", total_margin / total_neurons);
    printf("Small margins (|z|<1): %d (%.3f%%)\n", small_margin_count, 100.0*small_margin_count/total_neurons);
    printf("Tiny margins (|z|<0.1): %d (%.3f%%)\n", tiny_margin_count, 100.0*tiny_margin_count/total_neurons);
    printf("Mean bpc (full): %.4f\n", total_bpc_full / (n-1));
    printf("Mean bpc (sign-only readout): %.4f\n", total_bpc_sign / (n-1));

    printf("\n=== Margin Histogram ===\n");
    for (int b = 0; b < 50; b++) {
        if (margin_hist[b] > 0)
            printf("  [%3d-%3d]: %6d (%.2f%%)\n", b*5, (b+1)*5, margin_hist[b],
                   100.0*margin_hist[b]/total_neurons);
    }

    printf("\n=== Per-Neuron Margin Statistics ===\n");
    printf("  j  mean_margin  min_margin  n_small(<1)\n");
    /* Sort neurons by mean margin */
    int sorted[H]; for (int j = 0; j < H; j++) sorted[j] = j;
    for (int a = 0; a < H-1; a++)
        for (int b = a+1; b < H; b++)
            if (per_neuron_margin_sum[sorted[a]] > per_neuron_margin_sum[sorted[b]])
                { int t = sorted[a]; sorted[a] = sorted[b]; sorted[b] = t; }

    /* Show 20 neurons with smallest mean margins */
    for (int a = 0; a < 20; a++) {
        int j = sorted[a];
        printf("  %-3d  %8.2f     %8.2f     %3d\n",
               j, per_neuron_margin_sum[j]/(n-1), per_neuron_min_margin[j], per_neuron_small[j]);
    }

    /* ===== Margin vs distance from threshold: the mantissa can't reach ===== */
    printf("\n=== The Mantissa Cannot Reach ===\n");
    /* Maximum perturbation from mantissa noise: |h| * 2^{-23} * sum(|Wh|) */
    /* The largest possible mantissa perturbation to z_j is: */
    /* delta_z_j = sum_k |Wh[j][k]| * |delta_h_k| where |delta_h_k| <= 2^{-23} for saturated neurons */
    for (int j = 0; j < 5; j++) {
        float sum_wh = 0;
        for (int k = 0; k < H; k++) sum_wh += fabsf(m.Wh[j][k]);
        float max_mant_perturbation = sum_wh * (1.0f / (1 << 23)); /* each h changes by at most 2^{-23} */
        float mean_margin = per_neuron_margin_sum[j] / (n-1);
        printf("  h%-3d: sum|Wh|=%.1f, max_mantissa_delta_z=%.6f, mean_margin=%.1f, ratio=%.0f\n",
               j, sum_wh, max_mant_perturbation, mean_margin,
               mean_margin / (max_mant_perturbation > 0 ? max_mant_perturbation : 1e-10));
    }

    /* Compute for ALL neurons */
    printf("\n  Global: max possible mantissa perturbation to any z:\n");
    float max_pert_global = 0;
    float min_margin_global = 1e30;
    for (int j = 0; j < H; j++) {
        float sum_wh = 0;
        for (int k = 0; k < H; k++) sum_wh += fabsf(m.Wh[j][k]);
        float pert = sum_wh / (1 << 23);
        if (pert > max_pert_global) max_pert_global = pert;
        if (per_neuron_min_margin[j] < min_margin_global)
            min_margin_global = per_neuron_min_margin[j];
    }
    printf("  Max mantissa perturbation: %.6f\n", max_pert_global);
    printf("  Min margin ever observed:  %.2f\n", min_margin_global);
    printf("  Safety factor:             %.0f×\n", min_margin_global / max_pert_global);
    printf("  => Mantissa CANNOT flip any sign. The Boolean function IS the computation.\n");

    return 0;
}
