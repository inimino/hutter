/*
 * q5_redux.c — Q5: The Sat-RNN Redux.
 *
 * Can we build a simpler model that captures the sat-rnn's behavior?
 *
 * Experiments:
 * 1. Prune W_y to top-k neurons → measure bpc vs neuron count.
 * 2. Prune W_h: remove all edges below threshold → measure bpc.
 * 3. Combined: top-k neurons with pruned W_h.
 * 4. Boolean redux: run Boolean dynamics with pruned weights.
 * 5. Sparsity profile: what fraction of W_h can be zeroed?
 *
 * Usage: q5_redux <data_file> <model_file>
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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model orig; load_model(&orig, argv[2]);

    double base_bpc = eval_bpc(data, n, &orig);
    printf("=== Q5: The Sat-RNN Redux ===\n\n");
    printf("Baseline bpc: %.4f\n\n", base_bpc);

    /* ===== 1. Neuron importance ranking (same as Q3) ===== */
    /* Quick knockout to rank neurons */
    double knockout_delta[H];
    for (int j = 0; j < H; j++) {
        Model tmp; memcpy(&tmp, &orig, sizeof(Model));
        for (int o = 0; o < OUTPUT_SIZE; o++) tmp.Wy[o][j] = 0;
        knockout_delta[j] = eval_bpc(data, n, &tmp) - base_bpc;
    }

    /* Sort neurons by importance */
    int sorted[H]; for (int j = 0; j < H; j++) sorted[j] = j;
    for (int a = 0; a < H-1; a++)
        for (int b = a+1; b < H; b++)
            if (knockout_delta[sorted[b]] > knockout_delta[sorted[a]])
                { int t = sorted[a]; sorted[a] = sorted[b]; sorted[b] = t; }

    printf("Top 10 neurons by knockout importance:\n");
    for (int a = 0; a < 10; a++)
        printf("  h%-3d: delta=+%.4f\n", sorted[a], knockout_delta[sorted[a]]);

    /* ===== 2. W_h pruning: zero edges below threshold ===== */
    printf("\n=== W_h Pruning ===\n");
    printf("  threshold  pct_zeroed  bpc      delta\n");

    float thresholds[] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0};
    for (int ti = 0; ti < 11; ti++) {
        float thr = thresholds[ti];
        Model tmp; memcpy(&tmp, &orig, sizeof(Model));

        int zeroed = 0;
        for (int i = 0; i < H; i++)
            for (int j = 0; j < H; j++)
                if (fabsf(tmp.Wh[i][j]) < thr) { tmp.Wh[i][j] = 0; zeroed++; }

        double bpc = eval_bpc(data, n, &tmp);
        printf("  %5.1f      %5.1f%%     %7.4f  %+.4f\n",
               thr, 100.0*zeroed/(H*H), bpc, bpc - base_bpc);
    }

    /* ===== 3. W_x pruning ===== */
    printf("\n=== W_x Pruning ===\n");
    printf("  threshold  pct_zeroed  bpc      delta\n");

    for (int ti = 0; ti < 11; ti++) {
        float thr = thresholds[ti];
        Model tmp; memcpy(&tmp, &orig, sizeof(Model));

        int zeroed = 0;
        for (int i = 0; i < H; i++)
            for (int x = 0; x < INPUT_SIZE; x++)
                if (fabsf(tmp.Wx[i][x]) < thr) { tmp.Wx[i][x] = 0; zeroed++; }

        double bpc = eval_bpc(data, n, &tmp);
        printf("  %5.1f      %5.1f%%     %7.4f  %+.4f\n",
               thr, 100.0*zeroed/(H*INPUT_SIZE), bpc, bpc - base_bpc);
    }

    /* ===== 4. Combined: top-k neurons + W_h pruning ===== */
    printf("\n=== Combined: Top-K Neurons + W_h Pruning ===\n");
    printf("  k  wh_thr  wy_pct_kept  wh_pct_kept  bpc      delta\n");

    int k_values[] = {5, 10, 15, 20, 30, 128};
    float wh_thrs[] = {0, 1.0, 2.0, 3.0};
    for (int ki = 0; ki < 6; ki++) {
        for (int wi = 0; wi < 4; wi++) {
            int k = k_values[ki];
            float wh_thr = wh_thrs[wi];

            Model tmp; memcpy(&tmp, &orig, sizeof(Model));

            /* Zero non-top-k neurons' W_y columns */
            int wy_kept = 0;
            for (int j = 0; j < H; j++) {
                int is_top = 0;
                for (int a = 0; a < k && a < H; a++)
                    if (sorted[a] == j) { is_top = 1; break; }
                if (!is_top)
                    for (int o = 0; o < OUTPUT_SIZE; o++) tmp.Wy[o][j] = 0;
                else wy_kept++;
            }

            /* Zero W_h below threshold */
            int wh_kept = 0, wh_total = H*H;
            for (int i = 0; i < H; i++)
                for (int j = 0; j < H; j++) {
                    if (fabsf(tmp.Wh[i][j]) < wh_thr) tmp.Wh[i][j] = 0;
                    else wh_kept++;
                }

            double bpc = eval_bpc(data, n, &tmp);
            printf("  %3d  %4.1f    %5.1f%%        %5.1f%%       %7.4f  %+.4f\n",
                   k, wh_thr, 100.0*wy_kept/H, 100.0*wh_kept/wh_total, bpc, bpc - base_bpc);
        }
    }

    /* ===== 5. Weight magnitude distribution ===== */
    printf("\n=== Weight Magnitude Distribution ===\n");

    /* W_h */
    float wh_magnitudes[H*H];
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++)
            wh_magnitudes[i*H+j] = fabsf(orig.Wh[i][j]);
    /* Sort */
    for (int a = 0; a < H*H-1; a++)
        for (int b = a+1; b < H*H; b++)
            if (wh_magnitudes[b] < wh_magnitudes[a])
                { float t = wh_magnitudes[a]; wh_magnitudes[a] = wh_magnitudes[b]; wh_magnitudes[b] = t; }

    printf("W_h percentiles:\n");
    int pcts[] = {10, 25, 50, 75, 90, 95, 99};
    for (int i = 0; i < 7; i++) {
        int idx = pcts[i] * H * H / 100;
        printf("  %d%%: %.3f\n", pcts[i], wh_magnitudes[idx]);
    }
    printf("  min: %.3f, max: %.3f, mean: %.3f\n",
           wh_magnitudes[0], wh_magnitudes[H*H-1],
           ({double s=0; for(int i=0;i<H*H;i++) s+=wh_magnitudes[i]; s/(H*H);}));

    /* Total parameter count for the redux */
    printf("\n=== Redux Parameter Count ===\n");
    printf("Full model: %d params\n", H*INPUT_SIZE + H*H + H + OUTPUT_SIZE*H + OUTPUT_SIZE);

    /* Best redux config from above */
    int best_k = 15; float best_wh_thr = 2.0;
    int params_wy = best_k * OUTPUT_SIZE;
    int params_wh = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++)
            if (fabsf(orig.Wh[i][j]) >= best_wh_thr) params_wh++;
    int params_wx = H * INPUT_SIZE; /* keep all Wx for dynamics */
    int params_total = params_wx + params_wh + H + params_wy + OUTPUT_SIZE;
    int full_total = H*INPUT_SIZE + H*H + H + OUTPUT_SIZE*H + OUTPUT_SIZE;
    printf("Redux (k=%d, wh_thr=%.1f): %d params (%.1f%% of full)\n",
           best_k, best_wh_thr, params_total, 100.0*params_total/full_total);
    printf("  W_y: %d (only %d columns)\n", params_wy, best_k);
    printf("  W_h: %d (%.1f%% of full)\n", params_wh, 100.0*params_wh/(H*H));
    printf("  W_x: %d (full, for dynamics)\n", params_wx);

    return 0;
}
