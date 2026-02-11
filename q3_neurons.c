/*
 * q3_neurons.c — Q3: Which neurons carry the signal?
 *
 * For each neuron, measure its contribution to prediction quality:
 * 1. Knock out each neuron (set to 0) and measure bpc change.
 * 2. Knock out each neuron (set to sign-only) and measure bpc change.
 * 3. Per-neuron mutual information with output byte.
 * 4. Neuron importance ranking: which neurons matter most?
 * 5. Minimal subset: how few neurons give near-full bpc?
 *
 * Usage: q3_neurons <data_file> <model_file>
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

double compute_bpc(float* h_read, Model* m, int y) {
    double P[OUTPUT_SIZE], max_l = -1e30;
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double s = m->by[o];
        for (int j = 0; j < H; j++) s += m->Wy[o][j]*h_read[j];
        P[o] = s; if (s > max_l) max_l = s;
    }
    double se = 0;
    for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(P[o]-max_l); se += P[o]; }
    for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;
    return -log2(P[y] > 1e-30 ? P[y] : 1e-30);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    /* Run full RNN and store trajectories */
    float h_traj[1024][H];
    float h[H]; memset(h, 0, sizeof(h));
    double total_bpc_base = 0;
    for (int t = 0; t < n-1; t++) {
        float hn[H];
        rnn_step(hn, h, data[t], &m);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
        total_bpc_base += compute_bpc(hn, &m, data[t+1]);
    }
    double base_bpc = total_bpc_base / (n-1);
    printf("=== Q3: Which Neurons Carry the Signal? ===\n\n");
    printf("Baseline bpc: %.4f\n\n", base_bpc);

    /* ===== 1. Knockout: zero each neuron's readout ===== */
    printf("=== Readout Knockout (zero W_y column j) ===\n");
    printf("  j  bpc_knockout  delta_bpc  |W_y_col|  mean|h_j|\n");

    typedef struct { int j; double delta; double wy_norm; double mean_h; } NeuronInfo;
    NeuronInfo info[H];

    for (int j_ko = 0; j_ko < H; j_ko++) {
        /* Temporarily zero out W_y column j_ko */
        float saved_wy[OUTPUT_SIZE];
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            saved_wy[o] = m.Wy[o][j_ko];
            m.Wy[o][j_ko] = 0;
        }

        double total_bpc = 0;
        for (int t = 0; t < n-1; t++)
            total_bpc += compute_bpc(h_traj[t], &m, data[t+1]);

        double ko_bpc = total_bpc / (n-1);
        double delta = ko_bpc - base_bpc;

        /* Restore */
        for (int o = 0; o < OUTPUT_SIZE; o++) m.Wy[o][j_ko] = saved_wy[o];

        /* Compute W_y column norm and mean |h| */
        double wy_norm = 0, mean_h = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) wy_norm += m.Wy[o][j_ko] * m.Wy[o][j_ko];
        wy_norm = sqrt(wy_norm);
        for (int t = 0; t < n-1; t++) mean_h += fabsf(h_traj[t][j_ko]);
        mean_h /= (n-1);

        info[j_ko].j = j_ko;
        info[j_ko].delta = delta;
        info[j_ko].wy_norm = wy_norm;
        info[j_ko].mean_h = mean_h;
    }

    /* Sort by delta descending (largest bpc increase = most important) */
    int sorted[H]; for (int j = 0; j < H; j++) sorted[j] = j;
    for (int a = 0; a < H-1; a++)
        for (int b = a+1; b < H; b++)
            if (info[sorted[b]].delta > info[sorted[a]].delta)
                { int t = sorted[a]; sorted[a] = sorted[b]; sorted[b] = t; }

    for (int a = 0; a < 30; a++) {
        int j = sorted[a];
        printf("  %-3d  %7.4f      %+.4f    %6.2f     %.4f\n",
               j, base_bpc + info[j].delta, info[j].delta, info[j].wy_norm, info[j].mean_h);
    }

    /* ===== 2. Cumulative knockout: remove neurons in order of importance ===== */
    printf("\n=== Cumulative Knockout (remove neurons in importance order) ===\n");
    printf("  n_removed  bpc       delta     removed_neurons\n");

    /* Save full W_y */
    float saved_Wy[OUTPUT_SIZE][H];
    memcpy(saved_Wy, m.Wy, sizeof(m.Wy));

    double prev_bpc = base_bpc;
    for (int k = 0; k <= 128; k++) {
        if (k > 0) {
            /* Zero out the k-th most important neuron */
            int j = sorted[k-1];
            for (int o = 0; o < OUTPUT_SIZE; o++) m.Wy[o][j] = 0;
        }

        if (k <= 20 || k % 10 == 0 || k == 128) {
            double total_bpc = 0;
            for (int t = 0; t < n-1; t++)
                total_bpc += compute_bpc(h_traj[t], &m, data[t+1]);
            double bpc_k = total_bpc / (n-1);

            /* Show first few removed neurons */
            printf("  %3d        %7.4f   %+.4f   ", k, bpc_k, bpc_k - base_bpc);
            if (k > 0 && k <= 10) {
                for (int i = 0; i < k && i < 5; i++) printf("h%d ", sorted[i]);
                if (k > 5) printf("...");
            }
            printf("\n");
        }
    }

    /* Restore W_y */
    memcpy(m.Wy, saved_Wy, sizeof(m.Wy));

    /* ===== 3. Minimal subset: keep only top-k neurons ===== */
    printf("\n=== Minimal Subset (keep only top-k neurons) ===\n");
    printf("  k_kept  bpc       pct_of_gain\n");

    for (int k = 1; k <= 128; k++) {
        if (k <= 20 || k % 10 == 0 || k == 128) {
            /* Zero all except top-k */
            for (int o = 0; o < OUTPUT_SIZE; o++)
                for (int j = 0; j < H; j++) m.Wy[o][j] = 0;
            for (int a = 0; a < k; a++) {
                int j = sorted[a];
                for (int o = 0; o < OUTPUT_SIZE; o++) m.Wy[o][j] = saved_Wy[o][j];
            }

            double total_bpc = 0;
            for (int t = 0; t < n-1; t++)
                total_bpc += compute_bpc(h_traj[t], &m, data[t+1]);
            double bpc_k = total_bpc / (n-1);
            double gain_pct = 100.0 * (8.0 - bpc_k) / (8.0 - base_bpc);

            printf("  %3d      %7.4f   %6.1f%%\n", k, bpc_k, gain_pct);
        }
    }
    /* Restore */
    memcpy(m.Wy, saved_Wy, sizeof(m.Wy));

    /* ===== 4. Sign-only per neuron: which neurons need mantissa? ===== */
    printf("\n=== Per-Neuron Sign-Only Test ===\n");
    printf("(Replace h_j with sgn(h_j) for readout, keep dynamics full)\n");
    printf("  j  bpc_sign_j  delta  mean|h_j|  importance_rank\n");

    double sign_delta[H];
    for (int j_test = 0; j_test < H; j_test++) {
        double total_bpc = 0;
        for (int t = 0; t < n-1; t++) {
            float h_mod[H];
            memcpy(h_mod, h_traj[t], sizeof(h_mod));
            h_mod[j_test] = (h_mod[j_test] >= 0) ? 1.0f : -1.0f;
            total_bpc += compute_bpc(h_mod, &m, data[t+1]);
        }
        sign_delta[j_test] = total_bpc / (n-1) - base_bpc;
    }

    /* Sort by |sign_delta| to find neurons most affected by sign-snap */
    int sign_sorted[H]; for (int j = 0; j < H; j++) sign_sorted[j] = j;
    for (int a = 0; a < H-1; a++)
        for (int b = a+1; b < H; b++)
            if (fabsf(sign_delta[sign_sorted[b]]) > fabsf(sign_delta[sign_sorted[a]]))
                { int t = sign_sorted[a]; sign_sorted[a] = sign_sorted[b]; sign_sorted[b] = t; }

    for (int a = 0; a < 20; a++) {
        int j = sign_sorted[a];
        /* Find importance rank */
        int rank = -1;
        for (int r = 0; r < H; r++) if (sorted[r] == j) { rank = r; break; }
        printf("  %-3d  %7.4f   %+.4f  %.4f      %d\n",
               j, base_bpc + sign_delta[j], sign_delta[j], info[j].mean_h, rank);
    }

    /* ===== 5. W_y column analysis: what does each neuron predict? ===== */
    printf("\n=== Top 10 Neurons: What Do They Predict? ===\n");
    for (int a = 0; a < 10; a++) {
        int j = sorted[a];
        printf("h%-3d (delta=%.4f, |Wy|=%.2f, mean|h|=%.4f):\n",
               j, info[j].delta, info[j].wy_norm, info[j].mean_h);

        /* Top 5 positive and negative W_y entries */
        typedef struct { int o; float w; } WyEntry;
        WyEntry entries[OUTPUT_SIZE];
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            entries[o].o = o;
            entries[o].w = m.Wy[o][j];
        }
        /* Sort by weight */
        for (int x = 0; x < 5; x++)
            for (int y = x+1; y < OUTPUT_SIZE; y++)
                if (entries[y].w > entries[x].w) { WyEntry t = entries[x]; entries[x] = entries[y]; entries[y] = t; }
        printf("  Promotes: ");
        for (int x = 0; x < 5; x++) {
            char c = entries[x].o;
            printf("'%c'(%.1f) ", c >= 32 && c < 127 ? c : '.', entries[x].w);
        }
        printf("\n");

        /* Sort by weight ascending */
        for (int x = 0; x < OUTPUT_SIZE-1; x++)
            for (int y = x+1; y < OUTPUT_SIZE; y++)
                if (entries[y].w < entries[x].w) { WyEntry t = entries[x]; entries[x] = entries[y]; entries[y] = t; }
        printf("  Demotes:  ");
        for (int x = 0; x < 5; x++) {
            char c = entries[x].o;
            printf("'%c'(%.1f) ", c >= 32 && c < 127 ? c : '.', entries[x].w);
        }
        printf("\n");
    }

    return 0;
}
