/*
 * q2_offsets.c — Q2: Do the RNN's effective offsets match MI-greedy?
 *
 * From 20260208: MI-greedy skip-4 selects offsets [1,8,20,3] → 0.069 bpc.
 * From 20260209: Factor map shows dominant offset pair (1,7) for 52/128 neurons.
 *
 * Here we measure which offsets the RNN actually uses, in the Boolean dynamics:
 * 1. For each neuron j, find which past input positions most affect its sign.
 *    (Flip the input byte at t-d, run forward, check if sigma_j changes.)
 * 2. Build a "used offset" distribution per neuron and globally.
 * 3. Compare to MI-greedy offsets [1,8,20,3] and factor-map pair (1,7).
 *
 * Usage: q2_offsets <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define H HIDDEN_SIZE
#define MAX_DEPTH 30

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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    /* Run full RNN forward to get h at all positions */
    float h_traj[1024][H];
    float h[H]; memset(h, 0, sizeof(h));
    for (int t = 0; t < n - 1; t++) {
        float hn[H];
        rnn_step(hn, h, data[t], &m);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
    }

    /* For each test position t, flip data[t-d] and re-run forward from t-d,
     * measure sign changes at position t.
     * Do this for d = 1..MAX_DEPTH. */

    printf("=== Q2: RNN Offset Profile ===\n\n");

    /* Global: total sign changes per depth */
    double depth_sign_changes[MAX_DEPTH+1]; memset(depth_sign_changes, 0, sizeof(depth_sign_changes));
    /* Per-neuron per-depth */
    double neuron_depth_changes[H][MAX_DEPTH+1]; memset(neuron_depth_changes, 0, sizeof(neuron_depth_changes));
    /* Also measure output KL per depth */
    double depth_kl[MAX_DEPTH+1]; memset(depth_kl, 0, sizeof(depth_kl));
    int depth_count[MAX_DEPTH+1]; memset(depth_count, 0, sizeof(depth_count));

    /* Sample test positions */
    int test_positions[] = {42, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500};
    int n_test = 0;
    for (int i = 0; i < 13; i++) if (test_positions[i] < n-1) n_test++;

    for (int ti = 0; ti < n_test; ti++) {
        int t = test_positions[ti];

        /* Base output distribution at position t */
        double P_base[OUTPUT_SIZE]; float* h_base = h_traj[t];
        double max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m.by[o];
            for (int j = 0; j < H; j++) s += m.Wy[o][j]*h_base[j];
            P_base[o] = s; if (s > max_l) max_l = s;
        }
        double se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P_base[o] = exp(P_base[o]-max_l); se += P_base[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P_base[o] /= se;

        /* Base signs */
        int sigma_base[H];
        for (int j = 0; j < H; j++) sigma_base[j] = (h_base[j] >= 0) ? 1 : 0;

        for (int d = 1; d <= MAX_DEPTH && t-d >= 0; d++) {
            /* Flip data[t-d] to a random different byte */
            int orig_byte = data[t-d];
            int flip_byte = (orig_byte + 128) & 0xFF; /* just flip high bit */

            /* Re-run from t-d with flipped byte */
            float h_alt[H];
            /* Start from state just before t-d */
            if (t-d > 0)
                memcpy(h_alt, h_traj[t-d-1], sizeof(h_alt));
            else
                memset(h_alt, 0, sizeof(h_alt));

            /* Step at t-d with flipped byte */
            float hn[H];
            rnn_step(hn, h_alt, flip_byte, &m);
            memcpy(h_alt, hn, sizeof(h_alt));

            /* Continue with original bytes from t-d+1 to t */
            for (int s = t-d+1; s <= t; s++) {
                rnn_step(hn, h_alt, data[s], &m);
                memcpy(h_alt, hn, sizeof(h_alt));
            }

            /* Count sign changes */
            int sign_changes = 0;
            for (int j = 0; j < H; j++) {
                int s_alt = (h_alt[j] >= 0) ? 1 : 0;
                if (s_alt != sigma_base[j]) {
                    sign_changes++;
                    neuron_depth_changes[j][d]++;
                }
            }
            depth_sign_changes[d] += sign_changes;

            /* Output KL */
            double P_alt[OUTPUT_SIZE]; max_l = -1e30;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                double s = m.by[o];
                for (int j = 0; j < H; j++) s += m.Wy[o][j]*h_alt[j];
                P_alt[o] = s; if (s > max_l) max_l = s;
            }
            se = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++) { P_alt[o] = exp(P_alt[o]-max_l); se += P_alt[o]; }
            for (int o = 0; o < OUTPUT_SIZE; o++) P_alt[o] /= se;

            double kl = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++)
                if (P_base[o] > 1e-30 && P_alt[o] > 1e-30)
                    kl += P_base[o] * log2(P_base[o] / P_alt[o]);
            depth_kl[d] += kl;
            depth_count[d]++;
        }
    }

    /* Print depth profile */
    printf("Depth profile (averaged over %d test positions):\n", n_test);
    printf("  depth  mean_sign_changes  mean_output_KL\n");
    for (int d = 1; d <= MAX_DEPTH; d++) {
        if (depth_count[d] > 0)
            printf("  %-5d  %8.2f            %8.4f\n",
                   d, depth_sign_changes[d] / depth_count[d],
                   depth_kl[d] / depth_count[d]);
    }

    /* Per-neuron: which depth is most influential? */
    printf("\n=== Per-Neuron Dominant Offset ===\n");
    printf("  j  dom_offset  changes_at_d1  total_changes  profile\n");

    int offset_histogram[MAX_DEPTH+1]; memset(offset_histogram, 0, sizeof(offset_histogram));

    for (int j = 0; j < H; j++) {
        int best_d = 1; double best_v = 0;
        double total = 0;
        for (int d = 1; d <= MAX_DEPTH; d++) {
            if (neuron_depth_changes[j][d] > best_v) {
                best_v = neuron_depth_changes[j][d];
                best_d = d;
            }
            total += neuron_depth_changes[j][d];
        }
        offset_histogram[best_d]++;

        if (j < 20 || best_d > 5) {
            printf("  %-3d  d=%-3d       %.1f            %.1f         ",
                   j, best_d, neuron_depth_changes[j][1], total);
            /* Mini profile */
            for (int d = 1; d <= 10; d++)
                printf("%.0f ", neuron_depth_changes[j][d]);
            printf("\n");
        }
    }

    printf("\n=== Offset Histogram (dominant offset per neuron) ===\n");
    for (int d = 1; d <= MAX_DEPTH; d++)
        if (offset_histogram[d] > 0)
            printf("  d=%2d: %3d neurons (%.1f%%)\n", d, offset_histogram[d],
                   100.0*offset_histogram[d]/H);

    /* Compare to MI-greedy: [1,8,20,3] */
    printf("\n=== Comparison to MI-Greedy Offsets [1,8,20,3] ===\n");
    printf("Fraction of total sign-change signal at MI-greedy offsets:\n");
    double total_all = 0, total_greedy = 0;
    for (int d = 1; d <= MAX_DEPTH; d++) total_all += depth_sign_changes[d];
    int mi_greedy[] = {1, 3, 8, 20};
    for (int i = 0; i < 4; i++) {
        int d = mi_greedy[i];
        if (d <= MAX_DEPTH) {
            printf("  d=%2d: %.2f sign_changes (%.1f%% of total)\n",
                   d, depth_sign_changes[d] / n_test, 100.0*depth_sign_changes[d]/total_all);
            total_greedy += depth_sign_changes[d];
        }
    }
    printf("  MI-greedy [1,3,8,20] total: %.1f%% of all changes\n",
           100.0*total_greedy/total_all);

    /* Factor map pair (1,7): how much signal at offsets 1 and 7? */
    printf("\nFactor-map pair (1,7) signal:\n");
    printf("  d=1: %.1f%%,  d=7: %.1f%%,  combined: %.1f%%\n",
           100.0*depth_sign_changes[1]/total_all,
           100.0*depth_sign_changes[7]/total_all,
           100.0*(depth_sign_changes[1]+depth_sign_changes[7])/total_all);

    return 0;
}
