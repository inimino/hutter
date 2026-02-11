/*
 * q4_saturation.c — Q4: Saturation structure (settled vs active neurons).
 *
 * At each position, classify neurons:
 * - Settled: |h| >= 0.999, sign hasn't changed for k steps
 * - Active: sign changed in last few steps
 * - Transitioning: was settled, now changing
 *
 * Also: for each neuron, compute:
 * - Time since last sign change
 * - Distribution of "dwell times" (how long between sign changes)
 * - Phase diagram: which neurons are always settled, always active, or mixed?
 *
 * Usage: q4_saturation <data_file> <model_file>
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

    /* Per-neuron tracking */
    int sign[H]; memset(sign, 0, sizeof(sign)); /* current sign */
    int last_flip[H]; /* last position where sign changed */
    int n_flips[H]; memset(n_flips, 0, sizeof(n_flips));
    int dwell_hist[H][100]; memset(dwell_hist, 0, sizeof(dwell_hist)); /* dwell time histogram */
    float mean_abs_h[H]; memset(mean_abs_h, 0, sizeof(mean_abs_h));
    float min_abs_h[H]; for (int j = 0; j < H; j++) min_abs_h[j] = 2.0f;
    int sat_count[H]; memset(sat_count, 0, sizeof(sat_count)); /* times |h| >= 0.999 */

    for (int j = 0; j < H; j++) last_flip[j] = -1;

    printf("=== Q4: Saturation Structure ===\n\n");
    printf("  t  n_sat  n_flip  min|h|   active_neurons\n");

    for (int t = 0; t < n-1; t++) {
        float hn[H];
        for (int i = 0; i < H; i++) {
            float z = m.bh[i] + m.Wx[i][data[t]];
            for (int j = 0; j < H; j++) z += m.Wh[i][j]*h[j];
            hn[i] = tanhf(z);
        }
        memcpy(h, hn, sizeof(h));

        int n_sat = 0, n_flip = 0;
        float min_abs = 2.0f;
        int active_list[H]; int n_active = 0;

        for (int j = 0; j < H; j++) {
            float ah = fabsf(h[j]);
            mean_abs_h[j] += ah;
            if (ah < min_abs_h[j]) min_abs_h[j] = ah;
            if (ah >= 0.999f) { sat_count[j]++; n_sat++; }
            if (ah < min_abs) min_abs = ah;

            int new_sign = (h[j] >= 0) ? 1 : 0;
            if (t > 0 && new_sign != sign[j]) {
                /* Record dwell time */
                if (last_flip[j] >= 0) {
                    int dwell = t - last_flip[j];
                    if (dwell < 100) dwell_hist[j][dwell]++;
                }
                last_flip[j] = t;
                n_flips[j]++;
                n_flip++;
                active_list[n_active++] = j;
            }
            sign[j] = new_sign;
        }

        if (t < 10 || (t < 50 && t % 5 == 0) || t % 50 == 0 || t == n-2) {
            printf("  %-4d %3d    %3d    %.4f  ", t, n_sat, n_flip, min_abs);
            for (int i = 0; i < n_active && i < 8; i++) printf("h%d ", active_list[i]);
            if (n_active > 8) printf("...");
            printf("\n");
        }
    }

    /* ===== Neuron Classification ===== */
    printf("\n=== Neuron Classification ===\n");
    printf("  j  n_flips  mean|h|  min|h|  pct_saturated  dwell_mode  class\n");

    /* Sort by n_flips */
    int sorted[H]; for (int j = 0; j < H; j++) sorted[j] = j;
    for (int a = 0; a < H-1; a++)
        for (int b = a+1; b < H; b++)
            if (n_flips[sorted[b]] > n_flips[sorted[a]])
                { int t = sorted[a]; sorted[a] = sorted[b]; sorted[b] = t; }

    int n_always_sat = 0, n_mostly_sat = 0, n_active_neurons = 0, n_volatile = 0;

    for (int idx = 0; idx < H; idx++) {
        int j = sorted[idx];
        float mh = mean_abs_h[j] / (n-1);
        float pct_sat = 100.0f * sat_count[j] / (n-1);

        /* Find mode of dwell time */
        int mode_dwell = 0, mode_count = 0;
        for (int d = 1; d < 100; d++)
            if (dwell_hist[j][d] > mode_count) { mode_count = dwell_hist[j][d]; mode_dwell = d; }

        const char* cls;
        if (n_flips[j] == 0) { cls = "frozen"; n_always_sat++; }
        else if (n_flips[j] <= 5) { cls = "settled"; n_mostly_sat++; }
        else if (n_flips[j] <= 50) { cls = "active"; n_active_neurons++; }
        else { cls = "volatile"; n_volatile++; }

        if (idx < 30 || n_flips[j] == 0) {
            printf("  %-3d  %4d    %.4f  %.4f   %6.1f%%       %3d         %s\n",
                   j, n_flips[j], mh, min_abs_h[j], pct_sat, mode_dwell, cls);
        }
    }

    printf("\nClassification summary:\n");
    printf("  Frozen (0 flips):    %3d neurons\n", n_always_sat);
    printf("  Settled (1-5 flips): %3d neurons\n", n_mostly_sat);
    printf("  Active (6-50 flips): %3d neurons\n", n_active_neurons);
    printf("  Volatile (>50 flips):%3d neurons\n", n_volatile);

    /* ===== Dwell Time Distribution ===== */
    printf("\n=== Global Dwell Time Distribution ===\n");
    int global_dwell[100]; memset(global_dwell, 0, sizeof(global_dwell));
    for (int j = 0; j < H; j++)
        for (int d = 1; d < 100; d++)
            global_dwell[d] += dwell_hist[j][d];

    printf("  dwell  count\n");
    for (int d = 1; d < 50; d++)
        if (global_dwell[d] > 0)
            printf("  %3d    %5d\n", d, global_dwell[d]);

    /* ===== Flip Pattern: which neurons flip together? ===== */
    printf("\n=== Co-Flip Analysis ===\n");
    /* Re-run and track which neurons flip at each position */
    memset(h, 0, sizeof(h));
    int prev_sign[H]; memset(prev_sign, 0, sizeof(prev_sign));
    int coflip[H][H]; memset(coflip, 0, sizeof(coflip));
    int flip_count2[H]; memset(flip_count2, 0, sizeof(flip_count2));

    for (int t = 0; t < n-1; t++) {
        float hn[H];
        for (int i = 0; i < H; i++) {
            float z = m.bh[i] + m.Wx[i][data[t]];
            for (int j = 0; j < H; j++) z += m.Wh[i][j]*h[j];
            hn[i] = tanhf(z);
        }
        memcpy(h, hn, sizeof(h));

        int flipped[H]; int nf = 0;
        for (int j = 0; j < H; j++) {
            int s = (h[j] >= 0) ? 1 : 0;
            if (t > 0 && s != prev_sign[j]) {
                flipped[nf++] = j;
                flip_count2[j]++;
            }
            prev_sign[j] = s;
        }

        /* Record co-flips */
        for (int a = 0; a < nf; a++)
            for (int b = a+1; b < nf; b++)
                coflip[flipped[a]][flipped[b]]++;
    }

    /* Top co-flip pairs */
    printf("Top 15 co-flip pairs (neurons that flip at the same position):\n");
    typedef struct { int a; int b; int count; float jaccard; } CoFlip;
    CoFlip cf[H*H]; int ncf = 0;
    for (int a = 0; a < H; a++)
        for (int b = a+1; b < H; b++)
            if (coflip[a][b] > 0) {
                cf[ncf].a = a; cf[ncf].b = b; cf[ncf].count = coflip[a][b];
                int union_count = flip_count2[a] + flip_count2[b] - coflip[a][b];
                cf[ncf].jaccard = union_count > 0 ? (float)coflip[a][b] / union_count : 0;
                ncf++;
            }
    for (int x = 0; x < 15 && x < ncf; x++)
        for (int y = x+1; y < ncf; y++)
            if (cf[y].count > cf[x].count) { CoFlip t = cf[x]; cf[x] = cf[y]; cf[y] = t; }

    printf("  pair       co-flips  jaccard  individual_flips\n");
    for (int x = 0; x < 15 && x < ncf; x++)
        printf("  h%-3d,h%-3d  %4d      %.3f    %d, %d\n",
               cf[x].a, cf[x].b, cf[x].count, cf[x].jaccard,
               flip_count2[cf[x].a], flip_count2[cf[x].b]);

    return 0;
}
