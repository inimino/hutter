/*
 * q7_algebraic.c — Q7: Algebraic Collapse.
 *
 * Do compound patterns from the Boolean dynamics match data-countable
 * patterns in the superset UM?
 *
 * Method:
 * 1. Build a skip-bigram table from data (all offset pairs within window 30).
 * 2. For each prediction, compute the "Boolean attribution vector":
 *    which input bytes at which past offsets contributed, through which neurons.
 * 3. Check: are the top Boolean attribution entries (input, offset) pairs
 *    that also have high skip-bigram PMI?
 *
 * This tests whether the RNN has learned the same patterns that the
 * data directly contains, or something different.
 *
 * Usage: q7_algebraic <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define H HIDDEN_SIZE
#define MAX_OFFSET 20

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

char printable(int c) { return (c >= 32 && c < 127) ? c : '.'; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    /* ===== 1. Build skip-bigram PMI table ===== */
    printf("=== Skip-Bigram PMI from Data ===\n");

    /* Count: joint[d][x][y] = how many times data[t]=x and data[t+d]=y */
    /* We'll use d=1..MAX_OFFSET */
    static int joint[MAX_OFFSET+1][256][256];
    memset(joint, 0, sizeof(joint));
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));

    for (int t = 0; t < n; t++) {
        byte_count[data[t]]++;
        for (int d = 1; d <= MAX_OFFSET && t+d < n; d++)
            joint[d][data[t]][data[t+d]]++;
    }

    /* Top PMI pairs per offset */
    printf("Top skip-bigram PMI per offset (from data):\n");
    for (int d = 1; d <= 10; d++) {
        int n_pairs = n - d;
        typedef struct { int x; int y; double pmi; int count; } PMIPair;
        PMIPair best = {0, 0, -99, 0};

        for (int x = 0; x < 256; x++) {
            if (byte_count[x] == 0) continue;
            for (int y = 0; y < 256; y++) {
                if (byte_count[y] == 0 || joint[d][x][y] == 0) continue;
                double px = (double)byte_count[x] / n;
                double py = (double)byte_count[y] / n;
                double pxy = (double)joint[d][x][y] / n_pairs;
                double pmi = log2(pxy / (px * py));
                if (pmi > best.pmi) {
                    best.x = x; best.y = y; best.pmi = pmi;
                    best.count = joint[d][x][y];
                }
            }
        }
        printf("  d=%2d: '%c','%c' PMI=%.2f (count=%d)\n",
               d, printable(best.x), printable(best.y), best.pmi, best.count);
    }

    /* ===== 2. RNN backward attribution ===== */
    printf("\n=== RNN Backward Attribution vs Data PMI ===\n");

    /* Run RNN forward */
    static float h_traj[1025][H];
    float h[H]; memset(h, 0, sizeof(h));
    memset(h_traj, 0, sizeof(h_traj));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &m);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
    }

    /* For each test position t, compute attribution per (offset, input_byte) */
    int test_positions[] = {42, 80, 100, 150, 200, 300, 400, 500};
    int n_test = 0;
    for (int i = 0; i < 8; i++) if (test_positions[i] < n-2) n_test++;

    /* Global accumulator: for each offset d, what fraction of RNN attribution
     * aligns with high-PMI pairs in the data? */
    double rnn_attr_total[MAX_OFFSET+1]; memset(rnn_attr_total, 0, sizeof(rnn_attr_total));
    double rnn_attr_pmi_aligned[MAX_OFFSET+1]; memset(rnn_attr_pmi_aligned, 0, sizeof(rnn_attr_pmi_aligned));

    for (int ti = 0; ti < n_test; ti++) {
        int t = test_positions[ti];
        int y = data[t+1];
        float* h_t = h_traj[t];

        /* Compute baseline bpc */
        double P[OUTPUT_SIZE], max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m.by[o];
            for (int j = 0; j < H; j++) s += m.Wy[o][j]*h_t[j];
            P[o] = s; if (s > max_l) max_l = s;
        }
        double se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(P[o]-max_l); se += P[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;
        double bpc_base = -log2(P[y] > 1e-30 ? P[y] : 1e-30);

        /* For each past offset d, flip the input byte at t-d+1, re-run forward */
        printf("\nPosition t=%d, context=\"", t);
        int cs = (t > 15) ? t - 15 : 0;
        for (int i = cs; i <= t; i++) printf("%c", printable(data[i]));
        printf("\", true='%c', bpc=%.2f\n", printable(y), bpc_base);
        printf("  offset  input  delta_bpc  data_PMI(input,true)\n");

        for (int d = 1; d <= MAX_OFFSET && t-d+1 >= 0; d++) {
            int t_flip = t - d + 1; /* position to flip */
            int orig = data[t_flip];
            int flipped = (orig + 128) & 0xFF;

            /* Re-run from t_flip */
            float h_alt[H];
            if (t_flip > 0)
                memcpy(h_alt, h_traj[t_flip-1], sizeof(h_alt));
            else
                memset(h_alt, 0, sizeof(h_alt));

            /* Step at t_flip with flipped byte */
            float hn[H]; rnn_step(hn, h_alt, flipped, &m);
            memcpy(h_alt, hn, sizeof(h_alt));

            /* Continue with original bytes */
            for (int s = t_flip+1; s <= t; s++) {
                rnn_step(hn, h_alt, data[s], &m);
                memcpy(h_alt, hn, sizeof(h_alt));
            }

            /* Compute new bpc */
            max_l = -1e30;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                double s = m.by[o];
                for (int j = 0; j < H; j++) s += m.Wy[o][j]*h_alt[j];
                P[o] = s; if (s > max_l) max_l = s;
            }
            se = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(P[o]-max_l); se += P[o]; }
            for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;
            double bpc_alt = -log2(P[y] > 1e-30 ? P[y] : 1e-30);
            double delta = bpc_alt - bpc_base;

            /* Data PMI for (orig, y) at offset d */
            int n_pairs = n - d;
            double pmi = 0;
            if (byte_count[orig] > 0 && byte_count[y] > 0 && joint[d][orig][y] > 0) {
                double px = (double)byte_count[orig] / n;
                double py = (double)byte_count[y] / n;
                double pxy = (double)joint[d][orig][y] / n_pairs;
                pmi = log2(pxy / (px * py));
            }

            rnn_attr_total[d] += fabs(delta);
            if (pmi > 0.5) rnn_attr_pmi_aligned[d] += fabs(delta);

            if (d <= 10 || fabs(delta) > 0.3)
                printf("  d=%-3d  '%c'    %+.3f      %+.2f\n",
                       d, printable(orig), delta, pmi);
        }
    }

    /* ===== 3. Alignment summary ===== */
    printf("\n=== Alignment: RNN Attribution vs Data PMI ===\n");
    printf("  offset  total_attr  pmi_aligned_attr  pct_aligned\n");
    double grand_total = 0, grand_aligned = 0;
    for (int d = 1; d <= MAX_OFFSET; d++) {
        double pct = rnn_attr_total[d] > 0 ? 100.0 * rnn_attr_pmi_aligned[d] / rnn_attr_total[d] : 0;
        printf("  d=%-3d   %7.3f     %7.3f            %5.1f%%\n",
               d, rnn_attr_total[d], rnn_attr_pmi_aligned[d], pct);
        grand_total += rnn_attr_total[d];
        grand_aligned += rnn_attr_pmi_aligned[d];
    }
    printf("  Total:  %7.3f     %7.3f            %5.1f%%\n",
           grand_total, grand_aligned, 100.0*grand_aligned/grand_total);

    return 0;
}
