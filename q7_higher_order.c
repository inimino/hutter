/*
 * q7_higher_order.c — Does alignment improve with higher-order data statistics?
 *
 * Q7 found 61.2% alignment between RNN attribution and skip-bigram PMI.
 * This checks: does 3-gram or 4-gram PMI explain more of the RNN's behavior?
 *
 * Method:
 * For each prediction at position t:
 *   - Compute RNN sensitivity: flip data[t-d], re-run, measure bpc change
 *   - For each depth d, also compute:
 *     (a) Skip-bigram PMI(data[t-d], data[t+1]) at offset d
 *     (b) Skip-trigram PMI: condition on (data[t-d], data[t-d+k]) → data[t+1]
 *     (c) Context-conditioned sensitivity: does the RNN's sensitivity
 *         depend on data[t-d+1..t]?
 *
 * Usage: q7_higher_order <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define H HIDDEN_SIZE
#define MAX_D 20

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

double compute_bpc_at(float* h_t, int y, Model* m) {
    double P[OUTPUT_SIZE], max_l = -1e30;
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        double s = m->by[o];
        for (int j = 0; j < H; j++) s += m->Wy[o][j]*h_t[j];
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

    printf("=== Q7: Higher-Order PMI Alignment ===\n");
    printf("Data: %d bytes\n\n", n);

    /* Skip-bigram counts */
    static int bigram_joint[MAX_D+1][256][256];
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));
    memset(bigram_joint, 0, sizeof(bigram_joint));

    for (int t = 0; t < n; t++) {
        byte_count[data[t]]++;
        for (int d = 1; d <= MAX_D && t+d < n; d++)
            bigram_joint[d][data[t]][data[t+d]]++;
    }

    /* Skip-trigram counts: joint[d1][d2][x1][x2][y] — too big for full table.
     * Instead, for each test position, compute the trigram PMI on the fly
     * using conditional counts from the data. */

    /* Build forward RNN trajectory */
    static float h_traj[1025][H];
    float h[H]; memset(h, 0, sizeof(h));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &m);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
    }

    /* Test at many positions */
    int n_test = 0;
    double bigram_aligned_total[MAX_D+1], rnn_total[MAX_D+1];
    double trigram_aligned_total[MAX_D+1];
    memset(bigram_aligned_total, 0, sizeof(bigram_aligned_total));
    memset(trigram_aligned_total, 0, sizeof(trigram_aligned_total));
    memset(rnn_total, 0, sizeof(rnn_total));

    /* Test every 4th position from 30 onward */
    for (int t = 30; t < n-2; t += 4) {
        n_test++;
        int y = data[t+1];
        float* h_t = h_traj[t];
        double bpc_base = compute_bpc_at(h_t, y, &m);

        for (int d = 1; d <= MAX_D && t-d+1 >= 0; d++) {
            int t_flip = t - d + 1;
            int orig = data[t_flip];
            int flipped = (orig + 128) & 0xFF;

            /* Re-run from t_flip with flipped byte */
            float h_alt[H];
            if (t_flip > 0)
                memcpy(h_alt, h_traj[t_flip-1], sizeof(h_alt));
            else
                memset(h_alt, 0, sizeof(h_alt));

            float hn[H]; rnn_step(hn, h_alt, flipped, &m);
            memcpy(h_alt, hn, sizeof(h_alt));
            for (int s = t_flip+1; s <= t; s++) {
                rnn_step(hn, h_alt, data[s], &m);
                memcpy(h_alt, hn, sizeof(h_alt));
            }

            double bpc_alt = compute_bpc_at(h_alt, y, &m);
            double delta = fabs(bpc_alt - bpc_base);
            rnn_total[d] += delta;

            /* Skip-bigram PMI */
            int n_pairs = n - d;
            double pmi2 = 0;
            if (byte_count[orig] > 0 && byte_count[y] > 0 && bigram_joint[d][orig][y] > 0) {
                double px = (double)byte_count[orig] / n;
                double py = (double)byte_count[y] / n;
                double pxy = (double)bigram_joint[d][orig][y] / n_pairs;
                pmi2 = log2(pxy / (px * py));
            }
            if (pmi2 > 0.5) bigram_aligned_total[d] += delta;

            /* Skip-trigram: condition on the NEXT byte after the flipped position too.
             * PMI(orig, data[t_flip+1], y) at offsets (d, d-1).
             * To compute: count(orig, next, y) / (count(orig,next) * freq(y))
             * We do this by scanning data. */
            if (d >= 2 && t_flip+1 < n) {
                int next = data[t_flip+1];
                int count_triple = 0, count_pair = 0;
                int d2 = d - 1; /* offset from next to y */
                for (int s = 0; s + d < n; s++) {
                    if (data[s] == orig && s+1 < n && data[s+1] == next) {
                        count_pair++;
                        if (s + d < n && data[s+d] == y)
                            count_triple++;
                    }
                }
                double pmi3 = 0;
                if (count_pair > 2 && count_triple > 0) {
                    double p_triple = (double)count_triple / (n - d);
                    double p_pair = (double)count_pair / (n - 1);
                    double py = (double)byte_count[y] / n;
                    pmi3 = log2(p_triple / (p_pair * py));
                }
                if (pmi3 > 0.5) trigram_aligned_total[d] += delta;
            }
        }
    }

    printf("Tested %d positions\n\n", n_test);

    printf("=== Alignment by Offset ===\n");
    printf("offset  rnn_sensitivity  bigram_aligned%%  trigram_aligned%%\n");
    double grand_rnn = 0, grand_bi = 0, grand_tri = 0;
    for (int d = 1; d <= MAX_D; d++) {
        double bi_pct = rnn_total[d] > 0 ? 100.0 * bigram_aligned_total[d] / rnn_total[d] : 0;
        double tri_pct = rnn_total[d] > 0 ? 100.0 * trigram_aligned_total[d] / rnn_total[d] : 0;
        printf("  d=%-3d   %8.3f         %5.1f%%            %5.1f%%\n",
               d, rnn_total[d], bi_pct, tri_pct);
        grand_rnn += rnn_total[d];
        grand_bi += bigram_aligned_total[d];
        grand_tri += trigram_aligned_total[d];
    }
    printf("  Total   %8.3f         %5.1f%%            %5.1f%%\n",
           grand_rnn, 100.0*grand_bi/grand_rnn, 100.0*grand_tri/grand_rnn);

    printf("\nConclusion: trigram PMI explains %.1f%% vs bigram's %.1f%%\n",
           100.0*grand_tri/grand_rnn, 100.0*grand_bi/grand_rnn);
    printf("Improvement from 2-gram to 3-gram: %+.1f percentage points\n",
           100.0*(grand_tri - grand_bi)/grand_rnn);

    return 0;
}
