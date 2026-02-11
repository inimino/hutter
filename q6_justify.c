/*
 * q6_justify.c — Q6: Human-readable justification per prediction.
 *
 * For each position t (a sample of them), produce:
 * 1. The context (last 30 characters), true next byte, predicted distribution
 * 2. Top 5 neurons contributing to the prediction (via W_y)
 * 3. For each contributing neuron, its backward chain through W_h (depth 3)
 * 4. The input bytes that most affected each contributing neuron
 *
 * This is "total interpretation" made concrete: for every prediction,
 * here's WHY the model predicted what it predicted.
 *
 * Usage: q6_justify <data_file> <model_file>
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

char printable(int c) { return (c >= 32 && c < 127) ? c : '.'; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    /* Run full RNN and store trajectories */
    static float h_traj[1024][H];
    static int sign_traj[1024][H]; /* sign vectors */
    float h_cur[H]; memset(h_cur, 0, sizeof(h_cur));

    for (int t = 0; t < n-1; t++) {
        float hn[H];
        rnn_step(hn, h_cur, data[t], &m);
        memcpy(h_cur, hn, sizeof(h_cur));
        memcpy(h_traj[t], hn, sizeof(hn));
        for (int j = 0; j < H; j++)
            sign_traj[t][j] = (hn[j] >= 0) ? 1 : 0;
    }

    /* Sample positions to justify */
    int positions[] = {10, 20, 42, 50, 80, 100, 150, 200, 250, 300, 400, 500};
    int n_pos = 0;
    for (int i = 0; i < 12; i++) if (positions[i] < n-2) n_pos++;

    for (int pi = 0; pi < n_pos; pi++) {
        int t = positions[pi];
        int y = data[t+1]; /* true next byte */

        printf("========================================\n");
        printf("Position t=%d\n", t);

        /* Context */
        printf("Context: \"");
        int ctx_start = (t > 30) ? t - 30 : 0;
        for (int i = ctx_start; i <= t; i++) printf("%c", printable(data[i]));
        printf("\"\n");
        printf("True next: '%c' (0x%02x)\n", printable(y), y);

        /* Compute output distribution */
        float* h_t = h_traj[t];
        double P[OUTPUT_SIZE], logit[OUTPUT_SIZE], max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m.by[o];
            for (int j = 0; j < H; j++) s += m.Wy[o][j]*h_t[j];
            logit[o] = s; P[o] = s; if (s > max_l) max_l = s;
        }
        double se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(P[o]-max_l); se += P[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;

        double bpc = -log2(P[y] > 1e-30 ? P[y] : 1e-30);
        printf("P(true) = %.4f, bpc = %.3f\n", P[y], bpc);

        /* Top 5 predicted bytes */
        int top_bytes[5]; double top_probs[5];
        for (int k = 0; k < 5; k++) { top_bytes[k] = -1; top_probs[k] = -1; }
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            for (int k = 0; k < 5; k++) {
                if (P[o] > top_probs[k]) {
                    for (int l = 4; l > k; l--) { top_bytes[l] = top_bytes[l-1]; top_probs[l] = top_probs[l-1]; }
                    top_bytes[k] = o; top_probs[k] = P[o];
                    break;
                }
            }
        }
        printf("Top predictions: ");
        for (int k = 0; k < 5; k++)
            printf("'%c'(%.3f) ", printable(top_bytes[k]), top_probs[k]);
        printf("\n");

        /* Per-neuron contribution to the true byte's logit */
        /* logit(y) = by[y] + sum_j Wy[y][j]*h[j] */
        /* Contribution of neuron j = Wy[y][j]*h[j] */
        typedef struct { int j; double contrib; double delta_bpc; } NContrib;
        NContrib nc[H];

        for (int j = 0; j < H; j++) {
            nc[j].j = j;
            nc[j].contrib = m.Wy[y][j] * h_t[j];

            /* Also compute delta_bpc from flipping sign */
            float h_flip[H]; memcpy(h_flip, h_t, sizeof(float)*H);
            h_flip[j] = -h_flip[j];
            double P2[OUTPUT_SIZE]; max_l = -1e30;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                double s = m.by[o];
                for (int k = 0; k < H; k++) s += m.Wy[o][k]*h_flip[k];
                P2[o] = s; if (s > max_l) max_l = s;
            }
            se = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++) { P2[o] = exp(P2[o]-max_l); se += P2[o]; }
            for (int o = 0; o < OUTPUT_SIZE; o++) P2[o] /= se;
            nc[j].delta_bpc = -log2(P2[y] > 1e-30 ? P2[y] : 1e-30) - bpc;
        }

        /* Sort by |delta_bpc| */
        for (int a = 0; a < H-1; a++)
            for (int b = a+1; b < H; b++)
                if (fabs(nc[b].delta_bpc) > fabs(nc[a].delta_bpc))
                    { NContrib tmp = nc[a]; nc[a] = nc[b]; nc[b] = tmp; }

        printf("\nTop 5 contributing neurons (by |delta_bpc| when sign flipped):\n");
        for (int a = 0; a < 5; a++) {
            int j = nc[a].j;
            int s = sign_traj[t][j];
            printf("  h%-3d: sign=%+d, Wy*h=%+.3f, delta_bpc=%+.3f\n",
                   j, s ? 1 : -1, nc[a].contrib, nc[a].delta_bpc);

            /* Backward chain: what determined this neuron's sign? */
            /* z_j = bh + Wx[j][data[t]] + sum_k Wh[j][k]*h[t-1][k] */
            if (t > 0) {
                float* h_prev = h_traj[t-1];
                float z = m.bh[j] + m.Wx[j][data[t]];
                float contribs_wh[H];
                for (int k = 0; k < H; k++) {
                    contribs_wh[k] = m.Wh[j][k] * h_prev[k];
                    z += contribs_wh[k];
                }

                /* Top 3 Wh contributors */
                int top3[3] = {0,0,0}; float top3v[3] = {0,0,0};
                for (int k = 0; k < H; k++) {
                    if (fabsf(contribs_wh[k]) > fabsf(top3v[2])) {
                        /* Insert into top 3 */
                        for (int l = 0; l < 3; l++) {
                            if (fabsf(contribs_wh[k]) > fabsf(top3v[l])) {
                                for (int r = 2; r > l; r--) { top3[r] = top3[r-1]; top3v[r] = top3v[r-1]; }
                                top3[l] = k; top3v[l] = contribs_wh[k];
                                break;
                            }
                        }
                    }
                }

                printf("    z=%.1f (bias=%.1f, Wx['%c']=%.1f), top Wh: ",
                       z, m.bh[j], printable(data[t]), m.Wx[j][data[t]]);
                for (int l = 0; l < 3; l++)
                    printf("h%d(%+.1f) ", top3[l], top3v[l]);
                printf("\n");

                /* For the top source, what caused IT? */
                if (t > 1) {
                    int k0 = top3[0];
                    float* h_prev2 = h_traj[t-2];
                    float z2 = m.bh[k0] + m.Wx[k0][data[t-1]];
                    float contribs2[H];
                    for (int k = 0; k < H; k++) {
                        contribs2[k] = m.Wh[k0][k] * h_prev2[k];
                        z2 += contribs2[k];
                    }
                    int best_k = 0; float best_v = 0;
                    for (int k = 0; k < H; k++)
                        if (fabsf(contribs2[k]) > fabsf(best_v)) { best_k = k; best_v = contribs2[k]; }
                    printf("      h%d at t-%d: z=%.1f, input='%c', top source=h%d(%+.1f)\n",
                           k0, 1, z2, printable(data[t-1]), best_k, best_v);
                }
            }
        }

        /* Human-readable summary */
        printf("\nJustification: ");
        printf("The model predicts '%c' with P=%.3f (bpc=%.1f). ",
               printable(top_bytes[0]), top_probs[0], -log2(top_probs[0]));
        if (top_bytes[0] == y) printf("(CORRECT) ");
        else printf("(true='%c' at P=%.3f) ", printable(y), P[y]);

        printf("The prediction is driven by h%d (%+.3f bpc) ",
               nc[0].j, nc[0].delta_bpc);
        if (nc[0].delta_bpc > 0.1)
            printf("which would HURT the prediction if flipped");
        else if (nc[0].delta_bpc < -0.1)
            printf("which would HELP the prediction if flipped");
        else printf("(marginal effect)");

        if (t > 0) {
            printf(", caused by input '%c' at t=%d", printable(data[t]), t);
            printf(" and the state from context");
        }
        printf(".\n\n");
    }

    return 0;
}
