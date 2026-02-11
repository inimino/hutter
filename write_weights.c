/*
 * write_weights.c — Predict trained weight values from data statistics.
 *
 * Using the interpretation results (neuron roles, Boolean dynamics,
 * attribution chains), predict what the trained weight values should be.
 *
 * For W_x: PMI between input bytes and the outputs each neuron promotes.
 * For W_y: Conditional log-prob split by neuron sign.
 * For W_h: Boolean influence graph.
 * For b_h: Mean pre-activation bias.
 *
 * Then measure Pearson correlation between predicted and actual weights.
 *
 * Usage: write_weights <data_file> <model_file>
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

double pearson(double* x, double* y, int n) {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += x[i]; sy += y[i];
        sxx += x[i]*x[i]; syy += y[i]*y[i];
        sxy += x[i]*y[i];
    }
    double mx = sx/n, my = sy/n;
    double vx = sxx/n - mx*mx, vy = syy/n - my*my;
    if (vx < 1e-10 || vy < 1e-10) return 0;
    return (sxy/n - mx*my) / sqrt(vx*vy);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    printf("=== Write Weights: Predict Trained Values from Data ===\n");
    printf("Data: %d bytes, Model: 128 hidden\n\n", n);

    /* ========== Step 1: Run RNN to get hidden state trajectory ========== */
    static float h_traj[1025][H];
    static int sign_traj[1025][H];
    float h[H]; memset(h, 0, sizeof(h));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &m);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
        for (int j = 0; j < H; j++)
            sign_traj[t][j] = (hn[j] >= 0) ? 1 : 0;
    }
    int T = n - 1;

    /* ========== Step 2: Compute data statistics ========== */

    /* Byte counts */
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));
    for (int t = 0; t < n; t++) byte_count[data[t]]++;

    /* Bigram counts at offset 1 */
    static int bigram[256][256];
    memset(bigram, 0, sizeof(bigram));
    for (int t = 0; t < n-1; t++)
        bigram[data[t]][data[t+1]]++;

    /* PMI(x, y) at offset 1 */
    static double pmi[256][256];
    for (int x = 0; x < 256; x++) {
        for (int y = 0; y < 256; y++) {
            if (byte_count[x] > 0 && byte_count[y] > 0 && bigram[x][y] > 0) {
                double px = (double)byte_count[x] / n;
                double py = (double)byte_count[y] / n;
                double pxy = (double)bigram[x][y] / (n-1);
                pmi[x][y] = log2(pxy / (px * py));
            } else {
                pmi[x][y] = 0;
            }
        }
    }

    /* ========== Step 3: Predict W_x ========== */
    printf("=== Predicting W_x ===\n");

    /* For each neuron j, identify what outputs it promotes/demotes (from W_y) */
    /* W_x[j][x] ~ sum over promoted outputs of PMI(x, o) - sum over demoted outputs */
    static double pred_Wx[H][INPUT_SIZE];
    static double actual_Wx[H][INPUT_SIZE];

    for (int j = 0; j < H; j++) {
        /* Find top 5 promoted and demoted outputs */
        typedef struct { int o; float w; } WyEntry;
        WyEntry wy_sorted[OUTPUT_SIZE];
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            wy_sorted[o].o = o; wy_sorted[o].w = m.Wy[o][j];
        }
        /* Partial sort: top 5 */
        for (int a = 0; a < 5; a++)
            for (int b = a+1; b < OUTPUT_SIZE; b++)
                if (wy_sorted[b].w > wy_sorted[a].w)
                    { WyEntry t = wy_sorted[a]; wy_sorted[a] = wy_sorted[b]; wy_sorted[b] = t; }
        /* Bottom 5 */
        for (int a = OUTPUT_SIZE-5; a < OUTPUT_SIZE; a++)
            for (int b = 0; b < a; b++)
                if (wy_sorted[b].w > wy_sorted[a].w)
                    { WyEntry t = wy_sorted[a]; wy_sorted[a] = wy_sorted[b]; wy_sorted[b] = t; }

        for (int x = 0; x < INPUT_SIZE; x++) {
            double p = 0;
            /* Top 5 promoted: positive PMI contribution */
            for (int a = 0; a < 5; a++)
                p += wy_sorted[a].w * pmi[x][wy_sorted[a].o];
            /* Bottom 5 demoted: negative PMI contribution */
            for (int a = OUTPUT_SIZE-5; a < OUTPUT_SIZE; a++)
                p += wy_sorted[a].w * pmi[x][wy_sorted[a].o];
            pred_Wx[j][x] = p;
            actual_Wx[j][x] = m.Wx[j][x];
        }
    }

    /* Compute correlation */
    double* flat_pred_wx = (double*)pred_Wx;
    double* flat_actual_wx = (double*)actual_Wx;
    double r_wx = pearson(flat_pred_wx, flat_actual_wx, H * INPUT_SIZE);
    printf("W_x correlation (all %d entries): r = %.4f\n", H*INPUT_SIZE, r_wx);

    /* Per-neuron correlations */
    printf("Top 10 per-neuron W_x correlations:\n");
    double neuron_r_wx[H];
    for (int j = 0; j < H; j++)
        neuron_r_wx[j] = pearson(pred_Wx[j], actual_Wx[j], INPUT_SIZE);

    int sorted_wx[H]; for (int j = 0; j < H; j++) sorted_wx[j] = j;
    for (int a = 0; a < 10; a++)
        for (int b = a+1; b < H; b++)
            if (fabs(neuron_r_wx[sorted_wx[b]]) > fabs(neuron_r_wx[sorted_wx[a]]))
                { int t = sorted_wx[a]; sorted_wx[a] = sorted_wx[b]; sorted_wx[b] = t; }
    for (int a = 0; a < 10; a++)
        printf("  h%-3d: r = %+.4f\n", sorted_wx[a], neuron_r_wx[sorted_wx[a]]);

    double mean_abs_r_wx = 0;
    for (int j = 0; j < H; j++) mean_abs_r_wx += fabs(neuron_r_wx[j]);
    printf("Mean |r| across neurons: %.4f\n\n", mean_abs_r_wx / H);

    /* ========== Step 4: Predict W_y ========== */
    printf("=== Predicting W_y ===\n");

    /* For each neuron j and output o:
     * W_y[o][j] ~ mean log P(o | h_j > 0) - mean log P(o | h_j < 0) */
    static double pred_Wy[OUTPUT_SIZE][H];
    static double actual_Wy[OUTPUT_SIZE][H];

    /* Compute: when h_j > 0, what's the distribution of next bytes?
     * When h_j < 0, what's the distribution? */
    static int count_pos_next[H][256];
    static int count_neg_next[H][256];
    static int n_pos[H], n_neg[H];
    memset(count_pos_next, 0, sizeof(count_pos_next));
    memset(count_neg_next, 0, sizeof(count_neg_next));
    memset(n_pos, 0, sizeof(n_pos));
    memset(n_neg, 0, sizeof(n_neg));

    for (int t = 0; t < T-1; t++) {
        int y = data[t+1];
        for (int j = 0; j < H; j++) {
            if (sign_traj[t][j]) {
                count_pos_next[j][y]++;
                n_pos[j]++;
            } else {
                count_neg_next[j][y]++;
                n_neg[j]++;
            }
        }
    }

    for (int j = 0; j < H; j++) {
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double p_pos = (count_pos_next[j][o] + 0.5) / (n_pos[j] + 128.0);
            double p_neg = (count_neg_next[j][o] + 0.5) / (n_neg[j] + 128.0);
            pred_Wy[o][j] = log(p_pos) - log(p_neg);
            actual_Wy[o][j] = m.Wy[o][j];
        }
    }

    double* flat_pred_wy = (double*)pred_Wy;
    double* flat_actual_wy = (double*)actual_Wy;
    double r_wy = pearson(flat_pred_wy, flat_actual_wy, OUTPUT_SIZE * H);
    printf("W_y correlation (all %d entries): r = %.4f\n", OUTPUT_SIZE*H, r_wy);

    /* Per-neuron W_y correlations */
    printf("Top 10 per-neuron W_y correlations:\n");
    double neuron_r_wy[H];
    for (int j = 0; j < H; j++) {
        double pred_col[OUTPUT_SIZE], actual_col[OUTPUT_SIZE];
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            pred_col[o] = pred_Wy[o][j];
            actual_col[o] = actual_Wy[o][j];
        }
        neuron_r_wy[j] = pearson(pred_col, actual_col, OUTPUT_SIZE);
    }
    int sorted_wy[H]; for (int j = 0; j < H; j++) sorted_wy[j] = j;
    for (int a = 0; a < 10; a++)
        for (int b = a+1; b < H; b++)
            if (fabs(neuron_r_wy[sorted_wy[b]]) > fabs(neuron_r_wy[sorted_wy[a]]))
                { int t = sorted_wy[a]; sorted_wy[a] = sorted_wy[b]; sorted_wy[b] = t; }
    for (int a = 0; a < 10; a++)
        printf("  h%-3d: r = %+.4f\n", sorted_wy[a], neuron_r_wy[sorted_wy[a]]);

    double mean_abs_r_wy = 0;
    for (int j = 0; j < H; j++) mean_abs_r_wy += fabs(neuron_r_wy[j]);
    printf("Mean |r| across neurons: %.4f\n\n", mean_abs_r_wy / H);

    /* ========== Step 5: Predict W_h from Boolean influence ========== */
    printf("=== Predicting W_h ===\n");

    /* Boolean influence: for each pair (i,j), how often does flipping
     * sign(h_j) at time t change sign(h_i) at time t+1? */
    /* We approximate by computing the fraction of positions where
     * sign(h_j[t]) correlates with sign(h_i[t+1]) controlling for
     * other inputs. Simple proxy: correlation of signs. */
    static double pred_Wh[H][H];
    static double actual_Wh[H][H];

    /* Sign correlation between h_j at time t and h_i at time t+1 */
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            int agree = 0;
            for (int t = 0; t < T-1; t++) {
                if (sign_traj[t][j] == sign_traj[t+1][i]) agree++;
            }
            /* Map to [-1, 1] */
            double corr = 2.0 * agree / (T-1) - 1.0;
            /* Scale by geometric mean of neuron importance */
            pred_Wh[i][j] = corr;
            actual_Wh[i][j] = m.Wh[i][j];
        }
    }

    double* flat_pred_wh = (double*)pred_Wh;
    double* flat_actual_wh = (double*)actual_Wh;
    double r_wh_all = pearson(flat_pred_wh, flat_actual_wh, H * H);
    printf("W_h correlation (all %d entries): r = %.4f\n", H*H, r_wh_all);

    /* Correlation for just the important entries (|W_h| >= 3.0) */
    int n_big = 0;
    double pred_big[H*H], actual_big[H*H];
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++)
            if (fabsf(m.Wh[i][j]) >= 3.0) {
                pred_big[n_big] = pred_Wh[i][j];
                actual_big[n_big] = actual_Wh[i][j];
                n_big++;
            }
    double r_wh_big = pearson(pred_big, actual_big, n_big);
    printf("W_h correlation (|W_h| >= 3.0, %d entries): r = %.4f\n", n_big, r_wh_big);

    /* ========== Step 6: Predict b_h ========== */
    printf("\n=== Predicting b_h ===\n");

    /* b_h[j] should reflect the neuron's bias toward positive or negative.
     * Proxy: fraction of time neuron is positive, mapped to log-odds. */
    double pred_bh[H], actual_bh[H];
    for (int j = 0; j < H; j++) {
        int pos = 0;
        for (int t = 0; t < T; t++) if (sign_traj[t][j]) pos++;
        double frac = (double)pos / T;
        /* Log-odds, scaled */
        pred_bh[j] = log(frac / (1.0 - frac + 1e-6)) * 10.0;
        actual_bh[j] = m.bh[j];
    }
    double r_bh = pearson(pred_bh, actual_bh, H);
    printf("b_h correlation (%d entries): r = %.4f\n", H, r_bh);

    /* ========== Step 7: Improved W_x prediction using multiple offsets ========== */
    printf("\n=== Improved W_x Prediction (Multi-Offset) ===\n");

    /* Instead of just PMI at offset 1, use the neuron's sign-conditioned
     * byte distribution: which input bytes make neuron j positive? */
    static double pred_Wx2[H][INPUT_SIZE];
    static int count_pos_in[H][256], count_neg_in[H][256];
    memset(count_pos_in, 0, sizeof(count_pos_in));
    memset(count_neg_in, 0, sizeof(count_neg_in));

    for (int t = 0; t < T; t++) {
        int x = data[t];
        for (int j = 0; j < H; j++) {
            if (sign_traj[t][j])
                count_pos_in[j][x]++;
            else
                count_neg_in[j][x]++;
        }
    }

    for (int j = 0; j < H; j++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            double p_pos = (count_pos_in[j][x] + 0.5) / (n_pos[j] + 128.0);
            double p_neg = (count_neg_in[j][x] + 0.5) / (n_neg[j] + 128.0);
            pred_Wx2[j][x] = log(p_pos) - log(p_neg);
        }
    }

    double* flat_pred_wx2 = (double*)pred_Wx2;
    double r_wx2 = pearson(flat_pred_wx2, flat_actual_wx, H * INPUT_SIZE);
    printf("W_x correlation (sign-conditioned input): r = %.4f\n", r_wx2);

    /* Per-neuron */
    double mean_abs_r_wx2 = 0;
    printf("Top 10 per-neuron W_x (sign-conditioned) correlations:\n");
    double nr2[H];
    for (int j = 0; j < H; j++)
        nr2[j] = pearson(pred_Wx2[j], actual_Wx[j], INPUT_SIZE);
    int sorted2[H]; for (int j = 0; j < H; j++) sorted2[j] = j;
    for (int a = 0; a < 10; a++)
        for (int b = a+1; b < H; b++)
            if (fabs(nr2[sorted2[b]]) > fabs(nr2[sorted2[a]]))
                { int t = sorted2[a]; sorted2[a] = sorted2[b]; sorted2[b] = t; }
    for (int a = 0; a < 10; a++)
        printf("  h%-3d: r = %+.4f\n", sorted2[a], nr2[sorted2[a]]);
    for (int j = 0; j < H; j++) mean_abs_r_wx2 += fabs(nr2[j]);
    printf("Mean |r|: %.4f\n", mean_abs_r_wx2 / H);

    /* ========== Step 8: Improved W_h prediction using actual influence ========== */
    printf("\n=== Improved W_h Prediction (Actual Boolean Influence) ===\n");

    /* For each pair (i,j), actually flip h_j and see if h_i changes */
    static double pred_Wh2[H][H];
    for (int j = 0; j < H; j++) {
        /* Sample 50 positions */
        int n_sample = 50;
        if (n_sample > T-1) n_sample = T-1;
        for (int i = 0; i < H; i++) {
            int flips = 0;
            for (int s = 0; s < n_sample; s++) {
                int t = s * (T-1) / n_sample;
                float h_copy[H];
                memcpy(h_copy, h_traj[t], sizeof(h_copy));
                /* Original sign of h_i at t+1 */
                int orig_sign = sign_traj[t+1][i];
                /* Flip h_j */
                h_copy[j] = -h_copy[j];
                /* Recompute h_i at t+1 */
                float z = m.bh[i] + m.Wx[i][data[t+1]];
                for (int k = 0; k < H; k++) z += m.Wh[i][k]*h_copy[k];
                int new_sign = (tanhf(z) >= 0) ? 1 : 0;
                if (new_sign != orig_sign) flips++;
            }
            double influence = (double)flips / n_sample;
            /* Sign from correlation */
            int agree = 0;
            for (int t = 0; t < T-1; t++)
                if (sign_traj[t][j] == sign_traj[t+1][i]) agree++;
            double sign_corr = 2.0 * agree / (T-1) - 1.0;
            pred_Wh2[i][j] = influence * (sign_corr >= 0 ? 1.0 : -1.0);
        }
    }

    double* flat_pred_wh2 = (double*)pred_Wh2;
    double r_wh2_all = pearson(flat_pred_wh2, flat_actual_wh, H * H);
    printf("W_h correlation (influence-based, all): r = %.4f\n", r_wh2_all);

    /* Important entries only */
    n_big = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++)
            if (fabsf(m.Wh[i][j]) >= 3.0) {
                pred_big[n_big] = pred_Wh2[i][j];
                actual_big[n_big] = actual_Wh[i][j];
                n_big++;
            }
    double r_wh2_big = pearson(pred_big, actual_big, n_big);
    printf("W_h correlation (influence-based, |W_h|>=3.0, %d entries): r = %.4f\n", n_big, r_wh2_big);

    /* ========== Summary ========== */
    printf("\n========================================\n");
    printf("=== Weight Prediction Summary ===\n");
    printf("========================================\n");
    printf("Matrix    Method              Entries    Correlation\n");
    printf("------    ------              -------    -----------\n");
    printf("W_x       PMI-based           %6d     r = %.4f\n", H*INPUT_SIZE, r_wx);
    printf("W_x       Sign-conditioned    %6d     r = %.4f\n", H*INPUT_SIZE, r_wx2);
    printf("W_y       Sign-split logprob  %6d     r = %.4f\n", OUTPUT_SIZE*H, r_wy);
    printf("W_h       Sign correlation    %6d     r = %.4f\n", H*H, r_wh_all);
    printf("W_h       Boolean influence   %6d     r = %.4f\n", H*H, r_wh2_all);
    printf("W_h(big)  Sign correlation    %6d     r = %.4f\n", n_big, r_wh_big);
    printf("W_h(big)  Boolean influence   %6d     r = %.4f\n", n_big, r_wh2_big);
    printf("b_h       Sign log-odds       %6d     r = %.4f\n", H, r_bh);

    printf("\nInterpretation: fraction of weight variance explained by data statistics:\n");
    printf("  W_x: %.0f%% (sign-conditioned)\n", 100*r_wx2*r_wx2);
    printf("  W_y: %.0f%% (sign-split)\n", 100*r_wy*r_wy);
    printf("  W_h: %.0f%% (all), %.0f%% (important entries)\n",
           100*r_wh2_all*r_wh2_all, 100*r_wh2_big*r_wh2_big);
    printf("  b_h: %.0f%%\n", 100*r_bh*r_bh);

    return 0;
}
