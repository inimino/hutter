/*
 * write_weights3.c — Hebbian weight construction from data.
 *
 * Key insight: the trained weights encode the covariance structure of
 * the data's patterns. By the Hebbian learning rule:
 *   W_h[i][j] ~ cov(h_j(t), h_i(t+1))
 *   W_x[j][x] ~ cov(one_hot(x_t), h_j(t))
 *   W_y[o][j] ~ cov(h_j(t), one_hot(y_{t+1}))
 *
 * This is the RIGHT approach because:
 * 1. Gradient descent on cross-entropy is equivalent to Hebbian learning
 *    for the first update step (and approximately for subsequent steps
 *    when the learning rate is small).
 * 2. The UM isomorphism says: the trained weights encode data statistics.
 *    Covariance IS the data statistic.
 *
 * We construct all four weight matrices from the trained model's hidden
 * state trajectory, measure correlation, and evaluate the constructed model.
 *
 * Usage: write_weights3 <data_file> <model_file>
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

double pearson(double* x, double* y, int n) {
    double sx=0, sy=0, sxx=0, syy=0, sxy=0;
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
    Model trained; load_model(&trained, argv[2]);

    printf("=== Write Weights v3: Hebbian Construction ===\n");
    printf("Data: %d bytes\n\n", n);

    double trained_bpc = eval_bpc(data, n, &trained);
    printf("Trained model bpc: %.4f\n\n", trained_bpc);

    /* Run trained model to get hidden state trajectory */
    static float h_traj[1025][H];
    float h[H]; memset(h, 0, sizeof(h));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &trained);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
    }
    int T = n - 1;

    /* Compute mean hidden state */
    double h_mean[H]; memset(h_mean, 0, sizeof(h_mean));
    for (int t = 0; t < T; t++)
        for (int j = 0; j < H; j++)
            h_mean[j] += h_traj[t][j];
    for (int j = 0; j < H; j++) h_mean[j] /= T;

    /* ========== Hebbian W_h: cov(h_j(t), h_i(t+1)) ========== */
    printf("=== W_h: Hebbian cov(h_j(t), h_i(t+1)) ===\n");
    static double cov_hh[H][H];
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            double s = 0;
            for (int t = 0; t < T-1; t++)
                s += (h_traj[t][j] - h_mean[j]) * (h_traj[t+1][i] - h_mean[i]);
            cov_hh[i][j] = s / (T-1);
        }
    }

    /* Correlation with trained W_h */
    double flat_cov[H*H], flat_trained_wh[H*H];
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++) {
            flat_cov[i*H+j] = cov_hh[i][j];
            flat_trained_wh[i*H+j] = trained.Wh[i][j];
        }
    double r_wh = pearson(flat_cov, flat_trained_wh, H*H);
    printf("  r(cov, trained W_h) all: %.4f\n", r_wh);

    /* Important entries only */
    int n_big = 0;
    double big_cov[H*H], big_trained[H*H];
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++)
            if (fabsf(trained.Wh[i][j]) >= 3.0) {
                big_cov[n_big] = cov_hh[i][j];
                big_trained[n_big] = trained.Wh[i][j];
                n_big++;
            }
    double r_wh_big = pearson(big_cov, big_trained, n_big);
    printf("  r(cov, trained W_h) |W_h|>=3.0 (%d): %.4f\n", n_big, r_wh_big);

    /* Sign prediction accuracy */
    int wh_correct = 0, wh_total = 0;
    int wh_big_correct = 0, wh_big_total = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++) {
            if (fabsf(trained.Wh[i][j]) < 0.5) continue;
            wh_total++;
            int pred = (cov_hh[i][j] > 0) ? 1 : -1;
            int actual = (trained.Wh[i][j] > 0) ? 1 : -1;
            if (pred == actual) wh_correct++;
            if (fabsf(trained.Wh[i][j]) >= 3.0) {
                wh_big_total++;
                if (pred == actual) wh_big_correct++;
            }
        }
    printf("  Sign accuracy (|w|>=0.5): %d/%d = %.1f%%\n",
           wh_correct, wh_total, 100.0*wh_correct/wh_total);
    printf("  Sign accuracy (|w|>=3.0): %d/%d = %.1f%%\n",
           wh_big_correct, wh_big_total, 100.0*wh_big_correct/wh_big_total);

    /* ========== Hebbian W_x: cov(one_hot(x_t), h_j(t)) ========== */
    printf("\n=== W_x: Hebbian cov(input_x, h_j) ===\n");

    /* For each (j, x): average of h_j(t) when data[t] = x, minus mean h_j */
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));
    for (int t = 0; t < T; t++) byte_count[data[t]]++;

    static double cov_xh[H][INPUT_SIZE];
    for (int j = 0; j < H; j++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            if (byte_count[x] == 0) { cov_xh[j][x] = 0; continue; }
            double sum = 0;
            for (int t = 0; t < T; t++)
                if (data[t] == x) sum += h_traj[t][j];
            cov_xh[j][x] = sum / byte_count[x] - h_mean[j];
        }
    }

    double flat_cov_wx[H*INPUT_SIZE], flat_trained_wx[H*INPUT_SIZE];
    for (int j = 0; j < H; j++)
        for (int x = 0; x < INPUT_SIZE; x++) {
            flat_cov_wx[j*INPUT_SIZE+x] = cov_xh[j][x];
            flat_trained_wx[j*INPUT_SIZE+x] = trained.Wx[j][x];
        }
    double r_wx = pearson(flat_cov_wx, flat_trained_wx, H*INPUT_SIZE);
    printf("  r(cov, trained W_x) all: %.4f\n", r_wx);

    /* Per-neuron W_x */
    double best_r = 0; int best_j = -1;
    double mean_abs_r = 0;
    for (int j = 0; j < H; j++) {
        double rj = pearson(cov_xh[j], (double[INPUT_SIZE]){0}, INPUT_SIZE); /* need to compute properly */
        /* Actually compute per-neuron correlation */
        double pred[INPUT_SIZE], actual[INPUT_SIZE];
        for (int x = 0; x < INPUT_SIZE; x++) {
            pred[x] = cov_xh[j][x];
            actual[x] = trained.Wx[j][x];
        }
        rj = pearson(pred, actual, INPUT_SIZE);
        mean_abs_r += fabs(rj);
        if (fabs(rj) > fabs(best_r)) { best_r = rj; best_j = j; }
    }
    printf("  Mean |r| per neuron: %.4f, best: h%d r=%.4f\n", mean_abs_r/H, best_j, best_r);

    /* ========== Hebbian W_y: cov(h_j(t), output y_{t+1}) ========== */
    printf("\n=== W_y: Hebbian cov(h_j, output_y) ===\n");

    /* For each (o, j): average of h_j(t) when data[t+1] = o, minus mean h_j */
    int next_count[256]; memset(next_count, 0, sizeof(next_count));
    for (int t = 0; t < T-1; t++) next_count[data[t+1]]++;

    static double cov_hy[OUTPUT_SIZE][H];
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        for (int j = 0; j < H; j++) {
            if (next_count[o] == 0) { cov_hy[o][j] = 0; continue; }
            double sum = 0;
            for (int t = 0; t < T-1; t++)
                if (data[t+1] == o) sum += h_traj[t][j];
            cov_hy[o][j] = sum / next_count[o] - h_mean[j];
        }
    }

    double flat_cov_wy[OUTPUT_SIZE*H], flat_trained_wy[OUTPUT_SIZE*H];
    for (int o = 0; o < OUTPUT_SIZE; o++)
        for (int j = 0; j < H; j++) {
            flat_cov_wy[o*H+j] = cov_hy[o][j];
            flat_trained_wy[o*H+j] = trained.Wy[o][j];
        }
    double r_wy = pearson(flat_cov_wy, flat_trained_wy, OUTPUT_SIZE*H);
    printf("  r(cov, trained W_y) all: %.4f\n", r_wy);

    /* Per-neuron W_y */
    double best_r_wy = 0; int best_j_wy = -1;
    double mean_abs_r_wy = 0;
    for (int j = 0; j < H; j++) {
        double pred[OUTPUT_SIZE], actual[OUTPUT_SIZE];
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            pred[o] = cov_hy[o][j];
            actual[o] = trained.Wy[o][j];
        }
        double rj = pearson(pred, actual, OUTPUT_SIZE);
        mean_abs_r_wy += fabs(rj);
        if (fabs(rj) > fabs(best_r_wy)) { best_r_wy = rj; best_j_wy = j; }
    }
    printf("  Mean |r| per neuron: %.4f, best: h%d r=%.4f\n",
           mean_abs_r_wy/H, best_j_wy, best_r_wy);

    /* ========== Construct model from Hebbian covariances ========== */
    printf("\n=== Constructing Model from Hebbian Covariances ===\n");

    /* Find scaling factors by regression: trained = scale * cov + offset */
    /* For W_h: find scale that minimizes || trained - scale * cov ||^2 */
    double num = 0, den = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++) {
            num += trained.Wh[i][j] * cov_hh[i][j];
            den += cov_hh[i][j] * cov_hh[i][j];
        }
    double wh_scale = (den > 1e-10) ? num / den : 1.0;
    printf("  W_h scale: %.2f\n", wh_scale);

    num = den = 0;
    for (int j = 0; j < H; j++)
        for (int x = 0; x < INPUT_SIZE; x++) {
            num += trained.Wx[j][x] * cov_xh[j][x];
            den += cov_xh[j][x] * cov_xh[j][x];
        }
    double wx_scale = (den > 1e-10) ? num / den : 1.0;
    printf("  W_x scale: %.2f\n", wx_scale);

    num = den = 0;
    for (int o = 0; o < OUTPUT_SIZE; o++)
        for (int j = 0; j < H; j++) {
            num += trained.Wy[o][j] * cov_hy[o][j];
            den += cov_hy[o][j] * cov_hy[o][j];
        }
    double wy_scale = (den > 1e-10) ? num / den : 1.0;
    printf("  W_y scale: %.2f\n", wy_scale);

    /* Build constructed model */
    Model constructed;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++)
            constructed.Wh[i][j] = (float)(cov_hh[i][j] * wh_scale);

    for (int j = 0; j < H; j++)
        for (int x = 0; x < INPUT_SIZE; x++)
            constructed.Wx[j][x] = (float)(cov_xh[j][x] * wx_scale);

    for (int o = 0; o < OUTPUT_SIZE; o++)
        for (int j = 0; j < H; j++)
            constructed.Wy[o][j] = (float)(cov_hy[o][j] * wy_scale);

    /* b_h from mean pre-activation bias */
    for (int j = 0; j < H; j++) {
        double frac = 0;
        for (int t = 0; t < T; t++)
            if (h_traj[t][j] >= 0) frac += 1;
        frac /= T;
        constructed.bh[j] = (float)(log(frac / (1.0 - frac + 1e-6)) * 5.0);
    }

    /* b_y from marginal */
    for (int o = 0; o < OUTPUT_SIZE; o++)
        constructed.by[o] = logf((byte_count[o] + 0.5f) / (n + 128.0f));

    double constructed_bpc = eval_bpc(data, n, &constructed);
    printf("\n  Hebbian-constructed model bpc: %.4f\n", constructed_bpc);

    /* ========== Hybrid replacements ========== */
    printf("\n=== Hybrid Models ===\n");

    Model hybrid;
    memcpy(&hybrid, &trained, sizeof(Model));
    memcpy(hybrid.Wh, constructed.Wh, sizeof(hybrid.Wh));
    printf("  Trained + Hebbian W_h:  %.4f\n", eval_bpc(data, n, &hybrid));

    memcpy(&hybrid, &trained, sizeof(Model));
    memcpy(hybrid.Wx, constructed.Wx, sizeof(hybrid.Wx));
    printf("  Trained + Hebbian W_x:  %.4f\n", eval_bpc(data, n, &hybrid));

    memcpy(&hybrid, &trained, sizeof(Model));
    memcpy(hybrid.Wy, constructed.Wy, sizeof(hybrid.Wy));
    printf("  Trained + Hebbian W_y:  %.4f\n", eval_bpc(data, n, &hybrid));

    memcpy(&hybrid, &trained, sizeof(Model));
    memcpy(hybrid.bh, constructed.bh, sizeof(hybrid.bh));
    printf("  Trained + Hebbian b_h:  %.4f\n", eval_bpc(data, n, &hybrid));

    /* ========== Interpolation: alpha*trained + (1-alpha)*Hebbian ========== */
    printf("\n=== Interpolation: alpha*trained + (1-alpha)*Hebbian ===\n");
    printf("  alpha   W_h_bpc   W_x_bpc   W_y_bpc\n");
    double alphas[] = {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0};
    for (int ai = 0; ai < 9; ai++) {
        double a = alphas[ai];

        /* W_h interpolation */
        memcpy(&hybrid, &trained, sizeof(Model));
        for (int i = 0; i < H; i++)
            for (int j = 0; j < H; j++)
                hybrid.Wh[i][j] = (float)(a * trained.Wh[i][j] + (1-a) * constructed.Wh[i][j]);
        double bpc_wh = eval_bpc(data, n, &hybrid);

        /* W_x interpolation */
        memcpy(&hybrid, &trained, sizeof(Model));
        for (int j = 0; j < H; j++)
            for (int x = 0; x < INPUT_SIZE; x++)
                hybrid.Wx[j][x] = (float)(a * trained.Wx[j][x] + (1-a) * constructed.Wx[j][x]);
        double bpc_wx = eval_bpc(data, n, &hybrid);

        /* W_y interpolation */
        memcpy(&hybrid, &trained, sizeof(Model));
        for (int o = 0; o < OUTPUT_SIZE; o++)
            for (int j = 0; j < H; j++)
                hybrid.Wy[o][j] = (float)(a * trained.Wy[o][j] + (1-a) * constructed.Wy[o][j]);
        double bpc_wy = eval_bpc(data, n, &hybrid);

        printf("  %.1f     %.4f    %.4f    %.4f\n", a, bpc_wh, bpc_wx, bpc_wy);
    }

    /* ========== Optimized W_y on Hebbian dynamics ========== */
    printf("\n=== Hebbian Wx,Wh,bh + Optimized W_y ===\n");
    Model opt; memcpy(&opt, &constructed, sizeof(Model));

    /* Recompute h trajectory with constructed model */
    static float h_con[1025][H];
    memset(h, 0, sizeof(h));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &opt);
        memcpy(h, hn, sizeof(h));
        memcpy(h_con[t], hn, sizeof(hn));
    }

    /* Gradient descent on W_y */
    float lr = 0.5;
    for (int epoch = 0; epoch < 500; epoch++) {
        static float dWy[OUTPUT_SIZE][H];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));
        double total_loss = 0;

        for (int t = 0; t < T-1; t++) {
            float* ht = h_con[t];
            int y = data[t+1];
            double logits[OUTPUT_SIZE], max_l = -1e30;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                double s = opt.by[o];
                for (int j = 0; j < H; j++) s += opt.Wy[o][j]*ht[j];
                logits[o] = s; if (s > max_l) max_l = s;
            }
            double P[OUTPUT_SIZE], se = 0;
            for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(logits[o]-max_l); se += P[o]; }
            for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;
            total_loss += -log2(P[y] > 1e-30 ? P[y] : 1e-30);

            for (int o = 0; o < OUTPUT_SIZE; o++) {
                float err = (float)(P[o] - (o == y ? 1.0 : 0.0));
                dby[o] += err;
                for (int j = 0; j < H; j++)
                    dWy[o][j] += err * ht[j];
            }
        }

        float scale = lr / (T-1);
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            opt.by[o] -= scale * dby[o];
            for (int j = 0; j < H; j++)
                opt.Wy[o][j] -= scale * dWy[o][j];
        }

        if (epoch == 200) lr *= 0.3;
        if (epoch == 350) lr *= 0.3;

        if (epoch < 3 || epoch % 100 == 0 || epoch == 499)
            printf("  epoch %3d: %.4f bpc\n", epoch, total_loss / (T-1));
    }

    double opt_bpc = eval_bpc(data, n, &opt);
    printf("  Hebbian(Wx,Wh,bh) + Optimized(Wy): %.4f bpc\n", opt_bpc);

    /* ========== Summary ========== */
    printf("\n========================================\n");
    printf("=== Hebbian Construction Summary ===\n");
    printf("========================================\n\n");
    printf("Correlation with trained weights:\n");
    printf("  W_h all:      r = %.4f (R² = %.1f%%)\n", r_wh, 100*r_wh*r_wh);
    printf("  W_h |w|>=3.0: r = %.4f (R² = %.1f%%)\n", r_wh_big, 100*r_wh_big*r_wh_big);
    printf("  W_x all:      r = %.4f (R² = %.1f%%)\n", r_wx, 100*r_wx*r_wx);
    printf("  W_y all:      r = %.4f (R² = %.1f%%)\n", r_wy, 100*r_wy*r_wy);
    printf("\nSign prediction accuracy:\n");
    printf("  W_h (|w|>=0.5): %.1f%%\n", 100.0*wh_correct/wh_total);
    printf("  W_h (|w|>=3.0): %.1f%%\n", 100.0*wh_big_correct/wh_big_total);
    printf("\nModel evaluation:\n");
    printf("  Trained:                      %.4f bpc\n", trained_bpc);
    printf("  Hebbian (all from cov):       %.4f bpc\n", constructed_bpc);
    printf("  Hebbian + optimized W_y:      %.4f bpc\n", opt_bpc);
    printf("  Trained + Hebbian W_h:        %.4f bpc (+%.3f)\n",
           eval_bpc(data, n, ({memcpy(&hybrid, &trained, sizeof(Model));
           memcpy(hybrid.Wh, constructed.Wh, sizeof(hybrid.Wh)); &hybrid;})),
           eval_bpc(data, n, ({memcpy(&hybrid, &trained, sizeof(Model));
           memcpy(hybrid.Wh, constructed.Wh, sizeof(hybrid.Wh)); &hybrid;})) - trained_bpc);

    return 0;
}
