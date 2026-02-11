/*
 * write_weights2.c — Construct RNN weights from interpretation results.
 *
 * Uses the Boolean automaton interpretation to construct weights:
 * 1. W_h from Boolean influence graph (sign + magnitude)
 * 2. W_x from sign-conditioned input statistics
 * 3. W_y from conditional output distributions
 * 4. b_h from positive fraction
 *
 * Then evaluates the constructed model and compares with trained.
 * Also tests sign prediction accuracy.
 *
 * Usage: write_weights2 <data_file> <model_file>
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
    Model trained; load_model(&trained, argv[2]);

    printf("=== Write Weights v2: Construct Model from Interpretation ===\n");
    printf("Data: %d bytes\n\n", n);

    double trained_bpc = eval_bpc(data, n, &trained);
    printf("Trained model bpc: %.4f\n\n", trained_bpc);

    /* ========== Run trained model to get sign trajectory ========== */
    static float h_traj[1025][H];
    static int sign_traj[1025][H];
    float h[H]; memset(h, 0, sizeof(h));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &trained);
        memcpy(h, hn, sizeof(h));
        memcpy(h_traj[t], hn, sizeof(hn));
        for (int j = 0; j < H; j++)
            sign_traj[t][j] = (hn[j] >= 0) ? 1 : 0;
    }
    int T = n - 1;

    /* ========== Sign prediction accuracy ========== */
    printf("=== Weight Sign Prediction Accuracy ===\n");

    /* W_h: sign from temporal sign correlation */
    int wh_sign_correct = 0, wh_total = 0;
    int wh_big_correct = 0, wh_big_total = 0;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            if (fabsf(trained.Wh[i][j]) < 0.1) continue; /* skip near-zero */
            wh_total++;
            int agree = 0;
            for (int t = 0; t < T-1; t++)
                if (sign_traj[t][j] == sign_traj[t+1][i]) agree++;
            int pred_sign = (agree > (T-1)/2) ? 1 : -1;
            int actual_sign = (trained.Wh[i][j] > 0) ? 1 : -1;
            if (pred_sign == actual_sign) wh_sign_correct++;

            if (fabsf(trained.Wh[i][j]) >= 3.0) {
                wh_big_total++;
                if (pred_sign == actual_sign) wh_big_correct++;
            }
        }
    }
    printf("W_h sign accuracy (|w|>0.1): %d/%d = %.1f%%\n",
           wh_sign_correct, wh_total, 100.0*wh_sign_correct/wh_total);
    printf("W_h sign accuracy (|w|>=3.0): %d/%d = %.1f%%\n",
           wh_big_correct, wh_big_total, 100.0*wh_big_correct/wh_big_total);

    /* W_x: sign from sign-conditioned input byte distribution */
    int wx_sign_correct = 0, wx_total = 0;
    static int count_pos_in[H][256], count_neg_in[H][256];
    static int n_pos[H], n_neg[H];
    memset(count_pos_in, 0, sizeof(count_pos_in));
    memset(count_neg_in, 0, sizeof(count_neg_in));
    memset(n_pos, 0, sizeof(n_pos));
    memset(n_neg, 0, sizeof(n_neg));

    for (int t = 0; t < T; t++) {
        int x = data[t];
        for (int j = 0; j < H; j++) {
            if (sign_traj[t][j]) { count_pos_in[j][x]++; n_pos[j]++; }
            else { count_neg_in[j][x]++; n_neg[j]++; }
        }
    }

    for (int j = 0; j < H; j++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            if (fabsf(trained.Wx[j][x]) < 0.1) continue;
            wx_total++;
            double p_pos = (count_pos_in[j][x] + 0.5) / (n_pos[j] + 128.0);
            double p_neg = (count_neg_in[j][x] + 0.5) / (n_neg[j] + 128.0);
            int pred_sign = (p_pos > p_neg) ? 1 : -1;
            int actual_sign = (trained.Wx[j][x] > 0) ? 1 : -1;
            if (pred_sign == actual_sign) wx_sign_correct++;
        }
    }
    printf("W_x sign accuracy (|w|>0.1): %d/%d = %.1f%%\n",
           wx_sign_correct, wx_total, 100.0*wx_sign_correct/wx_total);

    /* ========== Construct model from data ========== */
    printf("\n=== Constructing Model from Data Statistics ===\n");

    Model constructed;
    memset(&constructed, 0, sizeof(constructed));

    /* b_y: from marginal distribution (same as reverse_iso2) */
    int byte_count[256]; memset(byte_count, 0, sizeof(byte_count));
    for (int t = 0; t < n; t++) byte_count[data[t]]++;
    for (int o = 0; o < OUTPUT_SIZE; o++)
        constructed.by[o] = logf((byte_count[o] + 0.5f) / (n + 128.0f));

    /* b_h: from sign fraction */
    for (int j = 0; j < H; j++) {
        double frac = (double)n_pos[j] / T;
        /* Scale to match trained magnitude range */
        constructed.bh[j] = (float)(log(frac / (1.0 - frac + 1e-6)) * 5.0);
    }

    /* W_x: from sign-conditioned input (log-ratio, scaled) */
    float wx_scale = 15.0; /* Match typical trained magnitude */
    for (int j = 0; j < H; j++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            double p_pos = (count_pos_in[j][x] + 0.5) / (n_pos[j] + 128.0);
            double p_neg = (count_neg_in[j][x] + 0.5) / (n_neg[j] + 128.0);
            double lr = log(p_pos) - log(p_neg);
            /* Clamp and scale */
            if (lr > 2.0) lr = 2.0;
            if (lr < -2.0) lr = -2.0;
            constructed.Wx[j][x] = (float)(lr * wx_scale);
        }
    }

    /* W_h: from sign correlation + influence magnitude */
    float wh_scale = 4.0;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            int agree = 0;
            for (int t = 0; t < T-1; t++)
                if (sign_traj[t][j] == sign_traj[t+1][i]) agree++;
            double corr = 2.0 * agree / (T-1) - 1.0;
            constructed.Wh[i][j] = (float)(corr * wh_scale);
        }
    }

    /* W_y: from sign-conditioned output distribution */
    static int count_pos_out[H][256], count_neg_out[H][256];
    memset(count_pos_out, 0, sizeof(count_pos_out));
    memset(count_neg_out, 0, sizeof(count_neg_out));

    for (int t = 0; t < T-1; t++) {
        int y = data[t+1];
        for (int j = 0; j < H; j++) {
            if (sign_traj[t][j]) count_pos_out[j][y]++;
            else count_neg_out[j][y]++;
        }
    }

    float wy_scale = 0.5;
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        for (int j = 0; j < H; j++) {
            double p_pos = (count_pos_out[j][o] + 0.5) / (n_pos[j] + 128.0);
            double p_neg = (count_neg_out[j][o] + 0.5) / (n_neg[j] + 128.0);
            double lr = log(p_pos) - log(p_neg);
            if (lr > 2.0) lr = 2.0;
            if (lr < -2.0) lr = -2.0;
            constructed.Wy[o][j] = (float)(lr * wy_scale);
        }
    }

    double constructed_bpc = eval_bpc(data, n, &constructed);
    printf("Constructed model bpc: %.4f (trained: %.4f)\n\n", constructed_bpc, trained_bpc);

    /* ========== Hybrid models: mix trained and constructed ========== */
    printf("=== Hybrid Models (replace one matrix with constructed version) ===\n");

    /* Keep everything trained except W_h */
    Model hybrid_wh; memcpy(&hybrid_wh, &trained, sizeof(Model));
    memcpy(hybrid_wh.Wh, constructed.Wh, sizeof(hybrid_wh.Wh));
    double bpc_hybrid_wh = eval_bpc(data, n, &hybrid_wh);
    printf("Trained + constructed W_h: %.4f\n", bpc_hybrid_wh);

    /* Keep everything trained except W_x */
    Model hybrid_wx; memcpy(&hybrid_wx, &trained, sizeof(Model));
    memcpy(hybrid_wx.Wx, constructed.Wx, sizeof(hybrid_wx.Wx));
    double bpc_hybrid_wx = eval_bpc(data, n, &hybrid_wx);
    printf("Trained + constructed W_x: %.4f\n", bpc_hybrid_wx);

    /* Keep everything trained except W_y */
    Model hybrid_wy; memcpy(&hybrid_wy, &trained, sizeof(Model));
    memcpy(hybrid_wy.Wy, constructed.Wy, sizeof(hybrid_wy.Wy));
    double bpc_hybrid_wy = eval_bpc(data, n, &hybrid_wy);
    printf("Trained + constructed W_y: %.4f\n", bpc_hybrid_wy);

    /* Keep everything trained except b_h */
    Model hybrid_bh; memcpy(&hybrid_bh, &trained, sizeof(Model));
    memcpy(hybrid_bh.bh, constructed.bh, sizeof(hybrid_bh.bh));
    double bpc_hybrid_bh = eval_bpc(data, n, &hybrid_bh);
    printf("Trained + constructed b_h: %.4f\n", bpc_hybrid_bh);

    /* ========== Optimize W_y on constructed Wx,Wh,bh ========== */
    printf("\n=== Constructed Wx,Wh,bh + Optimized W_y ===\n");

    /* Take constructed model, freeze Wx/Wh/bh, optimize W_y by gradient descent */
    Model opt_wy; memcpy(&opt_wy, &constructed, sizeof(Model));

    /* Run forward to collect h vectors */
    static float h_constructed[1025][H];
    memset(h, 0, sizeof(h));
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &opt_wy);
        memcpy(h, hn, sizeof(h));
        memcpy(h_constructed[t], hn, sizeof(hn));
    }

    /* Gradient descent on W_y and b_y */
    float lr = 0.5;
    for (int epoch = 0; epoch < 500; epoch++) {
        static float dWy[OUTPUT_SIZE][H];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));
        double total_loss = 0;

        for (int t = 0; t < T-1; t++) {
            float* ht = h_constructed[t];
            int y = data[t+1];

            float logits[OUTPUT_SIZE]; double max_l = -1e30;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                float s = opt_wy.by[o];
                for (int j = 0; j < H; j++) s += opt_wy.Wy[o][j]*ht[j];
                logits[o] = s;
                if (s > max_l) max_l = s;
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
            opt_wy.by[o] -= scale * dby[o];
            for (int j = 0; j < H; j++)
                opt_wy.Wy[o][j] -= scale * dWy[o][j];
        }

        if (epoch == 200) lr *= 0.3;
        if (epoch == 350) lr *= 0.3;

        if (epoch < 5 || epoch % 100 == 0 || epoch == 499) {
            double bpc = total_loss / (T-1);
            printf("  epoch %3d: %.4f bpc\n", epoch, bpc);
        }
    }

    double opt_bpc = eval_bpc(data, n, &opt_wy);
    printf("Constructed(Wx,Wh,bh) + Optimized(Wy): %.4f bpc\n", opt_bpc);

    /* ========== Also try: trained Wx,Wh,bh + optimized Wy from sign features ========== */
    printf("\n=== Trained Wx,Wh,bh → Sign Hidden States → Optimized W_y ===\n");

    /* Use sign(h) instead of h for readout, re-optimize W_y */
    Model sign_model; memcpy(&sign_model, &trained, sizeof(Model));
    /* Zero out W_y and re-optimize on sign features */
    memset(sign_model.Wy, 0, sizeof(sign_model.Wy));

    /* h vectors are already computed in h_traj */
    /* Replace with sign: ±1 */
    static float h_sign[1025][H];
    for (int t = 0; t < T; t++)
        for (int j = 0; j < H; j++)
            h_sign[t][j] = (h_traj[t][j] >= 0) ? 1.0f : -1.0f;

    lr = 0.5;
    for (int epoch = 0; epoch < 500; epoch++) {
        static float dWy[OUTPUT_SIZE][H];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));
        double total_loss = 0;

        for (int t = 0; t < T-1; t++) {
            float* ht = h_sign[t];
            int y = data[t+1];

            float logits[OUTPUT_SIZE]; double max_l = -1e30;
            for (int o = 0; o < OUTPUT_SIZE; o++) {
                float s = sign_model.by[o];
                for (int j = 0; j < H; j++) s += sign_model.Wy[o][j]*ht[j];
                logits[o] = s;
                if (s > max_l) max_l = s;
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
            sign_model.by[o] -= scale * dby[o];
            for (int j = 0; j < H; j++)
                sign_model.Wy[o][j] -= scale * dWy[o][j];
        }

        if (epoch == 200) lr *= 0.3;
        if (epoch == 350) lr *= 0.3;

        if (epoch < 5 || epoch % 100 == 0 || epoch == 499) {
            double bpc = total_loss / (T-1);
            printf("  epoch %3d: %.4f bpc\n", epoch, bpc);
        }
    }

    /* Evaluate sign model (with actual dynamics, sign readout) */
    /* Run full RNN but use sign for readout */
    memset(h, 0, sizeof(h));
    double sign_total = 0;
    for (int t = 0; t < n-1; t++) {
        float hn[H]; rnn_step(hn, h, data[t], &trained);
        memcpy(h, hn, sizeof(h));

        /* Use sign features for readout */
        float hs[H];
        for (int j = 0; j < H; j++) hs[j] = (hn[j] >= 0) ? 1.0f : -1.0f;

        double P[OUTPUT_SIZE], max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = sign_model.by[o];
            for (int j = 0; j < H; j++) s += sign_model.Wy[o][j]*hs[j];
            P[o] = s; if (s > max_l) max_l = s;
        }
        double se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(P[o]-max_l); se += P[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;
        int y = data[t+1];
        sign_total += -log2(P[y] > 1e-30 ? P[y] : 1e-30);
    }
    double sign_bpc = sign_total / (n-1);
    printf("Trained dynamics + Sign readout + Optimized W_y: %.4f bpc\n", sign_bpc);

    /* ========== Summary ========== */
    printf("\n========================================\n");
    printf("=== Construction Results Summary ===\n");
    printf("========================================\n");
    printf("Configuration                              bpc      Notes\n");
    printf("---------------------------------------------------------\n");
    printf("Uniform (no model)                         8.000\n");
    printf("Constructed (all from data stats)           %.4f   sign-corr W_h + log-ratio W_x\n", constructed_bpc);
    printf("Constructed Wx,Wh,bh + optimized W_y       %.4f   W_y by gradient descent\n", opt_bpc);
    printf("Trained dynamics + sign readout + opt W_y   %.4f   Boolean readout\n", sign_bpc);
    printf("Trained + constructed W_h                   %.4f   replace W_h only\n", bpc_hybrid_wh);
    printf("Trained + constructed W_x                   %.4f   replace W_x only\n", bpc_hybrid_wx);
    printf("Trained + constructed W_y                   %.4f   replace W_y only\n", bpc_hybrid_wy);
    printf("Trained + constructed b_h                   %.4f   replace b_h only\n", bpc_hybrid_bh);
    printf("Trained model                               %.4f   full f32\n", trained_bpc);
    printf("---------------------------------------------------------\n");

    printf("\nSign prediction accuracy:\n");
    printf("  W_h (|w|>0.1):  %.1f%%\n", 100.0*wh_sign_correct/wh_total);
    printf("  W_h (|w|>=3.0): %.1f%%\n", 100.0*wh_big_correct/wh_big_total);
    printf("  W_x (|w|>0.1):  %.1f%%\n", 100.0*wx_sign_correct/wx_total);

    return 0;
}
