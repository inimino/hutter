/*
 * compare_wh.c — Compare W_h between two trained RNN models.
 *
 * Tests the prediction: surviving skip-2-gram patterns should
 * correspond to W_h connections that strengthen from DSS=1024 to
 * DSS=2048, while artifact patterns should weaken.
 *
 * Usage: compare_wh <model_1024> <model_2048> <data_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256

typedef struct {
    float Wx[HIDDEN_SIZE][INPUT_SIZE];
    float Wh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bh[HIDDEN_SIZE];
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
    float h[HIDDEN_SIZE];
} RNN;

void load_model(RNN* rnn, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("load_model"); exit(1); }
    fread(rnn->Wx, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
    fread(rnn->Wh, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    fread(rnn->bh, sizeof(float), HIDDEN_SIZE, f);
    fread(rnn->Wy, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(rnn->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

void rnn_step(RNN* rnn, unsigned char x_t) {
    float h_new[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = rnn->bh[i] + rnn->Wx[i][x_t];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += rnn->Wh[i][j] * rnn->h[j];
        h_new[i] = tanhf(sum);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++)
        rnn->h[i] = h_new[i];
}

void softmax(float* logits, float* probs, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - maxv);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

static char safe(int c) { return (c >= 32 && c < 127) ? c : '.'; }

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model_1024> <model_2048> <data_file>\n", argv[0]);
        return 1;
    }

    RNN rnn1, rnn2;
    load_model(&rnn1, argv[1]);
    load_model(&rnn2, argv[2]);

    FILE* df = fopen(argv[3], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[2048];
    int len = fread(data, 1, 2048, df);
    fclose(df);

    printf("Model 1: %s\nModel 2: %s\nData: %d bytes\n\n", argv[1], argv[2], len);

    /* === 1. W_h comparison statistics === */
    printf("=== W_h Weight Comparison ===\n\n");

    double max1 = 0, max2 = 0;
    double sum_abs1 = 0, sum_abs2 = 0;
    double sum_diff = 0, sum_abs_diff = 0;
    int n = HIDDEN_SIZE * HIDDEN_SIZE;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float w1 = rnn1.Wh[i][j];
            float w2 = rnn2.Wh[i][j];
            if (fabsf(w1) > max1) max1 = fabsf(w1);
            if (fabsf(w2) > max2) max2 = fabsf(w2);
            sum_abs1 += fabsf(w1);
            sum_abs2 += fabsf(w2);
            sum_diff += (w2 - w1);
            sum_abs_diff += fabsf(w2 - w1);
        }
    }

    printf("                  Model 1 (1024)   Model 2 (2048)\n");
    printf("Max |W_h|:        %10.4f        %10.4f\n", max1, max2);
    printf("Mean |W_h|:       %10.4f        %10.4f\n", sum_abs1/n, sum_abs2/n);
    printf("Mean diff (2-1):  %10.4f\n", sum_diff/n);
    printf("Mean |diff|:      %10.4f\n\n", sum_abs_diff/n);

    /* Spectral norm comparison (power iteration) */
    for (int model = 0; model < 2; model++) {
        RNN* rnn = (model == 0) ? &rnn1 : &rnn2;
        float v[HIDDEN_SIZE], u[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) v[i] = 1.0f / sqrtf(HIDDEN_SIZE);
        float sigma = 0;
        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                u[i] = 0;
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    u[i] += rnn->Wh[i][j] * v[j];
            }
            float norm = 0;
            for (int i = 0; i < HIDDEN_SIZE; i++) norm += u[i] * u[i];
            norm = sqrtf(norm);
            for (int i = 0; i < HIDDEN_SIZE; i++) u[i] /= norm;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                v[j] = 0;
                for (int i = 0; i < HIDDEN_SIZE; i++)
                    v[j] += rnn->Wh[i][j] * u[i];
            }
            float norm2 = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) norm2 += v[j] * v[j];
            sigma = sqrtf(norm2);
            for (int j = 0; j < HIDDEN_SIZE; j++) v[j] /= sigma;
        }
        printf("Spectral norm (model %d): %.4f\n", model + 1, sigma);
    }

    /* === 2. Top W_h changes === */
    printf("\n=== Largest W_h Changes (2048 vs 1024) ===\n\n");

    typedef struct { int from; int to; float w1; float w2; float delta; } WChange;
    WChange changes[HIDDEN_SIZE * HIDDEN_SIZE];
    int nc = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            changes[nc].from = j;
            changes[nc].to = i;
            changes[nc].w1 = rnn1.Wh[i][j];
            changes[nc].w2 = rnn2.Wh[i][j];
            changes[nc].delta = rnn2.Wh[i][j] - rnn1.Wh[i][j];
            nc++;
        }

    /* Sort by absolute delta */
    for (int i = 0; i < nc - 1; i++)
        for (int j = i + 1; j < nc; j++)
            if (fabsf(changes[j].delta) > fabsf(changes[i].delta)) {
                WChange tmp = changes[i]; changes[i] = changes[j]; changes[j] = tmp;
            }

    printf("h_from -> h_to   W(1024)   W(2048)   delta\n");
    for (int i = 0; i < 25; i++)
        printf("h%-4d -> h%-4d  %+7.4f   %+7.4f   %+7.4f\n",
               changes[i].from, changes[i].to,
               changes[i].w1, changes[i].w2, changes[i].delta);

    /* === 3. Check our predicted highways === */
    printf("\n=== Predicted Highway Check ===\n\n");
    printf("Connection       W(1024)   W(2048)   delta    prediction\n");

    struct { int from; int to; const char* name; } highways[] = {
        {8, 52, "h8->h52 (main highway)"},
        {8, 8,  "h8->h8 (self-inhibit)"},
        {8, 90, "h8->h90 (fanout)"},
        {20, 97, "h20->h97 (secondary)"},
        {68, 99, "h68->h99 (output relay)"},
        {50, 76, "h50->h76 (strongest)"},
    };
    for (int k = 0; k < 6; k++) {
        int i = highways[k].to;
        int j = highways[k].from;
        float w1 = rnn1.Wh[i][j];
        float w2 = rnn2.Wh[i][j];
        printf("%-20s %+7.4f   %+7.4f   %+7.4f\n",
               highways[k].name, w1, w2, w2 - w1);
    }

    /* === 4. Per-neuron information flow comparison === */
    printf("\n=== Neuron Information Flow Comparison ===\n\n");

    /* Run both models forward on first 1024 bytes */
    float h1_states[2048][HIDDEN_SIZE];
    float h2_states[2048][HIDDEN_SIZE];

    memset(rnn1.h, 0, sizeof(rnn1.h));
    memset(rnn2.h, 0, sizeof(rnn2.h));
    for (int t = 0; t < len; t++) {
        rnn_step(&rnn1, data[t]);
        rnn_step(&rnn2, data[t]);
        memcpy(h1_states[t], rnn1.h, sizeof(rnn1.h));
        memcpy(h2_states[t], rnn2.h, sizeof(rnn2.h));
    }

    /* Compute neuron scores for both models */
    printf("neuron  score(1024)  score(2048)  delta    role_change\n");

    typedef struct { int j; double s1; double s2; double d; } NComp;
    NComp ncomp[HIDDEN_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double delta1 = 0, delta2 = 0;
        double contrib1 = 0, contrib2 = 0;
        int nt = 0;
        for (int t = 2; t < len - 1; t++) {
            if (t < 1) continue;
            delta1 += fabs(h1_states[t][j] - h1_states[t-1][j]);
            delta2 += fabs(h2_states[t][j] - h2_states[t-1][j]);
            int y = data[t + 1];
            contrib1 += rnn1.Wy[y][j] * h1_states[t][j];
            contrib2 += rnn2.Wy[y][j] * h2_states[t][j];
            nt++;
        }
        if (nt > 0) {
            delta1 /= nt; delta2 /= nt;
            contrib1 /= nt; contrib2 /= nt;
        }
        ncomp[j].j = j;
        ncomp[j].s1 = delta1 * fabs(contrib1);
        ncomp[j].s2 = delta2 * fabs(contrib2);
        ncomp[j].d = ncomp[j].s2 - ncomp[j].s1;
    }

    /* Sort by score in model 1 */
    for (int i = 0; i < HIDDEN_SIZE - 1; i++)
        for (int j = i + 1; j < HIDDEN_SIZE; j++)
            if (ncomp[j].s1 > ncomp[i].s1) {
                NComp tmp = ncomp[i]; ncomp[i] = ncomp[j]; ncomp[j] = tmp;
            }

    for (int i = 0; i < 15; i++)
        printf("h%-4d   %8.4f     %8.4f     %+7.4f\n",
               ncomp[i].j, ncomp[i].s1, ncomp[i].s2, ncomp[i].d);

    /* === 5. BPC comparison on both halves === */
    printf("\n=== BPC Evaluation ===\n\n");

    for (int model = 0; model < 2; model++) {
        RNN* rnn = (model == 0) ? &rnn1 : &rnn2;
        const char* name = (model == 0) ? "Model 1 (1024)" : "Model 2 (2048)";

        for (int half = 0; half < 2; half++) {
            int start = half * 1024;
            int end = start + 1024;
            if (end > len) break;

            memset(rnn->h, 0, sizeof(rnn->h));
            /* Warm up hidden state to start position */
            for (int t = 0; t < start; t++)
                rnn_step(rnn, data[t]);

            double total_loss = 0;
            int nc = 0;
            for (int t = start; t < end - 1; t++) {
                rnn_step(rnn, data[t]);
                float logits[OUTPUT_SIZE];
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    logits[i] = rnn->by[i];
                    for (int j = 0; j < HIDDEN_SIZE; j++)
                        logits[i] += rnn->Wy[i][j] * rnn->h[j];
                }
                float probs[OUTPUT_SIZE];
                softmax(logits, probs, OUTPUT_SIZE);
                float p = probs[data[t + 1]];
                if (p < 1e-8f) p = 1e-8f;
                total_loss -= log2f(p);
                nc++;
            }

            double bpc = total_loss / nc;
            printf("%-16s  bytes [%d,%d): %.4f bpc\n",
                   name, start, end, bpc);
        }
    }

    /* Correlation between W_h matrices */
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double x = rnn1.Wh[i][j], y = rnn2.Wh[i][j];
            sx += x; sy += y;
            sxx += x*x; syy += y*y; sxy += x*y;
        }
    double mx = sx/n, my = sy/n;
    double r = (sxy/n - mx*my) /
               (sqrt(sxx/n - mx*mx) * sqrt(syy/n - my*my) + 1e-10);
    printf("\nW_h correlation between models: r=%.4f\n", r);

    return 0;
}
