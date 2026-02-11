/*
 * q1_boolean.c — Analyze the Boolean transition function of the sat-rnn.
 *
 * The sign-only dynamics defines a Boolean function:
 *   sigma_{t+1} = f(sigma_t, x_t)
 * where sigma is a 128-bit sign vector and x is an 8-bit input.
 *
 * We analyze:
 * 1. Influence: which input neurons affect which output neurons
 * 2. Sensitivity: how many neurons flip per input byte
 * 3. Boolean structure: is f a threshold function? majority? XOR?
 * 4. The W_h sign pattern: positive vs negative connections
 * 5. Effective dimension: how many independent Boolean functions?
 *
 * Usage: q1_boolean <data_file> <model_file>
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
} Model;

void load_model(Model* m, const char* path) {
    FILE* f = fopen(path, "rb");
    fread(m->Wx, sizeof(float), HIDDEN_SIZE*INPUT_SIZE, f);
    fread(m->Wh, sizeof(float), HIDDEN_SIZE*HIDDEN_SIZE, f);
    fread(m->bh, sizeof(float), HIDDEN_SIZE, f);
    fread(m->Wy, sizeof(float), OUTPUT_SIZE*HIDDEN_SIZE, f);
    fread(m->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

/* Boolean step: sigma -> sigma' given input x */
void bool_step(int* s_out, int* s_in, int x, Model* m) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float z = m->bh[i] + m->Wx[i][x];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            z += m->Wh[i][j] * (s_in[j] ? 1.0f : -1.0f);
        s_out[i] = (z >= 0) ? 1 : 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <data> <model>\n", argv[0]); return 1; }

    FILE* f = fopen(argv[1], "rb");
    unsigned char data[1100]; int n = fread(data, 1, 1100, f); fclose(f);
    if (n > 1024) n = 1024;
    Model m; load_model(&m, argv[2]);

    /* ===== 1. W_h sign structure ===== */
    printf("=== W_h Sign Structure ===\n");
    int wh_pos = 0, wh_neg = 0, wh_zero = 0;
    double wh_pos_mag = 0, wh_neg_mag = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (m.Wh[i][j] > 0.001f) { wh_pos++; wh_pos_mag += m.Wh[i][j]; }
            else if (m.Wh[i][j] < -0.001f) { wh_neg++; wh_neg_mag -= m.Wh[i][j]; }
            else wh_zero++;
        }
    printf("W_h entries: %d positive (mean %.3f), %d negative (mean %.3f), %d near-zero\n",
           wh_pos, wh_pos_mag/wh_pos, wh_neg, wh_neg_mag/wh_neg, wh_zero);

    /* ===== 2. Per-neuron: effective fan-in ===== */
    printf("\n=== Per-Neuron Analysis (Boolean Transition) ===\n");
    printf("  j  fan_in(>0.1)  bias_sign  threshold  dominant_inputs\n");

    int total_fan_in = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        /* Count significant W_h connections */
        int fan_in = 0;
        float max_wh = 0;
        int max_j = -1;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (fabsf(m.Wh[i][j]) > 0.1f) fan_in++;
            if (fabsf(m.Wh[i][j]) > max_wh) { max_wh = fabsf(m.Wh[i][j]); max_j = j; }
        }
        total_fan_in += fan_in;

        /* Compute threshold: z = bh + Wx + sum(Wh * sigma) */
        /* The neuron flips when z crosses 0 */
        /* Bias + max possible Wx contribution */
        float max_wx = 0;
        for (int x = 0; x < 256; x++)
            if (fabsf(m.Wx[i][x]) > max_wx) max_wx = fabsf(m.Wx[i][x]);

        /* Total possible Wh contribution */
        float total_wh = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) total_wh += fabsf(m.Wh[i][j]);

        /* The threshold is where bh + Wx sits relative to the Wh range */
        float threshold = (total_wh > 0) ? (m.bh[i] + max_wx) / total_wh : 0;

        if (i < 20 || fan_in < 5 || fan_in > 50) {
            printf("  %-3d  %3d          %+.2f     %+.3f     max: j=%d (%.3f)\n",
                   i, fan_in, m.bh[i], threshold, max_j, max_wh);
        }
    }
    printf("Mean fan-in (|Wh|>0.1): %.1f\n", (float)total_fan_in / HIDDEN_SIZE);

    /* ===== 3. Boolean sensitivity analysis ===== */
    printf("\n=== Boolean Sensitivity: Flip One Input Neuron ===\n");

    /* Run Boolean dynamics on actual data */
    int sigma[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) sigma[j] = 0;

    /* Get to t=42 */
    for (int t = 0; t <= 42; t++) {
        int s_new[HIDDEN_SIZE];
        bool_step(s_new, sigma, data[t], &m);
        memcpy(sigma, s_new, sizeof(sigma));
    }

    printf("State at t=42: ");
    for (int j = 0; j < 16; j++) printf("%d", sigma[j]);
    printf("... (showing first 16)\n\n");

    /* Flip each input neuron and count output changes */
    printf("  j_flip  output_changes  most_affected\n");
    int total_sensitivity = 0;
    for (int j_flip = 0; j_flip < HIDDEN_SIZE; j_flip++) {
        int s_base[HIDDEN_SIZE], s_flip[HIDDEN_SIZE];
        int sigma_flip[HIDDEN_SIZE];
        memcpy(sigma_flip, sigma, sizeof(sigma));
        sigma_flip[j_flip] ^= 1; /* flip one bit */

        bool_step(s_base, sigma, data[43], &m);
        bool_step(s_flip, sigma_flip, data[43], &m);

        int changes = 0;
        int most_affected = -1;
        for (int i = 0; i < HIDDEN_SIZE; i++)
            if (s_base[i] != s_flip[i]) { changes++; most_affected = i; }
        total_sensitivity += changes;

        if (j_flip < 10 || changes > 10)
            printf("  %-3d     %3d             %d\n", j_flip, changes, most_affected);
    }
    printf("Mean sensitivity: %.2f output changes per input flip\n",
           (float)total_sensitivity / HIDDEN_SIZE);

    /* ===== 4. Input byte sensitivity ===== */
    printf("\n=== Input Byte Sensitivity ===\n");
    /* For t=42, how does changing the input byte change the next state? */
    int s_base[HIDDEN_SIZE];
    bool_step(s_base, sigma, data[43], &m);

    printf("  byte  changes  example_char\n");
    int byte_changes[256];
    for (int x = 0; x < 256; x++) {
        int s_alt[HIDDEN_SIZE];
        bool_step(s_alt, sigma, x, &m);
        int changes = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++)
            if (s_base[i] != s_alt[i]) changes++;
        byte_changes[x] = changes;
    }
    /* Find min, max, mean */
    int min_c = 128, max_c = 0; double sum_c = 0;
    for (int x = 0; x < 256; x++) {
        if (byte_changes[x] < min_c) min_c = byte_changes[x];
        if (byte_changes[x] > max_c) max_c = byte_changes[x];
        sum_c += byte_changes[x];
    }
    printf("Min changes: %d, Max: %d, Mean: %.1f\n", min_c, max_c, sum_c/256);

    /* ===== 5. Boolean dynamics bpc over all positions ===== */
    printf("\n=== Boolean Dynamics: Full Run ===\n");
    for (int j = 0; j < HIDDEN_SIZE; j++) sigma[j] = 0;
    double total_bpc = 0;
    int sign_flips_total = 0;

    for (int t = 0; t < n - 1; t++) {
        int s_new[HIDDEN_SIZE];
        bool_step(s_new, sigma, data[t], &m);

        int flips = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            if (s_new[j] != sigma[j]) flips++;
        sign_flips_total += flips;

        /* Predict using sign vector */
        float h_read[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            h_read[j] = s_new[j] ? 1.0f : -1.0f;

        double P[OUTPUT_SIZE], max_l = -1e30;
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            double s = m.by[o];
            for (int j = 0; j < HIDDEN_SIZE; j++) s += m.Wy[o][j]*h_read[j];
            P[o] = s; if (s > max_l) max_l = s;
        }
        double se = 0;
        for (int o = 0; o < OUTPUT_SIZE; o++) { P[o] = exp(P[o]-max_l); se += P[o]; }
        for (int o = 0; o < OUTPUT_SIZE; o++) P[o] /= se;

        int y = data[t+1];
        total_bpc += -log2(P[y] > 1e-30 ? P[y] : 1e-30);

        memcpy(sigma, s_new, sizeof(s_new));
    }

    printf("Boolean dynamics bpc: %.4f\n", total_bpc / (n-1));
    printf("Mean sign flips/step: %.1f\n", (float)sign_flips_total / (n-1));

    /* ===== 6. Count unique sign vectors ===== */
    printf("\n=== Unique Sign Vectors ===\n");
    /* Hash each 128-bit vector */
    unsigned long long hashes[1024];
    for (int j = 0; j < HIDDEN_SIZE; j++) sigma[j] = 0;
    int unique = 0;
    for (int t = 0; t < n - 1; t++) {
        int s_new[HIDDEN_SIZE];
        bool_step(s_new, sigma, data[t], &m);
        memcpy(sigma, s_new, sizeof(s_new));

        /* Simple hash */
        unsigned long long hash = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            hash = hash * 31 + s_new[j];
        hashes[t] = hash;
    }
    /* Count unique hashes */
    /* Sort and count */
    for (int i = 0; i < n-2; i++)
        for (int j = i+1; j < n-1; j++)
            if (hashes[j] < hashes[i]) { unsigned long long t = hashes[i]; hashes[i] = hashes[j]; hashes[j] = t; }
    unique = 1;
    for (int i = 1; i < n-1; i++)
        if (hashes[i] != hashes[i-1]) unique++;
    printf("Unique sign vectors: %d / %d positions\n", unique, n-1);
    printf("State entropy: %.1f bits (log2(%d))\n", log2(unique), unique);

    return 0;
}
