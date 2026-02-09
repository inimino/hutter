/*
 * binary_states.c — Analyze the sat-rnn's binary hidden states.
 *
 * Runs the forward pass on the dataset and records the hidden state at
 * every position. Since the model is deeply saturated (tanh → ±1),
 * quantizes to sign(h) and reports:
 *   - Number of distinct binary states
 *   - Hamming distance between consecutive states (neurons flipping per step)
 *   - Per-neuron: flip rate, mean activation, saturation depth
 *   - Which neurons are sticky (rarely flip) vs volatile (often flip)
 *
 * Usage: binary_states <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 4096

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

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data_file> <model_file>\n", argv[0]);
        return 1;
    }

    /* Load data */
    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[MAX_DATA];
    int N = fread(data, 1, MAX_DATA, df);
    fclose(df);
    printf("Data: %d bytes\n\n", N);

    /* Load model */
    RNN rnn;
    load_model(&rnn, argv[2]);
    memset(rnn.h, 0, sizeof(rnn.h));

    /* Run forward pass, record all hidden states */
    float h_states[MAX_DATA][HIDDEN_SIZE];
    signed char h_signs[MAX_DATA][HIDDEN_SIZE];

    for (int t = 0; t < N; t++) {
        rnn_step(&rnn, data[t]);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_states[t][j] = rnn.h[j];
            h_signs[t][j] = (rnn.h[j] >= 0) ? 1 : -1;
        }
    }

    /* Count distinct binary states */
    int distinct = 0;
    int state_id[MAX_DATA];  /* which distinct state each position maps to */
    int state_count[MAX_DATA]; /* how many positions share each state */
    memset(state_count, 0, sizeof(state_count));

    for (int t = 0; t < N; t++) {
        int found = -1;
        for (int s = 0; s < distinct; s++) {
            /* find first position with this state */
            int first = -1;
            for (int u = 0; u < t; u++) {
                if (state_id[u] == s) { first = u; break; }
            }
            if (first < 0) continue;

            int match = 1;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                if (h_signs[t][j] != h_signs[first][j]) { match = 0; break; }
            }
            if (match) { found = s; break; }
        }
        if (found >= 0) {
            state_id[t] = found;
            state_count[found]++;
        } else {
            state_id[t] = distinct;
            state_count[distinct] = 1;
            distinct++;
        }
    }

    printf("=== Binary Hidden States ===\n");
    printf("Positions: %d\n", N);
    printf("Distinct binary states: %d (%.1f%% of positions)\n",
           distinct, 100.0 * distinct / N);

    /* State size distribution */
    int size1 = 0, size2 = 0, size3plus = 0;
    int max_count = 0;
    for (int s = 0; s < distinct; s++) {
        if (state_count[s] == 1) size1++;
        else if (state_count[s] == 2) size2++;
        else size3plus++;
        if (state_count[s] > max_count) max_count = state_count[s];
    }
    printf("  Unique states (count=1): %d\n", size1);
    printf("  Shared by 2: %d\n", size2);
    printf("  Shared by 3+: %d (largest group: %d positions)\n\n", size3plus, max_count);

    /* Hamming distance between consecutive states */
    int hamming[MAX_DATA];
    int total_flips = 0;
    int min_ham = HIDDEN_SIZE, max_ham = 0;
    for (int t = 1; t < N; t++) {
        int d = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            if (h_signs[t][j] != h_signs[t-1][j]) d++;
        hamming[t] = d;
        total_flips += d;
        if (d < min_ham) min_ham = d;
        if (d > max_ham) max_ham = d;
    }
    printf("=== Hamming Distances (consecutive states) ===\n");
    printf("Mean flips per step: %.1f / %d neurons\n",
           (double)total_flips / (N-1), HIDDEN_SIZE);
    printf("Min: %d  Max: %d\n", min_ham, max_ham);

    /* Hamming histogram */
    int ham_hist[HIDDEN_SIZE+1];
    memset(ham_hist, 0, sizeof(ham_hist));
    for (int t = 1; t < N; t++) ham_hist[hamming[t]]++;
    printf("Histogram: ");
    for (int d = 0; d <= HIDDEN_SIZE; d++)
        if (ham_hist[d] > 0) printf("%d:%d ", d, ham_hist[d]);
    printf("\n\n");

    /* Per-neuron analysis */
    printf("=== Per-Neuron Analysis ===\n");
    printf("%-6s %-8s %-10s %-12s %-10s\n",
           "Neuron", "Flips", "FlipRate%", "MeanAct", "Saturation");

    int neuron_flips[HIDDEN_SIZE];
    float neuron_mean[HIDDEN_SIZE];
    float neuron_sat[HIDDEN_SIZE];  /* mean |h| - distance from saturation */

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        int flips = 0;
        float sum_h = 0, sum_abs = 0;
        for (int t = 0; t < N; t++) {
            sum_h += h_states[t][j];
            sum_abs += fabsf(h_states[t][j]);
            if (t > 0 && h_signs[t][j] != h_signs[t-1][j]) flips++;
        }
        neuron_flips[j] = flips;
        neuron_mean[j] = sum_h / N;
        neuron_sat[j] = sum_abs / N;
    }

    /* Sort by flip rate */
    int idx[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) idx[j] = j;
    for (int i = 0; i < HIDDEN_SIZE-1; i++)
        for (int j = i+1; j < HIDDEN_SIZE; j++)
            if (neuron_flips[idx[i]] < neuron_flips[idx[j]]) {
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

    /* Top 20 most volatile */
    printf("\nTop 20 volatile (most flips):\n");
    for (int i = 0; i < 20 && i < HIDDEN_SIZE; i++) {
        int j = idx[i];
        printf("  h%-4d %4d     %5.1f%%     %+.4f     %.4f\n",
               j, neuron_flips[j],
               100.0 * neuron_flips[j] / (N-1),
               neuron_mean[j], neuron_sat[j]);
    }

    /* Bottom 20 most sticky */
    printf("\nTop 20 sticky (fewest flips):\n");
    for (int i = HIDDEN_SIZE-1; i >= HIDDEN_SIZE-20 && i >= 0; i--) {
        int j = idx[i];
        printf("  h%-4d %4d     %5.1f%%     %+.4f     %.4f\n",
               j, neuron_flips[j],
               100.0 * neuron_flips[j] / (N-1),
               neuron_mean[j], neuron_sat[j]);
    }

    /* Summary statistics */
    int sticky = 0, volatile_ = 0, moderate = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float rate = (float)neuron_flips[j] / (N-1);
        if (rate < 0.05) sticky++;
        else if (rate > 0.30) volatile_++;
        else moderate++;
    }
    printf("\nClassification:\n");
    printf("  Sticky (<5%% flips): %d neurons\n", sticky);
    printf("  Moderate (5-30%%): %d neurons\n", moderate);
    printf("  Volatile (>30%%): %d neurons\n", volatile_);

    /* Overall saturation depth */
    float total_sat = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) total_sat += neuron_sat[j];
    printf("\nMean saturation (|h|): %.4f (1.0 = fully saturated)\n",
           total_sat / HIDDEN_SIZE);

    /* Count positions where h[j] is near 0 (not saturated) */
    int near_zero = 0;
    for (int t = 0; t < N; t++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            if (fabsf(h_states[t][j]) < 0.5) near_zero++;
    printf("Activations with |h| < 0.5: %d / %d (%.2f%%)\n",
           near_zero, N * HIDDEN_SIZE,
           100.0 * near_zero / (N * HIDDEN_SIZE));

    return 0;
}
