/*
 * rnn_trace.c — Mechanistic trace of pattern propagation through RNN.
 *
 * For each neuron j at each position t, compute the influence of input
 * x[t-d] on h_j[t] by propagating through the Jacobian chain:
 *
 *   dh[t]/dh[t-1] = diag(1 - h[t]^2) * W_h
 *
 * The product of d Jacobians gives the influence of h[t-d] on h[t].
 * Combined with W_x (input-to-hidden), this traces how each byte
 * in the input stream influences each neuron at each position.
 *
 * This verifies the factor map: if neuron j responds to offsets (d1,d2),
 * the gradient flow should show peaks at those offsets.
 *
 * Usage: rnn_trace <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 1100
#define MAX_DEPTH 30

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

    /* Load model */
    RNN rnn;
    load_model(&rnn, argv[2]);
    memset(rnn.h, 0, sizeof(rnn.h));

    /* Run forward pass, record all hidden states */
    float h_states[MAX_DATA][HIDDEN_SIZE];
    for (int t = 0; t < N; t++) {
        rnn_step(&rnn, data[t]);
        memcpy(h_states[t], rnn.h, sizeof(rnn.h));
    }
    printf("Data: %d bytes, forward pass done\n\n", N);

    /* ===================================================================
     * For each position t, compute Jacobian chain backward through time.
     *
     * J[t] = diag(1 - h[t]^2) * W_h
     *
     * Influence of h[t-d] on h[t] = J[t] * J[t-1] * ... * J[t-d+1]
     *
     * We track ||row_j|| of this product to measure how much neuron j
     * at time t depends on the hidden state at time t-d.
     * ================================================================= */

    /* Per-neuron, per-offset: average influence magnitude */
    double avg_influence[HIDDEN_SIZE][MAX_DEPTH];
    memset(avg_influence, 0, sizeof(avg_influence));

    /* Also track: for each neuron j, which input byte has strongest influence
     * at each offset. Influence of x[t-d] on h_j[t] = (J-chain) * W_x[:,x[t-d]] */
    double avg_input_influence[HIDDEN_SIZE][MAX_DEPTH];
    memset(avg_input_influence, 0, sizeof(avg_input_influence));

    int count[MAX_DEPTH];
    memset(count, 0, sizeof(count));

    /* For each position t, compute the backward Jacobian chain */
    for (int t = MAX_DEPTH; t < N; t++) {
        /* J_product starts as identity (influence of h[t] on h[t] = I) */
        /* We store row-wise: J_product[i][j] = dh_i[t] / dh_j[t-d] */
        float J_prod[HIDDEN_SIZE][HIDDEN_SIZE];
        /* Initialize to identity */
        memset(J_prod, 0, sizeof(J_prod));
        for (int i = 0; i < HIDDEN_SIZE; i++)
            J_prod[i][i] = 1.0f;

        for (int d = 1; d <= MAX_DEPTH && t - d >= 0; d++) {
            /* Compute J[t-d+1] = diag(1 - h[t-d+1]^2) * W_h */
            /* Then J_prod = J_prod * J[t-d+1]^(-1) ... actually we need:
             * influence of h[t-d] on h[t] = J[t] * J[t-1] * ... * J[t-d+1]
             *
             * We build this incrementally:
             * d=1: J_prod = J[t] = diag(1-h[t]^2) * W_h
             * d=2: J_prod = J[t] * J[t-1]
             * etc.
             *
             * So at each step, we RIGHT-multiply by J[t-d+1]:
             * Actually no. Let me think again.
             *
             * h[t] = tanh(W_h * h[t-1] + W_x * x[t] + b)
             * dh[t]/dh[t-1] = diag(1-h[t]^2) * W_h  (this is J[t])
             *
             * dh[t]/dh[t-d] = J[t] * J[t-1] * ... * J[t-d+1]
             *
             * So for d=1: result = J[t]
             *    for d=2: result = J[t] * J[t-1]
             *    etc.
             *
             * We maintain J_prod = J[t] * J[t-1] * ... * J[t-d_prev+1]
             * and then multiply on the right by J[t-d+1] = J[t - (d-1)]
             */

            int step_t = t - d + 1;  /* the time step whose Jacobian we multiply */
            float* h_at = h_states[step_t];

            /* Compute J_prod_new = J_prod * J[step_t]
             * where J[step_t][i][j] = (1 - h[step_t][i]^2) * W_h[i][j]
             * J_prod_new[r][c] = sum_k J_prod[r][k] * (1 - h[step_t][k]^2) * W_h[k][c]
             */
            float J_new[HIDDEN_SIZE][HIDDEN_SIZE];
            for (int r = 0; r < HIDDEN_SIZE; r++) {
                for (int c = 0; c < HIDDEN_SIZE; c++) {
                    float sum = 0;
                    for (int k = 0; k < HIDDEN_SIZE; k++) {
                        float deriv = 1.0f - h_at[k] * h_at[k];
                        sum += J_prod[r][k] * deriv * rnn.Wh[k][c];
                    }
                    J_new[r][c] = sum;
                }
            }
            memcpy(J_prod, J_new, sizeof(J_prod));

            /* For each neuron j (row j of J_prod), compute the L2 norm
             * across all source neurons. This is the total sensitivity of
             * neuron j at time t to the hidden state at time t-d. */
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float norm = 0;
                for (int k = 0; k < HIDDEN_SIZE; k++)
                    norm += J_prod[j][k] * J_prod[j][k];
                avg_influence[j][d - 1] += sqrt(norm);

                /* Influence of specific input x[t-d] on neuron j at time t:
                 * dh_j[t]/dx[t-d] = J_prod[j][:] * diag(1-h[t-d]^2) * W_x[:, x[t-d]]
                 * But since x is one-hot, W_x[:, x[t-d]] is just a column.
                 *
                 * Actually: dh_j[t]/dx_k[t-d] = sum_m J_prod[j][m] * (1-h[t-d][m]^2) * W_x[m][k]
                 * For the specific byte x[t-d] = data[t-d]:
                 */
                if (t - d >= 0) {
                    int byte_in = data[t - d];
                    float* h_prev = (t - d > 0) ? h_states[t - d] : NULL;
                    float input_inf = 0;
                    for (int m = 0; m < HIDDEN_SIZE; m++) {
                        float deriv_m;
                        if (h_prev)
                            deriv_m = 1.0f - h_prev[m] * h_prev[m]; /* not quite right for input */
                        else
                            deriv_m = 1.0f;
                        /* For input influence, we need: how does x[t-d] affect h[t-d]?
                         * h[t-d] = tanh(... + W_x[:][x[t-d]])
                         * dh_i[t-d]/dx[t-d] = (1-h[t-d][i]^2) * W_x[i][x[t-d]]
                         * Then propagated: dh_j[t]/dx[t-d] = sum_m J_prod[j][m] * deriv_m * W_x[m][byte_in]
                         */
                        float h_td = (t - d > 0) ? h_states[t - d][m] :
                            tanhf(rnn.bh[m] + rnn.Wx[m][data[0]]);
                        float d_m = 1.0f - h_td * h_td;
                        input_inf += J_prod[j][m] * d_m * rnn.Wx[m][byte_in];
                    }
                    avg_input_influence[j][d - 1] += fabs(input_inf);
                }
            }
            count[d - 1]++;
        }
    }

    /* Normalize */
    for (int d = 0; d < MAX_DEPTH; d++) {
        if (count[d] == 0) continue;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            avg_influence[j][d] /= count[d];
            avg_input_influence[j][d] /= count[d];
        }
    }

    /* ===================================================================
     * Report 1: Average influence magnitude per offset, aggregated
     * =================================================================== */

    printf("=== Average Hidden-to-Hidden Influence by Offset ===\n\n");
    printf("offset  mean_influence  median_neuron  max_neuron\n");
    for (int d = 0; d < MAX_DEPTH; d++) {
        double sum = 0, maxv = 0;
        double vals[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            vals[j] = avg_influence[j][d];
            sum += vals[j];
            if (vals[j] > maxv) maxv = vals[j];
        }
        /* Sort for median */
        for (int i = 0; i < HIDDEN_SIZE - 1; i++)
            for (int jj = i + 1; jj < HIDDEN_SIZE; jj++)
                if (vals[jj] < vals[i]) { double t = vals[i]; vals[i] = vals[jj]; vals[jj] = t; }

        printf("  %2d    %.6f        %.6f       %.6f\n",
               d + 1, sum / HIDDEN_SIZE, vals[63], maxv);
    }

    /* ===================================================================
     * Report 2: Per-neuron input influence profile
     * For each neuron, show the top 5 offsets by input influence
     * =================================================================== */

    printf("\n=== Per-Neuron Input Influence (top 5 offsets) ===\n\n");
    printf("neuron  [offset:influence ...]\n");

    /* Factor map best pairs from factor_map2 results */
    int best_pair[HIDDEN_SIZE][2]; /* will be filled from actual factor_map2 data */

    /* For now, report raw influence and let us compare with factor_map2 */
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        /* Find top 5 offsets for this neuron */
        int top[5] = {0, 0, 0, 0, 0};
        double top_val[5] = {0, 0, 0, 0, 0};

        for (int d = 0; d < MAX_DEPTH; d++) {
            double v = avg_input_influence[j][d];
            /* Insert into top-5 */
            for (int k = 0; k < 5; k++) {
                if (v > top_val[k]) {
                    for (int m = 4; m > k; m--) {
                        top[m] = top[m - 1];
                        top_val[m] = top_val[m - 1];
                    }
                    top[k] = d + 1;
                    top_val[k] = v;
                    break;
                }
            }
        }

        printf("h%-3d    ", j);
        for (int k = 0; k < 5; k++)
            printf("%2d:%.4f  ", top[k], top_val[k]);
        printf("\n");
    }

    /* ===================================================================
     * Report 3: Comparison with factor_map2 offset pairs
     * For each neuron, is its factor_map2 best pair in the top-2 by gradient?
     * =================================================================== */

    /* Known offset pairs from factor_map2 (hard-coded from results) */
    /* We can verify this by checking if the gradient-based top-2 matches */
    printf("\n=== Gradient vs Factor Map Agreement ===\n\n");

    /* Count how many neurons have offset 1 in their top-2 by gradient */
    int off1_in_top2 = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double v1 = avg_input_influence[j][0]; /* offset 1 */
        int rank = 0;
        for (int d = 1; d < MAX_DEPTH; d++)
            if (avg_input_influence[j][d] > v1) rank++;
        if (rank < 2) off1_in_top2++;
    }
    printf("Neurons with offset 1 in gradient top-2: %d / 128\n", off1_in_top2);

    /* For each offset from the greedy set, count how many neurons have it in top-3 */
    int greedy_offsets[] = {1, 8, 20, 3, 27, 2, 12, 7};
    printf("\nGreedy offset presence in gradient top-3 per neuron:\n");
    printf("offset  neurons_with_in_top3\n");
    for (int oi = 0; oi < 8; oi++) {
        int off = greedy_offsets[oi];
        int cnt = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double v = avg_input_influence[j][off - 1];
            int rank = 0;
            for (int d = 0; d < MAX_DEPTH; d++)
                if (d != off - 1 && avg_input_influence[j][d] > v) rank++;
            if (rank < 3) cnt++;
        }
        printf("  %2d    %d\n", off, cnt);
    }

    /* ===================================================================
     * Report 4: Trace specific examples
     * Show the full influence timeline for a few important neurons
     * =================================================================== */

    printf("\n=== Detailed Trace: Top Neurons ===\n\n");
    int trace_neurons[] = {8, 56, 68, 15, 99, 52};
    int n_trace = 6;

    for (int ni = 0; ni < n_trace; ni++) {
        int j = trace_neurons[ni];
        printf("Neuron h%d — input influence by offset:\n", j);
        printf("  offset: ");
        for (int d = 0; d < 20; d++)
            printf("%5d", d + 1);
        printf("\n  inflnc: ");
        for (int d = 0; d < 20; d++)
            printf("%.3f", avg_input_influence[j][d]);
        printf("\n  h-to-h: ");
        for (int d = 0; d < 20; d++)
            printf("%.3f", avg_influence[j][d]);
        printf("\n\n");
    }

    /* ===================================================================
     * Report 5: W_h eigenvalue analysis
     * The spectral radius determines how fast information decays.
     * =================================================================== */

    printf("=== W_h Column Norms (input mixing) ===\n\n");
    double col_norms[HIDDEN_SIZE];
    double max_col_norm = 0;
    double sum_col_norm = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double norm = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++)
            norm += rnn.Wh[i][j] * rnn.Wh[i][j];
        col_norms[j] = sqrt(norm);
        sum_col_norm += col_norms[j];
        if (col_norms[j] > max_col_norm) max_col_norm = col_norms[j];
    }
    printf("W_h column norms: mean=%.4f, max=%.4f\n", sum_col_norm / HIDDEN_SIZE, max_col_norm);

    /* Frobenius norm */
    double frob = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            frob += rnn.Wh[i][j] * rnn.Wh[i][j];
    printf("W_h Frobenius norm: %.4f\n", sqrt(frob));

    /* W_x column norms for common bytes */
    printf("\nW_x column norms for frequent bytes:\n");
    unsigned char freq_bytes[] = " <>/aeiounrstldhcmpfgbwyvkxjqz";
    for (int bi = 0; freq_bytes[bi]; bi++) {
        int b = freq_bytes[bi];
        double norm = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++)
            norm += rnn.Wx[i][b] * rnn.Wx[i][b];
        char safe = (b >= 32 && b < 127) ? b : '.';
        printf("  '%c' (0x%02x): %.4f\n", safe, b, sqrt(norm));
    }

    return 0;
}
