/*
 * construct_rnn.c — Construct RNN weights from data patterns, no training.
 *
 * Given a dataset, constructs an RNN whose weights implement the empirical
 * conditional distributions directly.
 *
 * Phase 1: Bigram predictor.
 *   - Compute P(y|x) from data
 *   - Construct Wx, Wh, bh, Wy, by to implement this lookup
 *   - Evaluate bpc and compare with trained sat-rnn and counting baselines
 *
 * The key insight: neuron permutation symmetry means we're free to assign
 * any neuron to any function. There's no "right" assignment — only
 * assignments that implement the desired patterns.
 *
 * Architecture: 256 input (one-hot) → 128 hidden (tanh) → 256 output (softmax)
 *
 * Usage: construct_rnn <data_file> [output_model] [skip_offset]
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

unsigned char* load_data(const char* path, int* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = malloc(*len);
    fread(data, 1, *len, f);
    fclose(f);
    return data;
}

void save_model(Model* m, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(m->Wx, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
    fwrite(m->Wh, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    fwrite(m->bh, sizeof(float), HIDDEN_SIZE, f);
    fwrite(m->Wy, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fwrite(m->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

void load_model(Model* m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fread(m->Wx, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
    fread(m->Wh, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    fread(m->bh, sizeof(float), HIDDEN_SIZE, f);
    fread(m->Wy, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(m->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

float eval_bpc(Model* m, unsigned char* data, int len) {
    float h[HIDDEN_SIZE] = {0};
    float h_new[HIDDEN_SIZE];
    double total_loss = 0;
    int count = 0;

    for (int t = 0; t < len - 1; t++) {
        /* h_new = tanh(bh + Wx*onehot(x_t) + Wh*h) */
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = m->bh[i] + m->Wx[i][data[t]];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += m->Wh[i][j] * h[j];
            h_new[i] = tanhf(sum);
        }

        /* logits = by + Wy*h_new */
        float logits[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float sum = m->by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += m->Wy[i][j] * h_new[j];
            logits[i] = sum;
        }

        /* cross-entropy */
        float maxl = logits[0];
        for (int i = 1; i < OUTPUT_SIZE; i++)
            if (logits[i] > maxl) maxl = logits[i];
        double lse = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++)
            lse += exp(logits[i] - maxl);
        lse = maxl + log(lse);
        total_loss += (lse - logits[data[t+1]]) / log(2.0);
        count++;

        memcpy(h, h_new, sizeof(h));
    }
    return total_loss / count;
}

/*
 * Solve Ax = b via Gaussian elimination (n×n system).
 * A is overwritten. b is overwritten with solution.
 */
void solve_system(double* A, double* b, int n) {
    /* Forward elimination with partial pivoting */
    for (int col = 0; col < n; col++) {
        /* Find pivot */
        int pivot = col;
        double maxval = fabs(A[col * n + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(A[row * n + col]);
            if (v > maxval) { maxval = v; pivot = row; }
        }
        /* Swap rows */
        if (pivot != col) {
            for (int j = 0; j < n; j++) {
                double tmp = A[col * n + j];
                A[col * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            double tmp = b[col]; b[col] = b[pivot]; b[pivot] = tmp;
        }
        /* Eliminate */
        double diag = A[col * n + col];
        if (fabs(diag) < 1e-12) continue;
        for (int row = col + 1; row < n; row++) {
            double factor = A[row * n + col] / diag;
            for (int j = col; j < n; j++)
                A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }
    /* Back substitution */
    for (int row = n - 1; row >= 0; row--) {
        double sum = b[row];
        for (int j = row + 1; j < n; j++)
            sum -= A[row * n + j] * b[j];
        double diag = A[row * n + row];
        b[row] = (fabs(diag) > 1e-12) ? sum / diag : 0;
    }
}

/*
 * Bigram construction using proper least-squares.
 *
 * The RNN computes: logits = by + Wy * tanh(Wx_col(x_t))
 * For Wh=0, bh=0, the hidden state depends only on the current input.
 *
 * With 128 hidden neurons and tanh saturation (Wx large), h(x) ∈ {-1,+1}^128.
 * This is a random binary hash of x. With 128 bits, all 256 byte values
 * get distinct hashes (collision prob < 2^{-128}).
 *
 * We want: by + Wy * h(x) = target_logits(x) for all observed x.
 * This is 128 unknowns (Wy row for each y) with up to 256 equations.
 * With n_obs > 128, it's overdetermined → use normal equations:
 *   (H^T H) Wy_col = H^T target_col
 * where H is n_obs × 128 and target is n_obs × 256.
 *
 * Target: the bigram conditional log P(y|x), computed from data with
 * backoff to unigram (matching the pattern-chain counting method).
 */
void construct_bigram(Model* m, unsigned char* data, int len) {
    /* Count bigrams and unigrams */
    int bg[256][256] = {{0}};
    int ctx_total[256] = {0};
    int marginal[256] = {0};

    for (int t = 0; t < len - 1; t++) {
        bg[data[t]][data[t+1]]++;
        ctx_total[data[t]]++;
        marginal[data[t+1]]++;
    }
    marginal[data[0]]++; /* count first byte too for unigram */

    /* Find observed input bytes */
    int obs[256], n_obs = 0;
    for (int x = 0; x < 256; x++)
        if (ctx_total[x] > 0) obs[n_obs++] = x;
    printf("Observed input bytes: %d\n", n_obs);

    /* Construct Wx: random binary hash, scaled to saturate tanh */
    srand(42);
    float scale = 10.0f;
    for (int j = 0; j < HIDDEN_SIZE; j++)
        for (int x = 0; x < INPUT_SIZE; x++)
            m->Wx[j][x] = scale * ((rand() % 2) * 2 - 1);

    /* Compute hidden vectors for observed bytes */
    double H[256][HIDDEN_SIZE]; /* H[i][j] for i-th observed byte */
    for (int i = 0; i < n_obs; i++) {
        int x = obs[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            H[i][j] = tanhf(m->Wx[j][x]); /* ≈ ±1 */
    }

    /* No recurrence for bigram */
    memset(m->Wh, 0, sizeof(m->Wh));
    memset(m->bh, 0, sizeof(m->bh));

    /* Compute by = log unigram distribution */
    for (int y = 0; y < OUTPUT_SIZE; y++)
        m->by[y] = logf((marginal[y] + 0.5f) / (len + 128.0f));

    /* Compute target logits: log P(y|x) using interpolated smoothing.
     * P(y|x) = lambda * bigram_MLE(y|x) + (1-lambda) * unigram(y)
     * This produces a proper probability distribution (sums to 1). */
    float lambda = 0.95f;  /* mostly bigram, light unigram mix */
    double uni_prob[256];
    double uni_total = len;
    for (int y = 0; y < 256; y++)
        uni_prob[y] = (marginal[y] + 0.5) / (uni_total + 128.0);

    double target[256][256]; /* target[obs_idx][y] */
    for (int i = 0; i < n_obs; i++) {
        int x = obs[i];
        for (int y = 0; y < 256; y++) {
            double bg_p = (double)bg[x][y] / ctx_total[x]; /* 0 if unseen */
            double p = lambda * bg_p + (1.0 - lambda) * uni_prob[y];
            target[i][y] = log(p);
        }
    }

    /* Build normal equations: (H^T H) w = H^T target
     * H is n_obs × 128, so H^T H is 128 × 128.
     * We solve independently for each output y. */
    double* HtH = calloc(HIDDEN_SIZE * HIDDEN_SIZE, sizeof(double));
    for (int j = 0; j < HIDDEN_SIZE; j++)
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            double sum = 0;
            for (int i = 0; i < n_obs; i++)
                sum += H[i][j] * H[i][k];
            HtH[j * HIDDEN_SIZE + k] = sum;
        }

    /* Add small regularization for numerical stability */
    for (int j = 0; j < HIDDEN_SIZE; j++)
        HtH[j * HIDDEN_SIZE + j] += 0.01;

    printf("Solving %d × %d normal equations for each of %d outputs...\n",
           HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    for (int y = 0; y < OUTPUT_SIZE; y++) {
        /* H^T * (target_col - by) */
        double rhs[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = 0;
            for (int i = 0; i < n_obs; i++)
                sum += H[i][j] * (target[i][y] - m->by[y]);
            rhs[j] = sum;
        }

        /* Copy HtH (solve_system overwrites it) */
        double* A = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double));
        memcpy(A, HtH, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(double));

        solve_system(A, rhs, HIDDEN_SIZE);

        for (int j = 0; j < HIDDEN_SIZE; j++)
            m->Wy[y][j] = rhs[j];

        free(A);
    }

    free(HtH);

    /* Verify reconstruction error */
    double total_err = 0;
    double max_err = 0;
    for (int i = 0; i < n_obs; i++) {
        int x = obs[i];
        for (int y = 0; y < 256; y++) {
            double pred = m->by[y];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                pred += m->Wy[y][j] * H[i][j];
            double err = fabs(pred - target[i][y]);
            total_err += err * err;
            if (err > max_err) max_err = err;
        }
    }
    printf("Reconstruction RMSE: %.6f, max error: %.6f\n",
           sqrt(total_err / (n_obs * 256)), max_err);
}

/*
 * Counting baselines for comparison.
 */
void counting_baselines(unsigned char* data, int len) {
    /* Bigram with backoff */
    int bg[256][256] = {{0}};
    int ctx_total[256] = {0};
    int marginal[256] = {0};
    for (int t = 0; t < len - 1; t++) {
        bg[data[t]][data[t+1]]++;
        ctx_total[data[t]]++;
    }
    for (int t = 0; t < len; t++) marginal[data[t]]++;

    /* Bigram bpc with backoff */
    double bigram_bpc = 0;
    for (int t = 0; t < len - 1; t++) {
        double p;
        if (bg[data[t]][data[t+1]] > 0) {
            p = (double)bg[data[t]][data[t+1]] / ctx_total[data[t]];
        } else {
            p = (marginal[data[t+1]] + 0.5) / (len + 128.0);
        }
        bigram_bpc += -log2(p);
    }
    bigram_bpc /= (len - 1);

    /* Bigram bpc with Laplace smoothing (for comparison) */
    double bigram_lap = 0;
    for (int t = 0; t < len - 1; t++) {
        double p = (bg[data[t]][data[t+1]] + 1.0) / (ctx_total[data[t]] + 256.0);
        bigram_lap += -log2(p);
    }
    bigram_lap /= (len - 1);

    /* Marginal bpc */
    double marg_bpc = 0;
    for (int t = 1; t < len; t++) {
        double p = (marginal[data[t]] + 0.5) / (len + 128.0);
        marg_bpc += -log2(p);
    }
    marg_bpc /= (len - 1);

    printf("\nCounting baselines:\n");
    printf("  Marginal:               %.4f bpc\n", marg_bpc);
    printf("  Bigram (backoff):       %.4f bpc\n", bigram_bpc);
    printf("  Bigram (Laplace):       %.4f bpc\n", bigram_lap);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file> [output_model] [skip_offset]\n", argv[0]);
        return 1;
    }

    int data_len;
    unsigned char* data = load_data(argv[1], &data_len);
    printf("Data: %d bytes\n\n", data_len);

    const char* output_path = (argc > 2) ? argv[2] : "/tmp/constructed.bin";

    Model m;
    memset(&m, 0, sizeof(m));

    printf("=== Bigram RNN construction (no training) ===\n\n");
    construct_bigram(&m, data, data_len);

    /* Evaluate */
    float bpc = eval_bpc(&m, data, data_len);
    printf("\nConstructed RNN bpc: %.4f\n", bpc);

    counting_baselines(data, data_len);

    /* Compare with trained model if available */
    FILE* test = fopen("docs/archive/20260206/sat_model.bin", "rb");
    if (test) {
        fclose(test);
        Model trained;
        load_model(&trained, "docs/archive/20260206/sat_model.bin");
        float trained_bpc = eval_bpc(&trained, data, data_len);
        printf("  Trained sat-rnn:        %.4f bpc\n", trained_bpc);
    }

    save_model(&m, output_path);
    printf("\nModel saved to %s\n", output_path);

    /* If a second data file is given as 4th arg, cross-evaluate */
    if (argc > 4) {
        int test_len;
        unsigned char* test_data = load_data(argv[4], &test_len);
        float test_bpc = eval_bpc(&m, test_data, test_len);
        printf("\nCross-evaluation on %s (%d bytes): %.4f bpc\n",
               argv[4], test_len, test_bpc);
        free(test_data);
    }

    free(data);
    return 0;
}
