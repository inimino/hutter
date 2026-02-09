/*
 * construct_skip.c — Construct an RNN with skip-pattern recurrence.
 *
 * Extends construct_rnn.c to encode skip-k-gram patterns into Wh.
 *
 * Key idea: partition the 128 neurons into groups:
 *   - Group 0 (neurons 0-15): fresh input encoding (overwritten each step)
 *   - Group 1 (neurons 16-31): carry input from 1 step ago via Wh
 *   - Group 2 (neurons 32-47): carry input from 2 steps ago
 *   - ...
 *   - Group 7 (neurons 112-127): carry input from 7 steps ago
 *
 * Each group has 16 neurons → 2^16 = 65536 possible states → more than
 * enough to distinguish 256 byte values.
 *
 * Wh implements a shift register: at each step, group k copies from
 * group k-1 (with tanh re-saturation). Group 0 is always fresh from Wx.
 *
 * Wy reads from ALL groups simultaneously, combining information from
 * multiple past offsets to predict the output.
 *
 * This directly implements the skip-k-gram construction:
 *   P(y | x_{t-1}, x_{t-2}, ..., x_{t-7}) ≈ softmax(by + Wy * h_t)
 * where h_t contains the hashed identities of the last 8 inputs.
 *
 * Usage: construct_skip <data_file> [output_model] [n_offsets] [test_file]
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
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = m->bh[i] + m->Wx[i][data[t]];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += m->Wh[i][j] * h[j];
            h_new[i] = tanhf(sum);
        }

        float logits[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float sum = m->by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += m->Wy[i][j] * h_new[j];
            logits[i] = sum;
        }

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

/* Gaussian elimination solver */
void solve_system(double* A, double* b, int n) {
    for (int col = 0; col < n; col++) {
        int pivot = col;
        double maxval = fabs(A[col * n + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(A[row * n + col]);
            if (v > maxval) { maxval = v; pivot = row; }
        }
        if (pivot != col) {
            for (int j = 0; j < n; j++) {
                double tmp = A[col * n + j];
                A[col * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            double tmp = b[col]; b[col] = b[pivot]; b[pivot] = tmp;
        }
        double diag = A[col * n + col];
        if (fabs(diag) < 1e-12) continue;
        for (int row = col + 1; row < n; row++) {
            double factor = A[row * n + col] / diag;
            for (int j = col; j < n; j++)
                A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }
    for (int row = n - 1; row >= 0; row--) {
        double sum = b[row];
        for (int j = row + 1; j < n; j++)
            sum -= A[row * n + j] * b[j];
        double diag = A[row * n + row];
        b[row] = (fabs(diag) > 1e-12) ? sum / diag : 0;
    }
}

/*
 * Construct skip-pattern RNN.
 *
 * n_groups = number of offset groups (including fresh input).
 * Group size = HIDDEN_SIZE / n_groups neurons per group.
 *
 * Neurons in group k encode the input from k steps ago.
 * Wh shifts: group k ← group k-1 at each step.
 * Wx writes only to group 0 (fresh input).
 *
 * The target for Wy is the n-gram conditional P(y|x_{t-1},...,x_{t-n_groups+1}).
 */
void construct_skip(Model* m, unsigned char* data, int len, int n_groups) {
    int group_size = HIDDEN_SIZE / n_groups;
    printf("Groups: %d, neurons per group: %d\n", n_groups, group_size);
    printf("Each group hashes a byte from a different offset.\n\n");

    /* Step 1: Generate random hash patterns for each group.
     * All groups use the same hash function (same random seed per group)
     * so that Wh can copy group k-1 → group k by identity. */

    /* Random binary hash: for byte x, group g, neuron j within group:
     * hash[x][g*group_size + j] = ±1 */
    float hash_sign[256][HIDDEN_SIZE];
    srand(42);
    for (int j = 0; j < group_size; j++) {
        for (int x = 0; x < 256; x++) {
            int sign = (rand() % 2) * 2 - 1;
            /* Same hash pattern replicated across all groups */
            for (int g = 0; g < n_groups; g++)
                hash_sign[x][g * group_size + j] = sign;
        }
    }

    /* Step 2: Construct Wx — writes to group 0 only */
    float scale = 10.0f;
    memset(m->Wx, 0, sizeof(m->Wx));
    for (int j = 0; j < group_size; j++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            m->Wx[j][x] = scale * hash_sign[x][j];
        }
    }

    /* Step 3: Construct Wh — shift register.
     * Group k (neurons k*gs...(k+1)*gs-1) copies from group k-1.
     * Since all groups use the same hash, this is a diagonal copy:
     *   Wh[(k)*gs + j][(k-1)*gs + j] = large positive (preserves sign through tanh)
     * Group 0 has no Wh input (overwritten by Wx). */
    memset(m->Wh, 0, sizeof(m->Wh));
    float carry_weight = 5.0f; /* tanh(5 * ±1) = ±0.9999 */
    for (int g = 1; g < n_groups; g++) {
        for (int j = 0; j < group_size; j++) {
            int dst = g * group_size + j;
            int src = (g - 1) * group_size + j;
            m->Wh[dst][src] = carry_weight;
        }
    }

    memset(m->bh, 0, sizeof(m->bh));

    /* Step 4: Simulate the RNN forward pass to get actual h vectors.
     * At each position t (after the warmup period), h_t should contain:
     *   group 0: hash of data[t]    (from Wx)
     *   group 1: hash of data[t-1]  (carried from previous step)
     *   group 2: hash of data[t-2]  (carried 2 steps)
     *   ...
     * We verify this by running the actual forward pass. */

    float h[HIDDEN_SIZE] = {0};
    float h_new[HIDDEN_SIZE];

    /* Collect h vectors for positions where all groups are valid */
    int start = n_groups - 1; /* need n_groups-1 steps of warmup */
    int n_valid = len - 1 - start;
    if (n_valid <= 0) {
        fprintf(stderr, "Data too short for %d groups\n", n_groups);
        return;
    }

    /* Store h vectors and target outputs */
    float* H = calloc(n_valid * HIDDEN_SIZE, sizeof(float));
    unsigned char* targets = malloc(n_valid);

    for (int t = 0; t < len - 1; t++) {
        /* Forward step */
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = m->bh[i] + m->Wx[i][data[t]];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += m->Wh[i][j] * h[j];
            h_new[i] = tanhf(sum);
        }
        memcpy(h, h_new, sizeof(h));

        if (t >= start) {
            int idx = t - start;
            memcpy(H + idx * HIDDEN_SIZE, h, HIDDEN_SIZE * sizeof(float));
            targets[idx] = data[t + 1];
        }
    }

    printf("Valid positions: %d (after %d warmup steps)\n", n_valid, start);

    /* Verify h vectors encode the right bytes */
    int correct = 0, total_checks = 0;
    for (int idx = 0; idx < n_valid && idx < 20; idx++) {
        int t = idx + start;
        float* hv = H + idx * HIDDEN_SIZE;
        /* Check group 0 matches hash of data[t] */
        int match = 1;
        for (int j = 0; j < group_size; j++) {
            float expected = (hash_sign[data[t]][j] > 0) ? 1.0f : -1.0f;
            if ((hv[j] > 0) != (expected > 0)) { match = 0; break; }
        }
        if (match) correct++;
        total_checks++;

        /* Check group 1 matches hash of data[t-1] */
        if (n_groups > 1 && t >= 1) {
            match = 1;
            for (int j = 0; j < group_size; j++) {
                float expected = (hash_sign[data[t-1]][j] > 0) ? 1.0f : -1.0f;
                if ((hv[group_size + j] > 0) != (expected > 0)) { match = 0; break; }
            }
        }
    }
    printf("Group 0 encoding accuracy (first 20): %d/%d\n", correct, total_checks);

    /* Step 5: Build n-gram conditional distribution from data */
    /* Count n-grams using the actual data (not the h vectors) */
    /* For each valid position, the context is data[t], data[t-1], ..., data[t-n_groups+1] */

    /* We use the h vectors directly to solve for Wy.
     * This automatically handles the n-gram counting because h encodes the context. */

    /* Compute marginal for by */
    int marginal[256] = {0};
    for (int t = 0; t < len; t++) marginal[data[t]]++;

    for (int y = 0; y < OUTPUT_SIZE; y++)
        m->by[y] = logf((marginal[y] + 0.5f) / (len + 128.0f));

    /* Build target: for each position, what should the output be?
     * We use the actual conditional: count(context, y) / count(context)
     * But since contexts can be very specific with n_groups > 2,
     * most will appear only once. We use the observed prediction as target. */

    /* Simple approach: for each position, target logits = one-hot of actual y
     * minus bias. This is equivalent to regressing h → y_onehot.
     * But this overfits. Better: compute empirical conditional per h vector. */

    /* Actually, the right approach is: since h vectors are hash codes of the
     * n-gram context, identical contexts produce identical h vectors.
     * Group h vectors by their binary sign pattern and count conditionals. */

    /* Optimize Wy and by via gradient descent on cross-entropy.
     * Wx and Wh are frozen — we only optimize the readout layer.
     * This is NOT RNN training (no BPTT). It's logistic regression
     * on the fixed hidden representations. */
    printf("Optimizing Wy via gradient descent (%d positions, %d features)...\n",
           n_valid, HIDDEN_SIZE);

    float lr = 0.5f;
    int n_epochs = 1000;
    float best_bpc = 1e9;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        /* Compute gradients over all positions */
        float dWy[OUTPUT_SIZE][HIDDEN_SIZE];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));

        double total_loss = 0;

        for (int i = 0; i < n_valid; i++) {
            float* hv = H + i * HIDDEN_SIZE;
            int y_true = targets[i];

            /* Forward: logits = by + Wy * h */
            float logits[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float sum = m->by[y];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    sum += m->Wy[y][j] * hv[j];
                logits[y] = sum;
            }

            /* Softmax */
            float maxl = logits[0];
            for (int y = 1; y < OUTPUT_SIZE; y++)
                if (logits[y] > maxl) maxl = logits[y];
            float probs[OUTPUT_SIZE];
            float sum_exp = 0;
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                probs[y] = expf(logits[y] - maxl);
                sum_exp += probs[y];
            }
            for (int y = 0; y < OUTPUT_SIZE; y++)
                probs[y] /= sum_exp;

            total_loss += -logf(probs[y_true] + 1e-10f) / logf(2.0f);

            /* Gradient: d(loss)/d(logit_y) = probs[y] - (y == y_true) */
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float grad = probs[y] - (y == y_true ? 1.0f : 0.0f);
                dby[y] += grad;
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    dWy[y][j] += grad * hv[j];
            }
        }

        float cur_bpc = total_loss / n_valid;

        /* Update */
        float scale_lr = lr / n_valid;
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            m->by[y] -= scale_lr * dby[y];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                m->Wy[y][j] -= scale_lr * dWy[y][j];
        }

        if (cur_bpc < best_bpc) best_bpc = cur_bpc;

        if (epoch % 50 == 0 || epoch == n_epochs - 1) {
            printf("  epoch %4d: %.4f bpc (best: %.4f)\n",
                   epoch, cur_bpc, best_bpc);
        }

        /* Decay learning rate */
        if (epoch == 300) lr *= 0.5f;
        if (epoch == 600) lr *= 0.5f;
        if (epoch == 800) lr *= 0.5f;
    }

    free(H);
    free(targets);

    printf("Skip-pattern construction complete.\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file> [output_model] [n_offsets] [test_file]\n",
                argv[0]);
        return 1;
    }

    int data_len;
    unsigned char* data = load_data(argv[1], &data_len);
    printf("Data: %d bytes\n\n", data_len);

    const char* output_path = (argc > 2) ? argv[2] : "/tmp/constructed_skip.bin";
    int n_groups = (argc > 3) ? atoi(argv[3]) : 4;

    Model m;
    memset(&m, 0, sizeof(m));

    printf("=== Skip-pattern RNN construction (no training) ===\n");
    printf("=== %d offset groups ===\n\n", n_groups);
    construct_skip(&m, data, data_len, n_groups);

    float bpc = eval_bpc(&m, data, data_len);
    printf("\nConstructed RNN bpc (train): %.4f\n", bpc);

    /* Compare with sat-rnn */
    FILE* test = fopen("docs/archive/20260206/sat_model.bin", "rb");
    if (test) {
        fclose(test);
        Model trained;
        load_model(&trained, "docs/archive/20260206/sat_model.bin");
        float trained_bpc = eval_bpc(&trained, data, data_len);
        printf("Trained sat-rnn bpc:        %.4f\n", trained_bpc);
    }

    save_model(&m, output_path);
    printf("\nModel saved to %s\n", output_path);

    if (argc > 4) {
        int test_len;
        unsigned char* test_data = load_data(argv[4], &test_len);
        float test_bpc = eval_bpc(&m, test_data, test_len);
        printf("Cross-eval on %s: %.4f bpc\n", argv[4], test_bpc);
        free(test_data);
    }

    free(data);
    return 0;
}
