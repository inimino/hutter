/*
 * construct_skip_mlp.c — Multi-offset construction with MLP readout.
 *
 * Same as construct_skip_greedy.c but with a 2-layer MLP readout
 * (hidden ReLU layer) instead of linear softmax. Tests whether
 * the readout loss (0.147 bpc for greedy-8) is due to the
 * linearity of the softmax or something else.
 *
 * Usage: construct_skip_mlp <data_file> <offsets> <mlp_hidden> [test_file]
 *   offsets: comma-separated, e.g. "1,8,20,3,27,2,12,7"
 *   mlp_hidden: size of the MLP hidden layer (e.g. 64, 128, 256)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_MLP 512

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

int parse_offsets(const char* str, int* offsets, int max_offsets) {
    int n = 0;
    const char* p = str;
    while (*p && n < max_offsets) {
        offsets[n++] = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return n;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <data_file> <offsets> <mlp_hidden> [test_file]\n", argv[0]);
        return 1;
    }

    int data_len;
    unsigned char* data = load_data(argv[1], &data_len);

    int offsets[32];
    int n_offsets = parse_offsets(argv[2], offsets, 32);
    int mlp_hidden = atoi(argv[3]);
    if (mlp_hidden > MAX_MLP) mlp_hidden = MAX_MLP;

    printf("Data: %d bytes\n", data_len);
    printf("Offsets (%d):", n_offsets);
    int max_offset = 0;
    for (int i = 0; i < n_offsets; i++) {
        printf(" %d", offsets[i]);
        if (offsets[i] > max_offset) max_offset = offsets[i];
    }
    printf("\nMLP hidden: %d\n", mlp_hidden);

    int neurons_per_offset = HIDDEN_SIZE / n_offsets;

    /* Generate random hash patterns */
    float hash_sign[256][HIDDEN_SIZE];
    srand(42);
    for (int j = 0; j < neurons_per_offset; j++) {
        for (int x = 0; x < 256; x++) {
            int sign = (rand() % 2) * 2 - 1;
            for (int g = 0; g < n_offsets; g++)
                hash_sign[x][g * neurons_per_offset + j] = (float)sign;
        }
    }

    /* Construct h vectors */
    int start = max_offset;
    int n_valid = data_len - 1 - start;
    if (n_valid <= 0) { fprintf(stderr, "Data too short\n"); return 1; }
    printf("Valid positions: %d\n\n", n_valid);

    float* H = malloc(n_valid * HIDDEN_SIZE * sizeof(float));
    unsigned char* targets = malloc(n_valid);

    for (int idx = 0; idx < n_valid; idx++) {
        int t = idx + start;
        float* hv = H + idx * HIDDEN_SIZE;
        for (int g = 0; g < n_offsets; g++) {
            int past_pos = t - offsets[g] + 1;
            unsigned char byte = data[past_pos];
            for (int j = 0; j < neurons_per_offset; j++)
                hv[g * neurons_per_offset + j] =
                    (hash_sign[byte][g * neurons_per_offset + j] > 0) ? 1.0f : -1.0f;
        }
        for (int j = n_offsets * neurons_per_offset; j < HIDDEN_SIZE; j++)
            hv[j] = 0.0f;
        targets[idx] = data[t + 1];
    }

    /* MLP: h (128) -> W1 (mlp_hidden x 128) + b1 -> ReLU -> W2 (256 x mlp_hidden) + b2 -> softmax */
    float* W1 = calloc(mlp_hidden * HIDDEN_SIZE, sizeof(float));
    float* b1 = calloc(mlp_hidden, sizeof(float));
    float* W2 = calloc(OUTPUT_SIZE * mlp_hidden, sizeof(float));
    float* b2 = calloc(OUTPUT_SIZE, sizeof(float));

    /* Xavier init for W1 */
    srand(123);
    float scale1 = sqrtf(2.0f / (HIDDEN_SIZE + mlp_hidden));
    for (int i = 0; i < mlp_hidden * HIDDEN_SIZE; i++)
        W1[i] = scale1 * ((float)rand() / RAND_MAX * 2 - 1);
    float scale2 = sqrtf(2.0f / (mlp_hidden + OUTPUT_SIZE));
    for (int i = 0; i < OUTPUT_SIZE * mlp_hidden; i++)
        W2[i] = scale2 * ((float)rand() / RAND_MAX * 2 - 1);

    /* Initialize b2 from unigram */
    int marginal[256] = {0};
    for (int t = 0; t < data_len; t++) marginal[data[t]]++;
    for (int y = 0; y < OUTPUT_SIZE; y++)
        b2[y] = logf((marginal[y] + 0.5f) / (data_len + 128.0f));

    /* Gradient descent */
    printf("Optimizing MLP (%d -> %d -> %d)...\n", HIDDEN_SIZE, mlp_hidden, OUTPUT_SIZE);

    float lr = 0.1f;
    int n_epochs = 2000;
    float best_bpc = 1e9f;

    float* dW1 = malloc(mlp_hidden * HIDDEN_SIZE * sizeof(float));
    float* db1 = malloc(mlp_hidden * sizeof(float));
    float* dW2 = malloc(OUTPUT_SIZE * mlp_hidden * sizeof(float));
    float* db2_grad = malloc(OUTPUT_SIZE * sizeof(float));
    float* a1 = malloc(mlp_hidden * sizeof(float)); /* pre-activation */
    float* z1 = malloc(mlp_hidden * sizeof(float)); /* post-ReLU */

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        memset(dW1, 0, mlp_hidden * HIDDEN_SIZE * sizeof(float));
        memset(db1, 0, mlp_hidden * sizeof(float));
        memset(dW2, 0, OUTPUT_SIZE * mlp_hidden * sizeof(float));
        memset(db2_grad, 0, OUTPUT_SIZE * sizeof(float));
        double total_loss = 0;

        for (int i = 0; i < n_valid; i++) {
            float* hv = H + i * HIDDEN_SIZE;
            int y_true = targets[i];

            /* Forward: layer 1 */
            for (int k = 0; k < mlp_hidden; k++) {
                float sum = b1[k];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    sum += W1[k * HIDDEN_SIZE + j] * hv[j];
                a1[k] = sum;
                z1[k] = (sum > 0) ? sum : 0; /* ReLU */
            }

            /* Forward: layer 2 (logits) */
            float logits[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float sum = b2[y];
                for (int k = 0; k < mlp_hidden; k++)
                    sum += W2[y * mlp_hidden + k] * z1[k];
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

            /* Backward: d(loss)/d(logits) */
            float dlogits[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++)
                dlogits[y] = probs[y] - (y == y_true ? 1.0f : 0.0f);

            /* Gradients for W2, b2 */
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                db2_grad[y] += dlogits[y];
                for (int k = 0; k < mlp_hidden; k++)
                    dW2[y * mlp_hidden + k] += dlogits[y] * z1[k];
            }

            /* Backward through layer 2 to get dz1 */
            float dz1[MAX_MLP];
            memset(dz1, 0, mlp_hidden * sizeof(float));
            for (int y = 0; y < OUTPUT_SIZE; y++)
                for (int k = 0; k < mlp_hidden; k++)
                    dz1[k] += W2[y * mlp_hidden + k] * dlogits[y];

            /* Backward through ReLU */
            float da1[MAX_MLP];
            for (int k = 0; k < mlp_hidden; k++)
                da1[k] = (a1[k] > 0) ? dz1[k] : 0;

            /* Gradients for W1, b1 */
            for (int k = 0; k < mlp_hidden; k++) {
                db1[k] += da1[k];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    dW1[k * HIDDEN_SIZE + j] += da1[k] * hv[j];
            }
        }

        float cur_bpc = total_loss / n_valid;

        /* Update */
        float scale_lr = lr / n_valid;
        for (int i = 0; i < mlp_hidden * HIDDEN_SIZE; i++)
            W1[i] -= scale_lr * dW1[i];
        for (int k = 0; k < mlp_hidden; k++)
            b1[k] -= scale_lr * db1[k];
        for (int i = 0; i < OUTPUT_SIZE * mlp_hidden; i++)
            W2[i] -= scale_lr * dW2[i];
        for (int y = 0; y < OUTPUT_SIZE; y++)
            b2[y] -= scale_lr * db2_grad[y];

        if (cur_bpc < best_bpc) best_bpc = cur_bpc;

        if (epoch % 200 == 0 || epoch == n_epochs - 1) {
            printf("  epoch %4d: %.4f bpc (best: %.4f)\n",
                   epoch, cur_bpc, best_bpc);
        }

        if (epoch == 500) lr *= 0.5f;
        if (epoch == 1000) lr *= 0.5f;
        if (epoch == 1500) lr *= 0.5f;
    }

    printf("\nTrain bpc: %.4f\n", best_bpc);

    /* Test evaluation */
    if (argc > 4) {
        int test_len;
        unsigned char* test_data = load_data(argv[4], &test_len);
        int test_start = max_offset;
        int test_valid = test_len - 1 - test_start;
        if (test_valid > 0) {
            double test_loss = 0;
            for (int idx = 0; idx < test_valid; idx++) {
                int t = idx + test_start;
                float hv[HIDDEN_SIZE];
                for (int g = 0; g < n_offsets; g++) {
                    int past_pos = t - offsets[g] + 1;
                    unsigned char byte = test_data[past_pos];
                    for (int j = 0; j < neurons_per_offset; j++)
                        hv[g * neurons_per_offset + j] =
                            (hash_sign[byte][g * neurons_per_offset + j] > 0) ? 1.0f : -1.0f;
                }
                for (int j = n_offsets * neurons_per_offset; j < HIDDEN_SIZE; j++)
                    hv[j] = 0.0f;

                /* Forward: layer 1 */
                for (int k = 0; k < mlp_hidden; k++) {
                    float sum = b1[k];
                    for (int j = 0; j < HIDDEN_SIZE; j++)
                        sum += W1[k * HIDDEN_SIZE + j] * hv[j];
                    z1[k] = (sum > 0) ? sum : 0;
                }
                /* Forward: layer 2 */
                float logits[OUTPUT_SIZE];
                for (int y = 0; y < OUTPUT_SIZE; y++) {
                    float sum = b2[y];
                    for (int k = 0; k < mlp_hidden; k++)
                        sum += W2[y * mlp_hidden + k] * z1[k];
                    logits[y] = sum;
                }
                float maxl = logits[0];
                for (int y = 1; y < OUTPUT_SIZE; y++)
                    if (logits[y] > maxl) maxl = logits[y];
                double lse = 0;
                for (int i = 0; i < OUTPUT_SIZE; i++)
                    lse += exp(logits[i] - maxl);
                lse = maxl + log(lse);
                test_loss += (lse - logits[test_data[t+1]]) / log(2.0);
            }
            printf("Test bpc:  %.4f (%d positions)\n",
                   (float)(test_loss / test_valid), test_valid);
        }
        free(test_data);
    }

    free(H); free(targets);
    free(W1); free(b1); free(W2); free(b2);
    free(dW1); free(db1); free(dW2); free(db2_grad);
    free(a1); free(z1);
    free(data);
    return 0;
}
