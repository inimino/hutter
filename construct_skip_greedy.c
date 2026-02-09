/*
 * construct_skip_greedy.c — Construct an RNN readout from greedy skip offsets.
 *
 * Instead of a shift-register (contiguous offsets), directly construct
 * hidden vectors from specified non-contiguous offsets.
 *
 * For each position t, h_t = concat(hash(data[t-offset_k+1]) for each k).
 * Then optimize Wy via gradient descent on cross-entropy.
 *
 * This answers: "given the ideal h vectors for these offsets, how well
 * does the readout work?" — the ceiling for any Wh implementation.
 *
 * Usage: construct_skip_greedy <data_file> <offsets> [test_file]
 *   offsets: comma-separated, e.g. "1,8,20,3"
 *
 * Compare with construct_skip.c (contiguous offsets via shift register).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256

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
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data_file> <offsets> [test_file]\n", argv[0]);
        fprintf(stderr, "  offsets: comma-separated, e.g. \"1,8,20,3\"\n");
        return 1;
    }

    int data_len;
    unsigned char* data = load_data(argv[1], &data_len);

    int offsets[32];
    int n_offsets = parse_offsets(argv[2], offsets, 32);

    printf("Data: %d bytes\n", data_len);
    printf("Offsets (%d):", n_offsets);
    int max_offset = 0;
    for (int i = 0; i < n_offsets; i++) {
        printf(" %d", offsets[i]);
        if (offsets[i] > max_offset) max_offset = offsets[i];
    }
    printf("\n");

    int neurons_per_offset = HIDDEN_SIZE / n_offsets;
    printf("Neurons per offset: %d (total %d, unused %d)\n",
           neurons_per_offset, neurons_per_offset * n_offsets,
           HIDDEN_SIZE - neurons_per_offset * n_offsets);

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
    int start = max_offset; /* warmup: need max_offset positions */
    int n_valid = data_len - 1 - start;
    if (n_valid <= 0) {
        fprintf(stderr, "Data too short for max offset %d\n", max_offset);
        return 1;
    }
    printf("Valid positions: %d (warmup: %d)\n\n", n_valid, start);

    float* H = malloc(n_valid * HIDDEN_SIZE * sizeof(float));
    unsigned char* targets = malloc(n_valid);

    for (int idx = 0; idx < n_valid; idx++) {
        int t = idx + start; /* position in data */
        float* hv = H + idx * HIDDEN_SIZE;

        /* For each offset group, hash the corresponding input byte */
        for (int g = 0; g < n_offsets; g++) {
            int past_pos = t - offsets[g] + 1;
            /* past_pos should be >= 0 since t >= max_offset */
            unsigned char byte = data[past_pos];
            for (int j = 0; j < neurons_per_offset; j++) {
                hv[g * neurons_per_offset + j] =
                    (hash_sign[byte][g * neurons_per_offset + j] > 0) ? 1.0f : -1.0f;
            }
        }
        /* Zero out unused neurons */
        for (int j = n_offsets * neurons_per_offset; j < HIDDEN_SIZE; j++)
            hv[j] = 0.0f;

        targets[idx] = data[t + 1];
    }

    /* Initialize Wy and by */
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
    memset(Wy, 0, sizeof(Wy));

    /* Initialize by from unigram distribution */
    int marginal[256] = {0};
    for (int t = 0; t < data_len; t++) marginal[data[t]]++;
    for (int y = 0; y < OUTPUT_SIZE; y++)
        by[y] = logf((marginal[y] + 0.5f) / (data_len + 128.0f));

    /* Gradient descent on cross-entropy */
    printf("Optimizing Wy (%d positions, %d features)...\n", n_valid, HIDDEN_SIZE);

    float lr = 0.5f;
    int n_epochs = 1000;
    float best_bpc = 1e9f;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        float dWy[OUTPUT_SIZE][HIDDEN_SIZE];
        float dby[OUTPUT_SIZE];
        memset(dWy, 0, sizeof(dWy));
        memset(dby, 0, sizeof(dby));
        double total_loss = 0;

        for (int i = 0; i < n_valid; i++) {
            float* hv = H + i * HIDDEN_SIZE;
            int y_true = targets[i];

            float logits[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float sum = by[y];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    sum += Wy[y][j] * hv[j];
                logits[y] = sum;
            }

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

            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float grad = probs[y] - (y == y_true ? 1.0f : 0.0f);
                dby[y] += grad;
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    dWy[y][j] += grad * hv[j];
            }
        }

        float cur_bpc = total_loss / n_valid;

        float scale_lr = lr / n_valid;
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            by[y] -= scale_lr * dby[y];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                Wy[y][j] -= scale_lr * dWy[y][j];
        }

        if (cur_bpc < best_bpc) best_bpc = cur_bpc;

        if (epoch % 100 == 0 || epoch == n_epochs - 1) {
            printf("  epoch %4d: %.4f bpc (best: %.4f)\n",
                   epoch, cur_bpc, best_bpc);
        }

        if (epoch == 300) lr *= 0.5f;
        if (epoch == 600) lr *= 0.5f;
        if (epoch == 800) lr *= 0.5f;
    }

    printf("\nTrain bpc: %.4f\n", best_bpc);

    /* Cross-evaluation on test data */
    if (argc > 3) {
        int test_len;
        unsigned char* test_data = load_data(argv[3], &test_len);

        int test_start = max_offset;
        int test_valid = test_len - 1 - test_start;
        if (test_valid <= 0) {
            fprintf(stderr, "Test data too short\n");
        } else {
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

                float logits[OUTPUT_SIZE];
                for (int y = 0; y < OUTPUT_SIZE; y++) {
                    float sum = by[y];
                    for (int j = 0; j < HIDDEN_SIZE; j++)
                        sum += Wy[y][j] * hv[j];
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

    free(H);
    free(targets);
    free(data);
    return 0;
}
