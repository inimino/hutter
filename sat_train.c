/*
 * sat_train: Train RNN on small data, save checkpoints.
 * Usage: sat_train <datafile> <model_prefix> <max_epochs> [checkpoint_every]
 *
 * Saves: <model_prefix>_epoch<N>.bin at each checkpoint
 *        <model_prefix>_best.bin for best model seen
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define BPTT_LENGTH 50

typedef struct {
    float Wx[HIDDEN_SIZE][INPUT_SIZE];
    float Wh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bh[HIDDEN_SIZE];
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
    float h[HIDDEN_SIZE];
} RNN;

void rnn_init(RNN* rnn) {
    srand(42);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++)
            rnn->Wx[i][j] = ((float)rand() / RAND_MAX - 0.5f) / HIDDEN_SIZE;
        for (int j = 0; j < HIDDEN_SIZE; j++)
            rnn->Wh[i][j] = ((float)rand() / RAND_MAX - 0.5f) / HIDDEN_SIZE;
        rnn->bh[i] = 0.0f;
        rnn->h[i] = 0.0f;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)
            rnn->Wy[i][j] = ((float)rand() / RAND_MAX - 0.5f) / HIDDEN_SIZE;
        rnn->by[i] = 0.0f;
    }
}

void save_model(RNN* rnn, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("save_model"); return; }
    fwrite(rnn->Wx, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
    fwrite(rnn->Wh, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    fwrite(rnn->bh, sizeof(float), HIDDEN_SIZE, f);
    fwrite(rnn->Wy, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fwrite(rnn->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

void softmax(float* logits, float* probs, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

double train_pass(RNN* rnn, unsigned char* data, int len, float lr) {
    double total_loss = 0.0;
    int total_chars = 0;

    for (int pos = 0; pos < len - 1; pos += BPTT_LENGTH) {
        int chunk_len = BPTT_LENGTH + 1;
        if (pos + chunk_len > len) chunk_len = len - pos;
        if (chunk_len < 2) break;

        unsigned char* seq = data + pos;
        int seq_len = chunk_len;

        float hs[BPTT_LENGTH + 1][HIDDEN_SIZE];
        float ps[BPTT_LENGTH][OUTPUT_SIZE];
        float step_loss = 0.0f;

        memcpy(hs[0], rnn->h, sizeof(float) * HIDDEN_SIZE);

        for (int t = 0; t < seq_len - 1; t++) {
            unsigned char x = seq[t];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                float s = rnn->bh[i] + rnn->Wx[i][x];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    s += rnn->Wh[i][j] * hs[t][j];
                hs[t + 1][i] = tanhf(s);
            }
            float ys[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                float s = rnn->by[i];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    s += rnn->Wy[i][j] * hs[t + 1][j];
                ys[i] = s;
            }
            softmax(ys, ps[t], OUTPUT_SIZE);
            float p = ps[t][seq[t + 1]];
            if (p < 1e-8f) p = 1e-8f;
            step_loss += -logf(p);
        }

        /* Backward pass */
        float dh_next[HIDDEN_SIZE] = {0};
        float dWx[HIDDEN_SIZE][INPUT_SIZE] = {{0}};
        float dWh[HIDDEN_SIZE][HIDDEN_SIZE] = {{0}};
        float dbh[HIDDEN_SIZE] = {0};
        float dWy[OUTPUT_SIZE][HIDDEN_SIZE] = {{0}};
        float dby[OUTPUT_SIZE] = {0};

        for (int t = seq_len - 2; t >= 0; t--) {
            unsigned char x = seq[t];
            unsigned char target = seq[t + 1];

            float dy[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++)
                dy[i] = ps[t][i] - (i == target ? 1.0f : 0.0f);

            for (int i = 0; i < OUTPUT_SIZE; i++) {
                dby[i] += dy[i];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    dWy[i][j] += dy[i] * hs[t + 1][j];
            }

            float dh[HIDDEN_SIZE] = {0};
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                for (int i = 0; i < OUTPUT_SIZE; i++)
                    dh[j] += rnn->Wy[i][j] * dy[i];
                dh[j] += dh_next[j];
            }

            float dh_raw[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++)
                dh_raw[i] = dh[i] * (1.0f - hs[t + 1][i] * hs[t + 1][i]);

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                dbh[i] += dh_raw[i];
                dWx[i][x] += dh_raw[i];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    dWh[i][j] += dh_raw[i] * hs[t][j];
            }

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dh_next[j] = 0.0f;
                for (int i = 0; i < HIDDEN_SIZE; i++)
                    dh_next[j] += rnn->Wh[i][j] * dh_raw[i];
            }
        }

        /* Gradient clipping */
        float norm = 0.0f;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++)
                norm += dWx[i][j] * dWx[i][j];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                norm += dWh[i][j] * dWh[i][j];
            norm += dbh[i] * dbh[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++)
                norm += dWy[i][j] * dWy[i][j];
            norm += dby[i] * dby[i];
        }
        norm = sqrtf(norm);
        float scale = (norm > 5.0f) ? 5.0f / norm : 1.0f;

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++)
                rnn->Wx[i][j] -= lr * scale * dWx[i][j];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                rnn->Wh[i][j] -= lr * scale * dWh[i][j];
            rnn->bh[i] -= lr * scale * dbh[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++)
                rnn->Wy[i][j] -= lr * scale * dWy[i][j];
            rnn->by[i] -= lr * scale * dby[i];
        }

        memcpy(rnn->h, hs[seq_len - 1], sizeof(float) * HIDDEN_SIZE);
        total_loss += step_loss;
        total_chars += seq_len - 1;
    }

    return total_loss / (total_chars * logf(2.0));
}

double eval_bpc(RNN* rnn, unsigned char* data, int len) {
    float h[HIDDEN_SIZE];
    memset(h, 0, sizeof(h));
    double total_loss = 0.0;

    for (int t = 0; t < len - 1; t++) {
        unsigned char x = data[t];
        unsigned char target = data[t + 1];
        float h_new[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float s = rnn->bh[i] + rnn->Wx[i][x];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                s += rnn->Wh[i][j] * h[j];
            h_new[i] = tanhf(s);
        }
        memcpy(h, h_new, sizeof(h));
        float logits[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float s = rnn->by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                s += rnn->Wy[i][j] * h[j];
            logits[i] = s;
        }
        float probs[OUTPUT_SIZE];
        softmax(logits, probs, OUTPUT_SIZE);
        float p = probs[target];
        if (p < 1e-8f) p = 1e-8f;
        total_loss += -logf(p);
    }
    return total_loss / ((len - 1) * logf(2.0));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <datafile> <model_prefix> <max_epochs> [ckpt_every]\n", argv[0]);
        return 1;
    }

    const char* datafile = argv[1];
    const char* prefix = argv[2];
    int max_epochs = atoi(argv[3]);
    int ckpt_every = (argc >= 5) ? atoi(argv[4]) : 500;

    FILE* f = fopen(datafile, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    int len = (int)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = malloc(len);
    fread(data, 1, len, f);
    fclose(f);

    printf("sat_train: %d bytes, %d epochs, ckpt every %d\n", len, max_epochs, ckpt_every);

    RNN rnn;
    rnn_init(&rnn);

    float lr_init = 0.01f;
    double best_bpc = 999.0;
    RNN best_rnn;
    char path[512];

    printf("%8s %10s %10s %8s\n", "epoch", "train_bpc", "eval_bpc", "lr");

    for (int epoch = 1; epoch <= max_epochs; epoch++) {
        float lr = lr_init * 0.5f * (1.0f + cosf(3.14159265f * epoch / max_epochs));
        if (lr < lr_init * 0.01f) lr = lr_init * 0.01f;

        memset(rnn.h, 0, sizeof(float) * HIDDEN_SIZE);
        double tbpc = train_pass(&rnn, data, len, lr);

        if (tbpc < best_bpc - 0.0001) {
            best_bpc = tbpc;
            memcpy(&best_rnn, &rnn, sizeof(RNN));
        }

        int report = (epoch <= 10) ||
                     (epoch <= 100 && epoch % 10 == 0) ||
                     (epoch <= 1000 && epoch % 100 == 0) ||
                     (epoch % 1000 == 0) ||
                     (epoch == max_epochs);

        if (report) {
            memset(rnn.h, 0, sizeof(float) * HIDDEN_SIZE);
            double ebpc = eval_bpc(&rnn, data, len);
            printf("%8d %10.4f %10.4f %8.6f%s\n",
                   epoch, tbpc, ebpc, lr, (tbpc <= best_bpc + 0.0001) ? " *" : "");
            fflush(stdout);
        }

        if (epoch % ckpt_every == 0) {
            snprintf(path, sizeof(path), "%s_epoch%d.bin", prefix, epoch);
            save_model(&rnn, path);
        }
    }

    /* Save best */
    snprintf(path, sizeof(path), "%s_best.bin", prefix);
    save_model(&best_rnn, path);
    memset(best_rnn.h, 0, sizeof(float) * HIDDEN_SIZE);
    double fbpc = eval_bpc(&best_rnn, data, len);
    printf("\nBest model: %.4f bpc -> %s\n", fbpc, path);

    /* Save final */
    snprintf(path, sizeof(path), "%s_final.bin", prefix);
    save_model(&rnn, path);

    free(data);
    return 0;
}
