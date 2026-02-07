/*
 * sat_sn_full: Full SN export including W_hh recurrent patterns.
 *
 * Loads sat_model.bin, exports complete pattern inventory:
 *   events.sn: 768 events (256 input + 256 hidden doubled-E + 256 output)
 *   patterns.sn: ALL patterns with strength >= 1, including Wx, Wh, Wy
 *
 * Sign convention for W_hh: Wh[j][k] = "from neuron k at time t to
 * neuron j at time t+1". Under doubled-E, source is always h_k+
 * (positive activations propagate). Positive weight -> h_k+ drives h_j+.
 * Negative weight -> h_k+ drives h_j-.
 *
 * Usage: sat_sn_full <model.bin> [output_dir]
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

void format_byte_event(char* buf, int byte, const char* prefix) {
    if (byte >= 32 && byte < 127 && byte != '"' && byte != '\\')
        sprintf(buf, "\"The %s is '%c'.\"", prefix, byte);
    else
        sprintf(buf, "\"The %s is 0x%02X.\"", prefix, byte);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.bin> [output_dir]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* outdir = (argc >= 3) ? argv[2] : ".";

    RNN rnn;
    load_model(&rnn, model_path);

    char path[512];

    /* === Export events.sn === */
    snprintf(path, sizeof(path), "%s/events.sn", outdir);
    FILE* ef = fopen(path, "w");
    if (!ef) { perror("events.sn"); return 1; }

    for (int i = 0; i < 256; i++) {
        char name[64];
        format_byte_event(name, i, "input");
        fprintf(ef, "%s 0.\n", name);
    }
    for (int j = 1; j <= HIDDEN_SIZE; j++) {
        fprintf(ef, "\"h%d+\" 0.\n", j);
        fprintf(ef, "\"h%d-\" 0.\n", j);
    }
    for (int o = 0; o < 256; o++) {
        char name[64];
        format_byte_event(name, o, "output");
        fprintf(ef, "%s 0.\n", name);
    }
    fclose(ef);
    printf("Wrote %s (768 events)\n", path);

    /* === Export patterns.sn === */
    snprintf(path, sizeof(path), "%s/patterns.sn", outdir);
    FILE* pf = fopen(path, "w");
    if (!pf) { perror("patterns.sn"); return 1; }

    int n_wx = 0, n_wh = 0, n_wy = 0;
    int hist_wx[16] = {0}, hist_wh[16] = {0}, hist_wy[16] = {0};
    int max_s_wx = 0, max_s_wh = 0, max_s_wy = 0;

    /* --- Wx: input -> hidden --- */
    fprintf(pf, "# Wx: input -> hidden\n");
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            float w = rnn.Wx[j][i];
            int strength = (int)(2.0f * fabsf(w) + 0.5f);
            if (strength < 1) continue;

            char src[64], dst[64];
            format_byte_event(src, i, "input");
            sprintf(dst, "\"h%d%c\"", j+1, (w > 0) ? '+' : '-');
            fprintf(pf, "%s %s %d.\n", src, dst, strength);
            n_wx++;
            if (strength < 16) hist_wx[strength]++;
            if (strength > max_s_wx) max_s_wx = strength;
        }
    }

    /* --- Wh: hidden -> hidden (recurrent) --- */
    fprintf(pf, "# Wh: hidden -> hidden (recurrent)\n");
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            float w = rnn.Wh[j][k];
            int strength = (int)(2.0f * fabsf(w) + 0.5f);
            if (strength < 1) continue;

            /* Source: h_k+ (positive activation at time t)
             * Dest: h_j+ if w > 0, h_j- if w < 0 (at time t+1) */
            char src[32], dst[32];
            sprintf(src, "\"h%d+\"", k+1);
            sprintf(dst, "\"h%d%c\"", j+1, (w > 0) ? '+' : '-');
            fprintf(pf, "%s %s %d.\n", src, dst, strength);
            n_wh++;
            if (strength < 16) hist_wh[strength]++;
            if (strength > max_s_wh) max_s_wh = strength;
        }
    }

    /* --- Wy: hidden -> output --- */
    fprintf(pf, "# Wy: hidden -> output\n");
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float w = rnn.Wy[o][j];
            int strength = (int)(2.0f * fabsf(w) + 0.5f);
            if (strength < 1) continue;

            char src[32], dst[64];
            sprintf(src, "\"h%d%c\"", j+1, (w > 0) ? '+' : '-');
            format_byte_event(dst, o, "output");
            fprintf(pf, "%s %s %d.\n", src, dst, strength);
            n_wy++;
            if (strength < 16) hist_wy[strength]++;
            if (strength > max_s_wy) max_s_wy = strength;
        }
    }

    fclose(pf);

    int total = n_wx + n_wh + n_wy;
    printf("Wrote %s (%d patterns)\n", path, total);

    /* === Summary === */
    printf("\n=== Pattern Summary ===\n\n");
    printf("Layer           Weights   Patterns  Max-str\n");
    printf("Wx (in->hid)    %6d    %6d    %d\n", INPUT_SIZE * HIDDEN_SIZE, n_wx, max_s_wx);
    printf("Wh (hid->hid)   %6d    %6d    %d\n", HIDDEN_SIZE * HIDDEN_SIZE, n_wh, max_s_wh);
    printf("Wy (hid->out)   %6d    %6d    %d\n", OUTPUT_SIZE * HIDDEN_SIZE, n_wy, max_s_wy);
    printf("Total           %6d    %6d\n", INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*HIDDEN_SIZE + OUTPUT_SIZE*HIDDEN_SIZE, total);

    printf("\n=== Strength Histogram ===\n\n");
    printf("Str   Wx     Wh     Wy\n");
    for (int s = 1; s <= 15; s++) {
        if (hist_wx[s] == 0 && hist_wh[s] == 0 && hist_wy[s] == 0) continue;
        printf("%3d   %5d  %5d  %5d\n", s, hist_wx[s], hist_wh[s], hist_wy[s]);
    }
    if (max_s_wx > 15 || max_s_wh > 15 || max_s_wy > 15)
        printf("(some patterns have strength > 15, shown in max above)\n");

    /* === Hub neuron analysis === */
    printf("\n=== Hub Neurons (W_hh) ===\n\n");

    int fan_in[HIDDEN_SIZE] = {0};
    int fan_out[HIDDEN_SIZE] = {0};

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            int strength = (int)(2.0f * fabsf(rnn.Wh[j][k]) + 0.5f);
            if (strength < 1) continue;
            fan_out[k]++;  /* k sends to j */
            fan_in[j]++;   /* j receives from k */
        }
    }

    /* Sort by total connections */
    typedef struct { int idx; int fi; int fo; int total; } Hub;
    Hub hubs[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hubs[i].idx = i;
        hubs[i].fi = fan_in[i];
        hubs[i].fo = fan_out[i];
        hubs[i].total = fan_in[i] + fan_out[i];
    }
    for (int a = 0; a < HIDDEN_SIZE - 1; a++)
        for (int b = a + 1; b < HIDDEN_SIZE; b++)
            if (hubs[b].total > hubs[a].total) {
                Hub tmp = hubs[a]; hubs[a] = hubs[b]; hubs[b] = tmp;
            }

    printf("Neuron  Fan-in  Fan-out  Total\n");
    for (int i = 0; i < 20 && hubs[i].total > 0; i++) {
        printf("h%-4d   %5d   %6d  %5d\n",
               hubs[i].idx + 1, hubs[i].fi, hubs[i].fo, hubs[i].total);
    }

    /* === Self-connections === */
    printf("\n=== Self-Connections (diagonal of W_hh) ===\n\n");
    printf("Neuron   Weight    Strength  Sign\n");
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float w = rnn.Wh[j][j];
        int strength = (int)(2.0f * fabsf(w) + 0.5f);
        if (fabsf(w) >= 0.25f) {
            printf("h%-4d    %+.4f   %d         %s\n",
                   j + 1, w, strength, (w > 0) ? "persist" : "flip");
        }
    }

    return 0;
}
