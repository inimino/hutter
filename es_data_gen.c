/*
 * es_data_gen.c — Generate JSON data for the Event Space viewer.
 *
 * Computes SVD of skip-bigram tables at each offset, outputs:
 *   - Singular values per offset
 *   - Left singular vectors u_k(x) per offset (input byte loadings)
 *   - Right singular vectors v_k(o) per offset (output byte loadings)
 *   - Event assignments (from sign bits of top 3 SVs)
 *   - Byte frequencies
 *   - Mutual information per offset
 *
 * Output is a JavaScript file that can be embedded in the viewer.
 *
 * Usage: es_data_gen <data_file> [max_bytes] > es_data.js
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_OFF 16
#define K 8  /* top 8 singular vectors */

static int cnt[256][256];
static int cnt_row[256];

/* SVD storage */
static double U[256][K];
static double S[K];
static double V[256][K];

void svd_top_k(double A[256][256], int nk) {
    double u[256], v[256];
    for (int k = 0; k < nk; k++) {
        srand(42 + k);
        double norm = 0;
        for (int j = 0; j < 256; j++) { v[j] = ((double)rand() / RAND_MAX) - 0.5; norm += v[j]*v[j]; }
        norm = sqrt(norm);
        for (int j = 0; j < 256; j++) v[j] /= norm;

        double sigma = 0;
        for (int iter = 0; iter < 300; iter++) {
            for (int i = 0; i < 256; i++) {
                u[i] = 0;
                for (int j = 0; j < 256; j++) u[i] += A[i][j] * v[j];
            }
            for (int p = 0; p < k; p++) {
                double dot = 0;
                for (int i = 0; i < 256; i++) dot += u[i] * U[i][p];
                for (int i = 0; i < 256; i++) u[i] -= dot * U[i][p];
            }
            sigma = 0;
            for (int i = 0; i < 256; i++) sigma += u[i]*u[i];
            sigma = sqrt(sigma);
            if (sigma < 1e-15) break;
            for (int i = 0; i < 256; i++) u[i] /= sigma;

            for (int j = 0; j < 256; j++) {
                v[j] = 0;
                for (int i = 0; i < 256; i++) v[j] += A[i][j] * u[i];
            }
            for (int p = 0; p < k; p++) {
                double dot = 0;
                for (int j = 0; j < 256; j++) dot += v[j] * V[j][p];
                for (int j = 0; j < 256; j++) v[j] -= dot * V[j][p];
            }
            norm = 0;
            for (int j = 0; j < 256; j++) norm += v[j]*v[j];
            norm = sqrt(norm);
            if (norm < 1e-15) break;
            for (int j = 0; j < 256; j++) v[j] /= norm;
        }
        S[k] = sigma;
        for (int i = 0; i < 256; i++) U[i][k] = u[i];
        for (int j = 0; j < 256; j++) V[j][k] = v[j];
    }
}

/* Byte class for display */
const char *byte_class(int b) {
    if (b >= 'a' && b <= 'z') return "lower";
    if (b >= 'A' && b <= 'Z') return "upper";
    if (b >= '0' && b <= '9') return "digit";
    if (b == ' ') return "space";
    if (b == '\n') return "newline";
    if (b == '\t') return "tab";
    if (b == '<' || b == '>' || b == '/' || b == '&') return "xml";
    if (b == '.' || b == ',' || b == ';' || b == ':' || b == '!' || b == '?') return "punct";
    if (b == '(' || b == ')' || b == '[' || b == ']' || b == '{' || b == '}') return "bracket";
    if (b == '"' || b == '\'') return "quote";
    if (b == '=' || b == '-' || b == '+' || b == '*' || b == '|' || b == '#') return "operator";
    if (b >= 32 && b < 127) return "printable";
    if (b < 32) return "control";
    return "high";
}

/* Printable label for a byte */
void print_byte_label(int b) {
    if (b >= 32 && b < 127) printf("%c", b);
    else if (b == 10) printf("\\n");
    else if (b == 13) printf("\\r");
    else if (b == 9) printf("\\t");
    else if (b == 0) printf("\\0");
    else printf("x%02x", b);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <data> [max]\n", argv[0]); return 1; }
    int max = 1000000;
    if (argc >= 3) max = atoi(argv[2]);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", argv[1]); return 1; }
    unsigned char *data = malloc(max + 1);
    int n = fread(data, 1, max, f);
    fclose(f);

    /* Byte counts */
    int bc[256]; memset(bc, 0, sizeof(bc));
    for (int t = 0; t < n; t++) bc[data[t]]++;

    /* Marginals */
    double marg[256];
    for (int o = 0; o < 256; o++)
        marg[o] = (bc[o] + 0.5) / (n + 128.0);

    /* Output JavaScript data */
    printf("const ES_DATA = {\n");
    printf("  n: %d,\n", n);
    printf("  nOffsets: %d,\n", MAX_OFF);
    printf("  nSV: %d,\n", K);

    /* Byte info */
    printf("  bytes: [\n");
    for (int b = 0; b < 256; b++) {
        printf("    {b:%d, count:%d, freq:%.6f, cls:\"%s\", label:\"",
               b, bc[b], (double)bc[b]/n, byte_class(b));
        /* JSON-safe label */
        if (b == '"') printf("\\\"");
        else if (b == '\\') printf("\\\\");
        else if (b >= 32 && b < 127) printf("%c", b);
        else if (b == 10) printf("\\n");
        else if (b == 13) printf("\\r");
        else if (b == 9) printf("\\t");
        else printf("x%02x", b);
        printf("\"}%s\n", b < 255 ? "," : "");
    }
    printf("  ],\n");

    /* Per-offset data */
    printf("  offsets: [\n");
    for (int g = 0; g < MAX_OFF; g++) {
        fprintf(stderr, "  Computing offset %d...\n", g);

        /* Build skip-bigram counts */
        memset(cnt, 0, sizeof(cnt));
        memset(cnt_row, 0, sizeof(cnt_row));
        for (int t = g + 1; t < n; t++) {
            int x = data[t - g - 1];
            int y = data[t];
            cnt[x][y]++;
            cnt_row[x]++;
        }

        /* Mutual information for this offset */
        double mi = 0;
        int total_pairs = 0;
        for (int x = 0; x < 256; x++) total_pairs += cnt_row[x];
        for (int x = 0; x < 256; x++) {
            if (cnt_row[x] == 0) continue;
            for (int o = 0; o < 256; o++) {
                if (cnt[x][o] == 0) continue;
                double pxy = (double)cnt[x][o] / total_pairs;
                double px = (double)cnt_row[x] / total_pairs;
                double py = marg[o];
                if (px > 0 && py > 0 && pxy > 0)
                    mi += pxy * log2(pxy / (px * py));
            }
        }

        /* Build centered matrix */
        static double A[256][256];
        for (int x = 0; x < 256; x++) {
            int tot = cnt_row[x];
            for (int o = 0; o < 256; o++) {
                double pc = (tot > 0) ? (cnt[x][o] + 0.1) / (tot + 25.6) : marg[o];
                A[x][o] = pc - marg[o];
            }
        }

        /* SVD */
        memset(U, 0, sizeof(U));
        memset(S, 0, sizeof(S));
        memset(V, 0, sizeof(V));
        svd_top_k(A, K);

        /* Total variance */
        double total_var = 0;
        for (int x = 0; x < 256; x++)
            for (int o = 0; o < 256; o++)
                total_var += A[x][o] * A[x][o];

        printf("    {\n");
        printf("      offset: %d,\n", g);
        printf("      mi: %.6f,\n", mi);
        printf("      totalVar: %.6f,\n", total_var);

        /* Singular values and cumulative variance */
        printf("      sv: [");
        double cum = 0;
        for (int k = 0; k < K; k++) {
            cum += S[k] * S[k];
            printf("{s:%.6f,cumVar:%.4f}%s", S[k], cum/total_var, k<K-1?",":"");
        }
        printf("],\n");

        /* Left singular vectors (input loadings) — only for bytes with count > 0 */
        printf("      u: [");
        for (int k = 0; k < K; k++) {
            printf("[");
            for (int b = 0; b < 256; b++) {
                printf("%.4f%s", U[b][k], b<255?",":"");
            }
            printf("]%s", k<K-1?",":"");
        }
        printf("],\n");

        /* Right singular vectors (output loadings) */
        printf("      v: [");
        for (int k = 0; k < K; k++) {
            printf("[");
            for (int b = 0; b < 256; b++) {
                printf("%.4f%s", V[b][k], b<255?",":"");
            }
            printf("]%s", k<K-1?",":"");
        }
        printf("],\n");

        /* Event assignments (sign bits of top 3 SVs) */
        printf("      events: [");
        for (int b = 0; b < 256; b++) {
            int ev = 0;
            for (int k = 0; k < 3; k++)
                if (U[b][k] > 0) ev |= (1 << k);
            printf("%d%s", ev, b<255?",":"");
        }
        printf("]\n");

        printf("    }%s\n", g < MAX_OFF-1 ? "," : "");
    }
    printf("  ]\n");
    printf("};\n");

    free(data);
    return 0;
}
