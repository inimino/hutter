/*
 * es_iso.c — Analyze the isomorphism between arch-native (SVD) and
 * human-native byte partitions in the event space.
 *
 * For each offset, compute:
 *   - SVD 8-group partition (sign bits of top 3 SVs)
 *   - Human 8-group partition (semantic byte classes)
 *   - Confusion matrix, optimal permutation (brute-force 8!)
 *   - Accuracy, NMI, centroid alignment in SVD subspace
 *   - Inner product preservation between the two labelings
 *
 * Usage: es_iso <data_file> [max_bytes]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_OFF 16
#define K 8
#define NG 8  /* number of groups on each side */

static int cnt[256][256];
static int cnt_row[256];
static double U[256][K], S[K], V[256][K];

void svd_top_k(double A[256][256], int nk) {
    double u[256], v[256];
    for (int k = 0; k < nk; k++) {
        srand(42 + k);
        double norm = 0;
        for (int j = 0; j < 256; j++) { v[j] = ((double)rand()/RAND_MAX)-0.5; norm += v[j]*v[j]; }
        norm = sqrt(norm);
        for (int j = 0; j < 256; j++) v[j] /= norm;
        double sigma = 0;
        for (int iter = 0; iter < 300; iter++) {
            for (int i = 0; i < 256; i++) {
                u[i] = 0;
                for (int j = 0; j < 256; j++) u[i] += A[i][j]*v[j];
            }
            for (int p = 0; p < k; p++) {
                double dot = 0;
                for (int i = 0; i < 256; i++) dot += u[i]*U[i][p];
                for (int i = 0; i < 256; i++) u[i] -= dot*U[i][p];
            }
            sigma = 0;
            for (int i = 0; i < 256; i++) sigma += u[i]*u[i];
            sigma = sqrt(sigma);
            if (sigma < 1e-15) break;
            for (int i = 0; i < 256; i++) u[i] /= sigma;
            for (int j = 0; j < 256; j++) {
                v[j] = 0;
                for (int i = 0; i < 256; i++) v[j] += A[i][j]*u[i];
            }
            for (int p = 0; p < k; p++) {
                double dot = 0;
                for (int j = 0; j < 256; j++) dot += v[j]*V[j][p];
                for (int j = 0; j < 256; j++) v[j] -= dot*V[j][p];
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

/* Human-native 8-group partition.
 * Group 0: lowercase letters (a-z)
 * Group 1: uppercase letters (A-Z)
 * Group 2: digits (0-9)
 * Group 3: whitespace (space, newline, tab)
 * Group 4: XML markup (<, >, /, &)
 * Group 5: punctuation (. , ; : ! ? ' " - ( ) [ ])
 * Group 6: other printable (= + * | # @ ^ ~ ` _ { } \ %)
 * Group 7: non-printable (control 0-31 ex ws, high 128-255)
 */
int human_group(int b) {
    if (b >= 'a' && b <= 'z') return 0;
    if (b >= 'A' && b <= 'Z') return 1;
    if (b >= '0' && b <= '9') return 2;
    if (b == ' ' || b == '\n' || b == '\t') return 3;
    if (b == '<' || b == '>' || b == '/' || b == '&') return 4;
    if (b == '.' || b == ',' || b == ';' || b == ':' ||
        b == '!' || b == '?' || b == '\'' || b == '"' ||
        b == '-' || b == '(' || b == ')' || b == '[' || b == ']') return 5;
    if (b >= 32 && b < 127) return 6;
    return 7;
}

const char *human_group_name(int g) {
    const char *names[] = {"lowercase","uppercase","digits","whitespace",
                           "XML","punctuation","other-print","non-print"};
    return names[g];
}

/* Brute-force optimal permutation of 8 elements.
 * Maximizes sum of confusion[svd_g][human_perm[svd_g]] weighted by byte freq.
 */
static int best_perm[NG];
static double best_score;

void permute(int perm[], int used[], int depth, double conf[][NG]) {
    if (depth == NG) {
        double score = 0;
        for (int i = 0; i < NG; i++) score += conf[i][perm[i]];
        if (score > best_score) {
            best_score = score;
            memcpy(best_perm, perm, sizeof(int)*NG);
        }
        return;
    }
    for (int j = 0; j < NG; j++) {
        if (!used[j]) {
            used[j] = 1;
            perm[depth] = j;
            permute(perm, used, depth+1, conf);
            used[j] = 0;
        }
    }
}

/* NMI between two partitions (weighted by byte count) */
double nmi(int *part_a, int *part_b, int *bc, int n_bytes, int n_total) {
    double joint[NG][NG];
    double marg_a[NG], marg_b[NG];
    memset(joint, 0, sizeof(joint));
    memset(marg_a, 0, sizeof(marg_a));
    memset(marg_b, 0, sizeof(marg_b));

    for (int b = 0; b < n_bytes; b++) {
        double w = (double)bc[b] / n_total;
        joint[part_a[b]][part_b[b]] += w;
        marg_a[part_a[b]] += w;
        marg_b[part_b[b]] += w;
    }

    double mi = 0, ha = 0, hb = 0;
    for (int i = 0; i < NG; i++) {
        if (marg_a[i] > 0) ha -= marg_a[i] * log2(marg_a[i]);
        if (marg_b[i] > 0) hb -= marg_b[i] * log2(marg_b[i]);
        for (int j = 0; j < NG; j++) {
            if (joint[i][j] > 0)
                mi += joint[i][j] * log2(joint[i][j] / (marg_a[i] * marg_b[j]));
        }
    }
    double denom = (ha + hb) / 2.0;
    return denom > 0 ? mi / denom : 0;
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

    int bc[256]; memset(bc, 0, sizeof(bc));
    for (int t = 0; t < n; t++) bc[data[t]]++;

    double marg[256];
    for (int o = 0; o < 256; o++) marg[o] = (bc[o]+0.5)/(n+128.0);

    /* Human partition */
    int hgroup[256];
    for (int b = 0; b < 256; b++) hgroup[b] = human_group(b);

    /* Count active bytes */
    int n_active = 0;
    for (int b = 0; b < 256; b++) if (bc[b] > 0) n_active++;

    printf("=== Event Space Isomorphism Analysis ===\n");
    printf("Data: %d bytes, %d active byte values\n\n", n, n_active);

    /* Human group summary */
    printf("Human-native 8-group partition:\n");
    for (int g = 0; g < NG; g++) {
        int cnt_g = 0, freq_g = 0;
        for (int b = 0; b < 256; b++) {
            if (hgroup[b] == g) { cnt_g++; freq_g += bc[b]; }
        }
        printf("  H%d %-12s: %3d bytes, %6.2f%% of data\n",
               g, human_group_name(g), cnt_g, 100.0*freq_g/n);
    }
    printf("\n");

    /* Per-offset analysis */
    for (int g_off = 0; g_off < MAX_OFF; g_off++) {
        /* Build skip-bigram and centered matrix */
        memset(cnt, 0, sizeof(cnt));
        memset(cnt_row, 0, sizeof(cnt_row));
        for (int t = g_off + 1; t < n; t++) {
            int x = data[t - g_off - 1], y = data[t];
            cnt[x][y]++;
            cnt_row[x]++;
        }

        static double A[256][256];
        for (int x = 0; x < 256; x++) {
            int tot = cnt_row[x];
            for (int o = 0; o < 256; o++) {
                double pc = (tot > 0) ? (cnt[x][o]+0.1)/(tot+25.6) : marg[o];
                A[x][o] = pc - marg[o];
            }
        }

        memset(U, 0, sizeof(U));
        memset(S, 0, sizeof(S));
        memset(V, 0, sizeof(V));
        svd_top_k(A, K);

        /* SVD event assignment (sign bits of top 3) */
        int svd_group[256];
        for (int b = 0; b < 256; b++) {
            int ev = 0;
            for (int k = 0; k < 3; k++)
                if (U[b][k] > 0) ev |= (1 << k);
            svd_group[b] = ev;
        }

        /* Confusion matrix (freq-weighted): conf[svd][human] */
        double conf[NG][NG];
        memset(conf, 0, sizeof(conf));
        for (int b = 0; b < 256; b++) {
            conf[svd_group[b]][hgroup[b]] += (double)bc[b] / n;
        }

        /* Find optimal permutation (maximize diagonal sum) */
        best_score = -1;
        int perm[NG], used[NG];
        memset(used, 0, sizeof(used));
        permute(perm, used, 0, conf);

        /* Accuracy under best permutation */
        double acc = 0;
        for (int i = 0; i < NG; i++) acc += conf[i][best_perm[i]];

        /* NMI */
        double nmi_val = nmi(svd_group, hgroup, bc, 256, n);

        /* Centroid analysis in 3D SVD space */
        /* SVD centroids (weighted by frequency) */
        double c_svd[NG][3], w_svd[NG];
        memset(c_svd, 0, sizeof(c_svd));
        memset(w_svd, 0, sizeof(w_svd));
        for (int b = 0; b < 256; b++) {
            double w = (double)bc[b] / n;
            int g = svd_group[b];
            w_svd[g] += w;
            for (int k = 0; k < 3; k++) c_svd[g][k] += w * U[b][k];
        }
        for (int g = 0; g < NG; g++)
            if (w_svd[g] > 0) for (int k = 0; k < 3; k++) c_svd[g][k] /= w_svd[g];

        /* Human centroids (weighted by frequency) */
        double c_hum[NG][3], w_hum[NG];
        memset(c_hum, 0, sizeof(c_hum));
        memset(w_hum, 0, sizeof(w_hum));
        for (int b = 0; b < 256; b++) {
            double w = (double)bc[b] / n;
            int g = hgroup[b];
            w_hum[g] += w;
            for (int k = 0; k < 3; k++) c_hum[g][k] += w * U[b][k];
        }
        for (int g = 0; g < NG; g++)
            if (w_hum[g] > 0) for (int k = 0; k < 3; k++) c_hum[g][k] /= w_hum[g];

        /* Cosine similarity between paired centroids under optimal perm */
        double mean_cos = 0;
        int n_pairs = 0;
        for (int i = 0; i < NG; i++) {
            int j = best_perm[i];
            if (w_svd[i] < 1e-9 || w_hum[j] < 1e-9) continue;
            double dot = 0, na = 0, nb = 0;
            for (int k = 0; k < 3; k++) {
                dot += c_svd[i][k] * c_hum[j][k];
                na += c_svd[i][k] * c_svd[i][k];
                nb += c_hum[j][k] * c_hum[j][k];
            }
            na = sqrt(na); nb = sqrt(nb);
            if (na > 1e-12 && nb > 1e-12) {
                mean_cos += dot / (na * nb);
                n_pairs++;
            }
        }
        if (n_pairs > 0) mean_cos /= n_pairs;

        /* Inner product preservation: for pairs of active bytes,
         * do same-group assignments agree? */
        int agree_same = 0, total_same_svd = 0, agree_diff = 0, total_diff_svd = 0;
        for (int b1 = 0; b1 < 256; b1++) {
            if (bc[b1] == 0) continue;
            for (int b2 = b1+1; b2 < 256; b2++) {
                if (bc[b2] == 0) continue;
                int same_svd = (svd_group[b1] == svd_group[b2]);
                int same_hum = (hgroup[b1] == hgroup[b2]);
                /* Under permutation, same SVD group ↔ same human group */
                int mapped_hum = (best_perm[svd_group[b1]] == hgroup[b1] &&
                                  best_perm[svd_group[b2]] == hgroup[b2]);
                if (same_svd) { total_same_svd++; if (same_hum) agree_same++; }
                else { total_diff_svd++; if (!same_hum) agree_diff++; }
            }
        }

        /* Output side: V-space centroids */
        double c_svd_v[NG][3], w_svd_v[NG];
        memset(c_svd_v, 0, sizeof(c_svd_v));
        memset(w_svd_v, 0, sizeof(w_svd_v));
        double c_hum_v[NG][3], w_hum_v[NG];
        memset(c_hum_v, 0, sizeof(c_hum_v));
        memset(w_hum_v, 0, sizeof(w_hum_v));
        for (int b = 0; b < 256; b++) {
            double w = (double)bc[b] / n;
            int sg = svd_group[b], hg = hgroup[b];
            /* For output side, use V vectors with same groupings */
            /* (output events based on sign of V, not U) */
            int ov = 0;
            for (int k = 0; k < 3; k++)
                if (V[b][k] > 0) ov |= (1 << k);
            w_svd_v[ov] += w;
            for (int k = 0; k < 3; k++) c_svd_v[ov][k] += w * V[b][k];
            w_hum_v[hg] += w;
            for (int k = 0; k < 3; k++) c_hum_v[hg][k] += w * V[b][k];
        }
        for (int g = 0; g < NG; g++) {
            if (w_svd_v[g] > 0) for (int k = 0; k < 3; k++) c_svd_v[g][k] /= w_svd_v[g];
            if (w_hum_v[g] > 0) for (int k = 0; k < 3; k++) c_hum_v[g][k] /= w_hum_v[g];
        }

        /* V-side optimal permutation */
        int v_svd_group[256];
        for (int b = 0; b < 256; b++) {
            int ev = 0;
            for (int k = 0; k < 3; k++)
                if (V[b][k] > 0) ev |= (1 << k);
            v_svd_group[b] = ev;
        }
        double vconf[NG][NG];
        memset(vconf, 0, sizeof(vconf));
        for (int b = 0; b < 256; b++)
            vconf[v_svd_group[b]][hgroup[b]] += (double)bc[b] / n;
        best_score = -1;
        memset(used, 0, sizeof(used));
        permute(perm, used, 0, vconf);
        double v_acc = 0;
        for (int i = 0; i < NG; i++) v_acc += vconf[i][best_perm[i]];
        int v_best_perm[NG];
        memcpy(v_best_perm, best_perm, sizeof(best_perm));
        double v_nmi = nmi(v_svd_group, hgroup, bc, 256, n);

        /* V-side centroid cosine */
        double v_mean_cos = 0;
        int v_pairs = 0;
        for (int i = 0; i < NG; i++) {
            int j = v_best_perm[i];
            if (w_svd_v[i] < 1e-9 || w_hum_v[j] < 1e-9) continue;
            double dot = 0, na = 0, nb = 0;
            for (int k = 0; k < 3; k++) {
                dot += c_svd_v[i][k] * c_hum_v[j][k];
                na += c_svd_v[i][k] * c_svd_v[i][k];
                nb += c_hum_v[j][k] * c_hum_v[j][k];
            }
            na = sqrt(na); nb = sqrt(nb);
            if (na > 1e-12 && nb > 1e-12) {
                v_mean_cos += dot / (na * nb);
                v_pairs++;
            }
        }
        if (v_pairs > 0) v_mean_cos /= v_pairs;

        /* Print results */
        printf("--- Offset %d ---\n", g_off);
        printf("  SVs: σ=[%.3f, %.3f, %.3f]  var=[%.1f%%, %.1f%%, %.1f%%]\n",
               S[0], S[1], S[2],
               100*S[0]*S[0]/(S[0]*S[0]+S[1]*S[1]+S[2]*S[2]+1e-15),
               100*S[1]*S[1]/(S[0]*S[0]+S[1]*S[1]+S[2]*S[2]+1e-15),
               100*S[2]*S[2]/(S[0]*S[0]+S[1]*S[1]+S[2]*S[2]+1e-15));

        /* Reload best_perm for U-side for printing */
        best_score = -1;
        memset(used, 0, sizeof(used));
        permute(perm, used, 0, conf);

        printf("  INPUT SIDE (U):\n");
        printf("    Accuracy: %.1f%%   NMI: %.3f   Centroid cos: %.3f\n",
               100*acc, nmi_val, mean_cos);
        printf("    Permutation: ");
        for (int i = 0; i < NG; i++)
            printf("SVD%d→%s%s", i, human_group_name(best_perm[i]), i<7?", ":"");
        printf("\n");

        printf("    Pair agreement: same-group %.1f%% (%d/%d), diff-group %.1f%% (%d/%d)\n",
               total_same_svd > 0 ? 100.0*agree_same/total_same_svd : 0,
               agree_same, total_same_svd,
               total_diff_svd > 0 ? 100.0*agree_diff/total_diff_svd : 0,
               agree_diff, total_diff_svd);

        /* Show confusion matrix for interesting groups */
        printf("    Confusion (freq%%):\n");
        printf("    %12s", "");
        for (int j = 0; j < NG; j++) printf(" %8s", human_group_name(j));
        printf("\n");
        for (int i = 0; i < NG; i++) {
            double row_sum = 0;
            for (int j = 0; j < NG; j++) row_sum += conf[i][j];
            if (row_sum < 0.001) continue;
            printf("    SVD%-8d", i);
            for (int j = 0; j < NG; j++)
                printf(" %7.2f%%", 100*conf[i][j]);
            printf("  (=%.1f%%)\n", 100*row_sum);
        }

        printf("  OUTPUT SIDE (V):\n");
        printf("    Accuracy: %.1f%%   NMI: %.3f   Centroid cos: %.3f\n",
               100*v_acc, v_nmi, v_mean_cos);
        printf("    Permutation: ");
        for (int i = 0; i < NG; i++)
            printf("SVD%d→%s%s", i, human_group_name(v_best_perm[i]), i<7?", ":"");
        printf("\n");
        printf("\n");
    }

    free(data);
    return 0;
}
