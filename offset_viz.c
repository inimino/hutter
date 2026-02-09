/*
 * offset_viz.c - Visualize what the greedy skip offsets see
 *
 * For each output position, shows what bytes the greedy offsets [1,8,20,3,27]
 * would observe, revealing the structural interpretation.
 *
 * Usage: ./offset_viz <datafile>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DATA 2048

static unsigned char data[MAX_DATA];
static int data_len;

static char safe(int c) {
    if (c == '\n') return '.';
    if (c == '\t') return '_';
    if (c >= 32 && c < 127) return c;
    return '?';
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <datafile>\n", argv[0]); return 1; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    data_len = fread(data, 1, MAX_DATA, f);
    fclose(f);

    int offsets[] = {1, 8, 20, 3, 27};
    int noff = 5;
    int max_off = 27;

    printf("Greedy offsets: [1, 8, 20, 3, 27]\n");
    printf("Showing every 20th position from offset %d to %d\n\n", max_off, data_len-1);

    printf("pos  output  @-1   @-3   @-8   @-20  @-27  context_window\n");
    printf("---  ------  ----  ----  ----  ----  ----  ------------------------------------------\n");

    for (int t = max_off; t < data_len; t += 20) {
        printf("%3d  '%c'     '%c'   '%c'   '%c'   '%c'   '%c'   ",
               t, safe(data[t]),
               safe(data[t-1]), safe(data[t-3]),
               safe(data[t-8]), safe(data[t-20]),
               safe(data[t-27]));

        /* Show context window: positions t-30 to t */
        int start = t - 30;
        if (start < 0) start = 0;
        for (int i = start; i <= t; i++) {
            if (i == t) printf("[");
            printf("%c", safe(data[i]));
            if (i == t) printf("]");
        }
        printf("\n");
    }

    /* Show line lengths */
    printf("\n=== Line lengths ===\n");
    int line_start = 0;
    int line_num = 0;
    for (int i = 0; i < data_len; i++) {
        if (data[i] == '\n' || i == data_len - 1) {
            int len = i - line_start;
            printf("line %2d: len=%3d  \"", line_num, len);
            int show = len < 60 ? len : 60;
            for (int j = line_start; j < line_start + show; j++)
                printf("%c", safe(data[j]));
            if (len > 60) printf("...");
            printf("\"\n");
            line_start = i + 1;
            line_num++;
        }
    }

    /* Autocorrelation: for each offset, how often does data[t] == data[t-offset]? */
    printf("\n=== Byte repetition at each offset ===\n");
    printf("offset  matches  rate\n");
    for (int d = 1; d <= 50; d++) {
        int matches = 0;
        int total = 0;
        for (int t = d; t < data_len; t++) {
            if (data[t] == data[t-d]) matches++;
            total++;
        }
        printf("  %2d     %4d    %.3f%s\n", d, matches, (double)matches/total,
               (d==1||d==3||d==8||d==20||d==27) ? "  <-- greedy" : "");
    }

    return 0;
}
