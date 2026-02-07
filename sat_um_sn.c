/*
 * sat_um_sn: Export n-gram UM patterns in SN format.
 *
 * Builds n-gram trie from data, exports as SN patterns:
 *   - Bigrams: input event -> output event with strength = floor(log2(count)+0.5)
 *   - Trigrams+: uses context events "ctx:<chars>" to carry state
 *
 * Also exports events_um.sn with input/output events plus context events.
 *
 * Usage: sat_um_sn <datafile> [max_order] [output_dir]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_CTX 64

typedef struct TrieNode {
    struct TrieNode* children[256];
    int counts[256];
    int total;
} TrieNode;

TrieNode* trie_new(void) {
    return calloc(1, sizeof(TrieNode));
}

void trie_free(TrieNode* n) {
    if (!n) return;
    for (int i = 0; i < 256; i++)
        trie_free(n->children[i]);
    free(n);
}

void trie_insert(TrieNode* root, unsigned char* ctx, int ctx_len, unsigned char next) {
    TrieNode* node = root;
    for (int i = 0; i < ctx_len; i++) {
        if (!node->children[ctx[i]])
            node->children[ctx[i]] = trie_new();
        node = node->children[ctx[i]];
    }
    node->counts[next]++;
    node->total++;
}

TrieNode* trie_lookup(TrieNode* root, unsigned char* ctx, int ctx_len) {
    TrieNode* node = root;
    for (int i = 0; i < ctx_len; i++) {
        if (!node->children[ctx[i]]) return NULL;
        node = node->children[ctx[i]];
    }
    return node;
}

void format_byte_event(char* buf, int byte, const char* prefix) {
    if (byte >= 32 && byte < 127 && byte != '"' && byte != '\\')
        sprintf(buf, "\"The %s is '%c'.\"", prefix, byte);
    else
        sprintf(buf, "\"The %s is 0x%02X.\"", prefix, byte);
}

/* Format context string for SN event name */
void format_ctx_event(char* buf, unsigned char* ctx, int len) {
    strcpy(buf, "\"ctx:");
    int pos = 5; /* after "ctx: */
    for (int i = 0; i < len; i++) {
        unsigned char c = ctx[i];
        if (c >= 32 && c < 127 && c != '"' && c != '\\') {
            buf[pos++] = c;
        } else {
            pos += sprintf(buf + pos, "\\x%02X", c);
        }
    }
    buf[pos++] = '"';
    buf[pos] = '\0';
}

/* Count unique context events needed, and write them */
typedef struct {
    unsigned char ctx[MAX_CTX];
    int len;
} CtxKey;

#define MAX_CTX_EVENTS 65536
static CtxKey ctx_keys[MAX_CTX_EVENTS];
static int n_ctx_events = 0;

int find_or_add_ctx(unsigned char* ctx, int len) {
    for (int i = 0; i < n_ctx_events; i++) {
        if (ctx_keys[i].len == len && memcmp(ctx_keys[i].ctx, ctx, len) == 0)
            return i;
    }
    if (n_ctx_events >= MAX_CTX_EVENTS) return -1;
    memcpy(ctx_keys[n_ctx_events].ctx, ctx, len);
    ctx_keys[n_ctx_events].len = len;
    return n_ctx_events++;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <datafile> [max_order] [output_dir]\n", argv[0]);
        return 1;
    }

    int max_order = 11;
    if (argc >= 3) max_order = atoi(argv[2]);
    const char* outdir = (argc >= 4) ? argv[3] : ".";

    /* Load data */
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    int len = (int)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = malloc(len);
    fread(data, 1, len, f);
    fclose(f);

    printf("=== N-gram UM to SN ===\n");
    printf("Data: %d bytes, max order: %d\n\n", len, max_order);

    /* Build trie */
    TrieNode* root = trie_new();
    for (int t = 0; t < len - 1; t++) {
        int max_ctx = t + 1;
        if (max_ctx > max_order) max_ctx = max_order;
        for (int ctx_len = 0; ctx_len <= max_ctx; ctx_len++) {
            unsigned char* ctx = data + t + 1 - ctx_len;
            trie_insert(root, ctx, ctx_len, data[t + 1]);
        }
    }

    /* === Phase 1: Collect all context events needed === */
    int pattern_counts[max_order + 1];
    memset(pattern_counts, 0, sizeof(pattern_counts));

    /* For orders 3+, we need context events */
    for (int order = 3; order <= max_order; order++) {
        for (int t = order - 2; t < len - 1; t++) {
            int ctx_len = order - 1;
            if (ctx_len > t) continue;
            unsigned char* ctx = data + t + 1 - ctx_len;
            TrieNode* node = trie_lookup(root, ctx, ctx_len);
            if (!node) continue;

            for (int c = 0; c < 256; c++) {
                if (node->counts[c] < 2) continue;
                int strength = (int)(log2(node->counts[c]) + 0.5);
                if (strength < 1) continue;

                /* Register context events for all prefixes of ctx */
                for (int plen = 1; plen <= ctx_len; plen++) {
                    find_or_add_ctx(ctx, plen);
                }
            }
        }
    }

    printf("Context events needed: %d\n", n_ctx_events);

    /* === Phase 2: Export events_um.sn === */
    char path[512];
    snprintf(path, sizeof(path), "%s/events_um.sn", outdir);
    FILE* ef = fopen(path, "w");
    if (!ef) { perror("events_um.sn"); return 1; }

    /* Standard input/output events */
    for (int i = 0; i < 256; i++) {
        char name[64];
        format_byte_event(name, i, "input");
        fprintf(ef, "%s 0.\n", name);
    }
    for (int o = 0; o < 256; o++) {
        char name[64];
        format_byte_event(name, o, "output");
        fprintf(ef, "%s 0.\n", name);
    }
    /* Context events */
    for (int i = 0; i < n_ctx_events; i++) {
        char name[256];
        format_ctx_event(name, ctx_keys[i].ctx, ctx_keys[i].len);
        fprintf(ef, "%s 0.\n", name);
    }
    fclose(ef);
    printf("Wrote %s (%d events: 256 input + 256 output + %d context)\n",
           path, 512 + n_ctx_events, n_ctx_events);

    /* === Phase 3: Export patterns_um.sn === */
    snprintf(path, sizeof(path), "%s/patterns_um.sn", outdir);
    FILE* pf = fopen(path, "w");
    if (!pf) { perror("patterns_um.sn"); return 1; }

    int total_patterns = 0;

    /* --- Bigrams (order 2): input -> output directly --- */
    fprintf(pf, "# Bigrams: input -> output\n");
    int n_bigrams = 0;
    {
        TrieNode* unigram_root = root;
        for (int a = 0; a < 256; a++) {
            if (!unigram_root->children[a]) continue;
            TrieNode* node = unigram_root->children[a];
            for (int b = 0; b < 256; b++) {
                if (node->counts[b] < 2) continue;
                int strength = (int)(log2(node->counts[b]) + 0.5);
                if (strength < 1) continue;

                char src[64], dst[64];
                format_byte_event(src, a, "input");
                format_byte_event(dst, b, "output");
                fprintf(pf, "%s %s %d.\n", src, dst, strength);
                n_bigrams++;
                total_patterns++;
            }
        }
    }
    printf("Bigram patterns: %d\n", n_bigrams);

    /* --- Higher orders: use context events --- */
    for (int order = 3; order <= max_order; order++) {
        fprintf(pf, "# Order %d\n", order);
        int n_this_order = 0;

        for (int t = order - 2; t < len - 1; t++) {
            int ctx_len = order - 1;
            if (ctx_len > t) continue;
            unsigned char* ctx = data + t + 1 - ctx_len;
            TrieNode* node = trie_lookup(root, ctx, ctx_len);
            if (!node) continue;

            for (int c = 0; c < 256; c++) {
                if (node->counts[c] < 2) continue;
                int strength = (int)(log2(node->counts[c]) + 0.5);
                if (strength < 1) continue;

                /* We only need to emit patterns we haven't already emitted.
                 * Since the same n-gram can appear at multiple positions t,
                 * we should deduplicate. We'll do this by only emitting when
                 * t is the first occurrence of this context. */
                int is_first = 1;
                for (int tt = order - 2; tt < t; tt++) {
                    if (tt + 1 < ctx_len) continue;
                    unsigned char* prev_ctx = data + tt + 1 - ctx_len;
                    if (memcmp(prev_ctx, ctx, ctx_len) == 0) {
                        is_first = 0;
                        break;
                    }
                }
                if (!is_first) continue;

                /* Emit: ctx_event -> output with strength */
                char ctx_ev[256], dst[64];
                format_ctx_event(ctx_ev, ctx, ctx_len);
                format_byte_event(dst, c, "output");
                fprintf(pf, "%s %s %d.\n", ctx_ev, dst, strength);
                n_this_order++;
                total_patterns++;
            }
        }
        pattern_counts[order] = n_this_order;
        if (n_this_order > 0)
            printf("Order %d patterns: %d\n", order, n_this_order);
    }

    /* Also emit context-building patterns:
     * input(a) -> ctx:a (entering context from bigram)
     * ctx:ab + input(c) -> ctx:abc (extending context) */
    fprintf(pf, "# Context-building patterns\n");
    int n_ctx_patterns = 0;
    for (int i = 0; i < n_ctx_events; i++) {
        int clen = ctx_keys[i].len;
        if (clen == 1) {
            /* input(a) -> ctx:a */
            char src[64], dst[256];
            format_byte_event(src, ctx_keys[i].ctx[0], "input");
            format_ctx_event(dst, ctx_keys[i].ctx, 1);
            /* Strength: how many times this byte appears */
            TrieNode* node = trie_lookup(root, ctx_keys[i].ctx, 1);
            int count = node ? node->total : 0;
            int strength = (count >= 2) ? (int)(log2(count) + 0.5) : 1;
            fprintf(pf, "%s %s %d.\n", src, dst, strength);
            n_ctx_patterns++;
            total_patterns++;
        } else {
            /* ctx:prefix + input(last) -> ctx:full */
            char src[256], dst[256];
            format_ctx_event(src, ctx_keys[i].ctx, clen - 1);
            format_ctx_event(dst, ctx_keys[i].ctx, clen);
            /* Find or create the sub-context */
            TrieNode* node = trie_lookup(root, ctx_keys[i].ctx, clen);
            int count = node ? node->total : 0;
            int strength = (count >= 2) ? (int)(log2(count) + 0.5) : 1;
            fprintf(pf, "%s %s %d.\n", src, dst, strength);
            n_ctx_patterns++;
            total_patterns++;
        }
    }
    printf("Context-building patterns: %d\n", n_ctx_patterns);

    fclose(pf);
    printf("\nWrote %s (%d total patterns)\n", path, total_patterns);

    /* === Verify: compute bpc at order 11 to check consistency === */
    printf("\n=== BPC Verification ===\n");
    for (int order = 2; order <= max_order; order += (order < 5 ? 1 : (order < 11 ? 2 : 1))) {
        double total_loss = 0.0;
        int n_pred = 0;
        for (int t = 0; t < len - 1; t++) {
            unsigned char target = data[t + 1];
            double prob = 0;
            for (int ctx_len = order - 1; ctx_len >= 0; ctx_len--) {
                if (ctx_len > t) continue;
                unsigned char* ctx = data + t + 1 - ctx_len;
                TrieNode* node = trie_lookup(root, ctx, ctx_len);
                if (node && node->total > 0 && node->counts[target] > 0) {
                    prob = (double)node->counts[target] / node->total;
                    break;
                }
            }
            if (prob <= 0) prob = 1.0 / 256;
            total_loss += -log2(prob);
            n_pred++;
        }
        printf("Order %2d: %.4f bpc\n", order, total_loss / n_pred);
    }

    trie_free(root);
    free(data);
    return 0;
}
