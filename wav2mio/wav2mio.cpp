#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using BYTE = uint8_t;
using SBYTE = int8_t;
using SWORD = int16_t;
using WORD = uint16_t;
using UDWORD = uint32_t;
using SDWORD = int32_t;
using ULONG = unsigned long;

static constexpr UDWORD ERINA_CODE_FLAG = 0x80000000u;
static constexpr int ERINA_HUFFMAN_ESCAPE = 0x7FFFFFFF;
static constexpr int ERINA_HUFFMAN_NULL = 0x8000;
static constexpr int ERINA_HUFFMAN_MAX = 0x4000;
static constexpr int ERINA_HUFFMAN_ROOT = 0x200;
static constexpr int MIO_HUFFMAN_SYMBOLS = 256;
static constexpr int MIN_DCT_DEGREE = 2;
static constexpr int MAX_DCT_DEGREE = 12;
static constexpr BYTE MIO_LEAD_BLOCK = 0x01;
static constexpr UDWORD CVTYPE_LOSSLESS_ERI = 0x03020000u;
static constexpr UDWORD CVTYPE_LOT_ERI = 0x00000005u;
static constexpr UDWORD CVTYPE_LOT_ERI_MSS = 0x00000105u;
static constexpr UDWORD ERI_RUNLENGTH_HUFFMAN = 0xFFFFFFFCu;
static constexpr UDWORD ERI_RUNLENGTH_GAMMA = 0xFFFFFFFFu;
static constexpr double ERI_PI = 3.141592653589;
static constexpr int LEAD_INTERVAL = 8;
static constexpr int PACKET_SAMPLES = 16384;

struct SinCos {
    float rSin;
    float rCos;
};

struct HuffNode {
    WORD weight;
    WORD parent;
    UDWORD child_code;
};

struct HuffTree {
    HuffNode hn[0x201];
    int symLookup[MIO_HUFFMAN_SYMBOLS];
    int iEscape;
    int iTreePointer;
};

struct BitWriter {
    std::vector<BYTE> buf;
    UDWORD acc = 0;
    int nbits = 0;

    void putBits(UDWORD v, int n) {
        while (n > 0) {
            int room = 32 - nbits;
            int take = n < room ? n : room;
            UDWORD chunk = (take >= 32) ? v : ((v >> (n - take)) & (((UDWORD)1 << take) - 1));
            acc = (take >= 32) ? chunk : ((acc << take) | chunk);
            nbits += take;
            n -= take;
            if (nbits == 32) {
                buf.push_back((BYTE)(acc >> 24));
                buf.push_back((BYTE)(acc >> 16));
                buf.push_back((BYTE)(acc >> 8));
                buf.push_back((BYTE)acc);
                acc = 0;
                nbits = 0;
            }
        }
    }

    void putBit(int b) { putBits((UDWORD)(b & 1), 1); }

    void flush() {
        if (nbits > 0) putBits(0, 32 - nbits);
    }
};

struct MatrixState {
    int nDegreeWidth = 0;
    int nDegreeNum = 0;
    int nSubbandDegree = -1;
    int freqPoint[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int freqWidth[7] = { 0, 0, 0, 0, 0, 0, 0 };
    std::vector<int> revA, revB;
    std::vector<float> revS, revC;
};

struct EncParam {
    double lowWeight = 4.0;
    double middleWeight = 3.0;
    double powerScale = 0.5;
    int oddWeight = 1;
    int peThreshold = 2;
    int degree = 10;
    int useMss = 0;
    bool lossless = false;
};

struct MioInfo {
    UDWORD fdwTransformation = 0;
    UDWORD dwArchitecture = 0;
    UDWORD dwChannelCount = 0;
    UDWORD dwSamplesPerSec = 0;
    UDWORD dwSubbandDegree = 0;
    UDWORD dwAllSampleCount = 0;
    UDWORD dwLappedDegree = 0;
    UDWORD dwBitsPerSample = 0;
};

struct WavData {
    std::vector<SWORD> pcm;
    UDWORD nFrames = 0;
    UDWORD channels = 0;
    UDWORD rate = 0;
};

static float g_riCosPI4 = 0.0f;
static float g_ri2CosPI4 = 0.0f;
static float g_dctK[12][2048];

static void initMatrix() {
    g_riCosPI4 = (float)cos(ERI_PI * 0.25);
    g_ri2CosPI4 = 2.0f * g_riCosPI4;
    for (int i = 1; i < MAX_DCT_DEGREE; i++) {
        int n = 1 << i;
        double nr = ERI_PI / (4.0 * n);
        double dr = nr + nr;
        double ir = nr;
        for (int j = 0; j < n; j++) {
            g_dctK[i][j] = (float)cos(ir);
            ir += dr;
        }
    }
}

static void fastDCT(float* dst, unsigned dstStep, float* src, float* wrk, unsigned deg) {
    if (deg < MIN_DCT_DEGREE || deg > MAX_DCT_DEGREE) return;
    if (deg == MIN_DCT_DEGREE) {
        float b[4];
        b[0] = src[0] + src[3];
        b[2] = src[0] - src[3];
        b[1] = src[1] + src[2];
        b[3] = src[1] - src[2];
        dst[0] = 0.5f * (b[0] + b[1]);
        dst[dstStep * 2] = g_riCosPI4 * (b[0] - b[1]);
        b[2] = g_dctK[1][0] * b[2];
        b[3] = g_dctK[1][1] * b[3];
        b[0] = b[2] + b[3];
        b[1] = g_ri2CosPI4 * (b[2] - b[3]);
        b[1] -= b[0];
        dst[dstStep] = b[0];
        dst[dstStep * 3] = b[1];
        return;
    }
    unsigned num = 1u << deg;
    unsigned half = num >> 1;
    for (unsigned i = 0; i < half; i++) {
        wrk[i] = src[i] + src[num - i - 1];
        wrk[i + half] = src[i] - src[num - i - 1];
    }
    unsigned step2 = dstStep << 1;
    fastDCT(dst, step2, wrk, src, deg - 1);
    float* k = g_dctK[deg - 1];
    src = wrk + half;
    dst += dstStep;
    for (unsigned i = 0; i < half; i++) src[i] *= k[i];
    fastDCT(dst, step2, src, wrk, deg - 1);
    float* p = dst;
    for (unsigned i = 0; i < half; i++) { *p += *p; p += step2; }
    p = dst;
    for (unsigned i = 1; i < half; i++) { p[step2] -= *p; p += step2; }
}

static void idctInverse(float* dst, const float* src, int deg) {
    int n = 1 << deg;
    std::vector<float> a(src, src + n);
    std::vector<float> w(n);
    fastDCT(dst, 1, a.data(), w.data(), (unsigned)deg);
    float s = (float)(2.0 / n);
    for (int i = 0; i < n; i++) dst[i] *= s;
}

static void revolve2x2(float* p1, float* p2, float s, float c, int step, int cnt) {
    for (int i = 0; i < cnt; i++) {
        float a = *p1, b = *p2;
        *p1 = a * c - b * s;
        *p2 = a * s + b * c;
        p1 += step;
        p2 += step;
    }
}

static void revolve2x2Inv(float* p1, float* p2, float s, float c, int step, int cnt) {
    for (int i = 0; i < cnt; i++) {
        float a = *p1, b = *p2;
        *p1 = a * c + b * s;
        *p2 = b * c - a * s;
        p1 += step;
        p2 += step;
    }
}

static void buildRevolve(MatrixState& ms, int deg) {
    int num = 1 << deg;
    int lc = 1, n = num / 2;
    while (n >= 8) { n /= 8; ++lc; }
    std::vector<SinCos> rev((size_t)lc * 8);
    double kk = ERI_PI / (num * 2.0);
    SinCos* next = rev.data();
    int step = 2;
    do {
        for (int i = 0; i < 7; i++) {
            double ws = 1.0, a = 0.0;
            for (int j = 0; j < i; j++) {
                a += step;
                ws = ws * next[j].rSin + next[j].rCos * cos(a * kk);
            }
            double r = atan2(ws, cos((a + step) * kk));
            next[i].rSin = (float)sin(r);
            next[i].rCos = (float)cos(r);
        }
        next += 7;
        step *= 8;
    } while (step < num);

    ms.revA.clear(); ms.revB.clear(); ms.revS.clear(); ms.revC.clear();
    SinCos* pr = rev.data();
    int index = 1, st = 2, lcc = (num / 2) / 8;
    for (;;) {
        pr += 7;
        index += st * 7;
        st *= 8;
        if (lcc <= 8) break;
        lcc /= 8;
    }
    int k = index + st * (lcc - 2);
    for (int j = lcc - 2; j >= 0; j--) {
        ms.revA.push_back(k); ms.revB.push_back(k + st);
        ms.revS.push_back(pr[j].rSin); ms.revC.push_back(pr[j].rCos);
        k -= st;
    }
    for (;;) {
        if (lcc > (num / 2) / 8) break;
        pr -= 7;
        st /= 8;
        index -= st * 7;
        for (int i = 0; i < lcc; i++) {
            k = i * (st * 8) + index + st * 6;
            for (int j = 6; j >= 0; j--) {
                ms.revA.push_back(k); ms.revB.push_back(k + st);
                ms.revS.push_back(pr[j].rSin); ms.revC.push_back(pr[j].rCos);
                k -= st;
            }
        }
        lcc *= 8;
    }
}

static void oddGivensForward(float* v, MatrixState& ms) {
    for (int t = (int)ms.revA.size() - 1; t >= 0; t--) {
        int a = ms.revA[t], b = ms.revB[t];
        float s = ms.revS[t], c = ms.revC[t];
        float x = v[a], y = v[b];
        v[a] = x * c - y * s;
        v[b] = y * c + x * s;
    }
}

static void plotInverse(float* g, const float* p, int n) {
    for (int i = 0; i < n; i += 2) {
        g[i] = p[i] + p[i + 1];
        g[i + 1] = p[i] - p[i + 1];
    }
}

static void lotSplit(const float* o, float* pe, float* po, int n) {
    for (int i = 0; i < n; i += 2) {
        pe[i] = 0.5f * (o[i] + o[i + 1]);
        po[i + 1] = 0.5f * (o[i] - o[i + 1]);
    }
}

static void matrixInit(MatrixState& ms, int deg) {
    static const int fw[7] = { -6, -6, -5, -4, -3, -2, -1 };
    if (ms.nSubbandDegree == deg && !ms.revA.empty()) return;
    ms.nSubbandDegree = deg;
    ms.nDegreeWidth = 1 << deg;
    ms.nDegreeNum = 1 << deg;
    int j = 0;
    for (int i = 0; i < 7; i++) {
        int w = 1 << (deg + fw[i]);
        ms.freqWidth[i] = w;
        ms.freqPoint[i] = j + (w / 2);
        j += w;
    }
    buildRevolve(ms, deg);
}

static void ehtInit(HuffTree* t) {
    for (int i = 0; i < MIO_HUFFMAN_SYMBOLS; i++) t->symLookup[i] = ERINA_HUFFMAN_NULL;
    t->iEscape = ERINA_HUFFMAN_NULL;
    t->iTreePointer = ERINA_HUFFMAN_ROOT;
    memset(t->hn, 0, sizeof(t->hn));
    t->hn[ERINA_HUFFMAN_ROOT].weight = 0;
    t->hn[ERINA_HUFFMAN_ROOT].parent = ERINA_HUFFMAN_NULL;
    t->hn[ERINA_HUFFMAN_ROOT].child_code = ERINA_HUFFMAN_NULL;
}

static void ehtRecount(HuffTree* t, int p) {
    int c = (int)t->hn[p].child_code;
    t->hn[p].weight = (WORD)(t->hn[c].weight + t->hn[c + 1].weight);
}

static void ehtSetLeafAt(HuffTree* t, int idx, UDWORD cc) {
    if (!(cc & ERINA_CODE_FLAG)) return;
    int code = (int)(cc & ~ERINA_CODE_FLAG);
    if (code != ERINA_HUFFMAN_ESCAPE) t->symLookup[code & 0xFF] = idx;
    else t->iEscape = idx;
}

static void ehtNormalize(HuffTree* t, int entry) {
    while (entry < ERINA_HUFFMAN_ROOT) {
        int swap = entry + 1;
        WORD w = t->hn[entry].weight;
        while (swap < ERINA_HUFFMAN_ROOT) {
            if (t->hn[swap].weight >= w) break;
            ++swap;
        }
        if (entry == --swap) {
            entry = (int)t->hn[entry].parent;
            ehtRecount(t, entry);
            continue;
        }
        if (!(t->hn[entry].child_code & ERINA_CODE_FLAG)) {
            int ch = (int)t->hn[entry].child_code;
            t->hn[ch].parent = (WORD)swap;
            t->hn[ch + 1].parent = (WORD)swap;
        }
        else {
            ehtSetLeafAt(t, swap, t->hn[entry].child_code);
        }
        if (!(t->hn[swap].child_code & ERINA_CODE_FLAG)) {
            int ch = (int)t->hn[swap].child_code;
            t->hn[ch].parent = (WORD)entry;
            t->hn[ch + 1].parent = (WORD)entry;
        }
        else {
            ehtSetLeafAt(t, entry, t->hn[swap].child_code);
        }
        HuffNode node;
        WORD pe = t->hn[entry].parent;
        WORD ps = t->hn[swap].parent;
        node = t->hn[swap];
        t->hn[swap] = t->hn[entry];
        t->hn[entry] = node;
        t->hn[swap].parent = ps;
        t->hn[entry].parent = pe;
        ehtSetLeafAt(t, entry, t->hn[entry].child_code);
        ehtSetLeafAt(t, swap, t->hn[swap].child_code);
        ehtRecount(t, ps);
        entry = ps;
    }
}

static void ehtAddNew(HuffTree* t, int code) {
    if (t->iTreePointer > 0) {
        int i = t->iTreePointer = t->iTreePointer - 2;
        HuffNode* nb = &t->hn[i];
        nb->weight = 1;
        nb->child_code = ERINA_CODE_FLAG | (UDWORD)code;
        t->symLookup[code & 0xFF] = i;
        HuffNode* root = &t->hn[ERINA_HUFFMAN_ROOT];
        if (root->child_code != (UDWORD)ERINA_HUFFMAN_NULL) {
            HuffNode* parent = &t->hn[i + 2];
            HuffNode* child = &t->hn[i + 1];
            t->hn[i + 1] = t->hn[i + 2];
            ehtSetLeafAt(t, i + 1, t->hn[i + 1].child_code);
            parent->weight = (WORD)(nb->weight + child->weight);
            parent->parent = child->parent;
            parent->child_code = (UDWORD)i;
            nb->parent = child->parent = (WORD)(i + 2);
            ehtNormalize(t, i + 2);
        }
        else {
            nb->parent = ERINA_HUFFMAN_ROOT;
            HuffNode* esc = &t->hn[t->iEscape = i + 1];
            esc->weight = 1;
            esc->parent = ERINA_HUFFMAN_ROOT;
            esc->child_code = ERINA_CODE_FLAG | (UDWORD)ERINA_HUFFMAN_ESCAPE;
            root->weight = 2;
            root->child_code = (UDWORD)i;
        }
    }
    else {
        int i = t->iTreePointer;
        HuffNode* e = &t->hn[i];
        if (e->child_code == (ERINA_CODE_FLAG | (UDWORD)ERINA_HUFFMAN_ESCAPE)) e = &t->hn[i + 1];
        int idx = (int)(e - t->hn);
        e->child_code = ERINA_CODE_FLAG | (UDWORD)code;
        t->symLookup[code & 0xFF] = idx;
    }
}

static void ehtHalfRebuild(HuffTree* t) {
    int next = ERINA_HUFFMAN_ROOT;
    for (int i = ERINA_HUFFMAN_ROOT - 1; i >= t->iTreePointer; i--) {
        if (t->hn[i].child_code & ERINA_CODE_FLAG) {
            t->hn[i].weight = (WORD)((t->hn[i].weight + 1) >> 1);
            t->hn[next--] = t->hn[i];
        }
    }
    ++next;
    int i = t->iTreePointer;
    for (;;) {
        t->hn[i] = t->hn[next];
        if (next + 1 <= ERINA_HUFFMAN_ROOT) {
            t->hn[i + 1] = t->hn[next + 1];
        }
        else {
            t->hn[i + 1] = t->hn[ERINA_HUFFMAN_ROOT];
        }
        next += 2;
        HuffNode* c1 = &t->hn[i];
        HuffNode* c2 = &t->hn[i + 1];
        if (!(c1->child_code & ERINA_CODE_FLAG)) {
            int ch = (int)c1->child_code;
            t->hn[ch].parent = (WORD)i;
            t->hn[ch + 1].parent = (WORD)i;
        }
        else ehtSetLeafAt(t, i, c1->child_code);
        if (!(c2->child_code & ERINA_CODE_FLAG)) {
            int ch = (int)c2->child_code;
            t->hn[ch].parent = (WORD)(i + 1);
            t->hn[ch + 1].parent = (WORD)(i + 1);
        }
        else ehtSetLeafAt(t, i + 1, c2->child_code);
        WORD w = (WORD)(c1->weight + c2->weight);
        if (next <= ERINA_HUFFMAN_ROOT) {
            int j = next;
            for (;;) {
                if (w <= t->hn[j].weight) {
                    t->hn[j - 1].weight = w;
                    t->hn[j - 1].child_code = (UDWORD)i;
                    break;
                }
                t->hn[j - 1] = t->hn[j];
                if (++j > ERINA_HUFFMAN_ROOT) {
                    t->hn[ERINA_HUFFMAN_ROOT].weight = w;
                    t->hn[ERINA_HUFFMAN_ROOT].child_code = (UDWORD)i;
                    break;
                }
            }
            --next;
        }
        else {
            t->hn[ERINA_HUFFMAN_ROOT].weight = w;
            t->hn[ERINA_HUFFMAN_ROOT].parent = ERINA_HUFFMAN_NULL;
            t->hn[ERINA_HUFFMAN_ROOT].child_code = (UDWORD)i;
            c1->parent = ERINA_HUFFMAN_ROOT;
            c2->parent = ERINA_HUFFMAN_ROOT;
            break;
        }
        i += 2;
    }
}

static void ehtIncrease(HuffTree* t, int entry) {
    t->hn[entry].weight++;
    ehtNormalize(t, entry);
    if (t->hn[ERINA_HUFFMAN_ROOT].weight >= ERINA_HUFFMAN_MAX) ehtHalfRebuild(t);
}

static int leafCodeAt(HuffTree* t, int idx) {
    if (idx < 0 || idx > ERINA_HUFFMAN_ROOT) return -2;
    UDWORD cc = t->hn[idx].child_code;
    if (!(cc & ERINA_CODE_FLAG)) return -2;
    return (int)(cc & ~ERINA_CODE_FLAG);
}

static int findEntry(HuffTree* t, int code) {
    int i = t->symLookup[code & 0xFF];
    if (i != ERINA_HUFFMAN_NULL && leafCodeAt(t, i) == code) return i;
    for (int k = t->iTreePointer; k <= ERINA_HUFFMAN_ROOT; k++)
        if (leafCodeAt(t, k) == code) return k;
    return ERINA_HUFFMAN_NULL;
}

static void putGamma(BitWriter& bw, int v) {
    if (v == 1) { bw.putBit(0); return; }
    bw.putBit(1);
    int k = 0, x = v;
    while (x >= 2) { x >>= 1; k++; }
    int code = v - (1 << k);
    for (int j = k - 1; j >= 0; j--) {
        bw.putBit((code >> j) & 1);
        bw.putBit(j > 0 ? 1 : 0);
    }
}

static void emitPath(BitWriter& bw, HuffTree* t, int entry) {
    std::vector<int> st;
    st.reserve(64);
    int cur = entry;
    if (cur < 0 || cur > ERINA_HUFFMAN_ROOT) return;
    while (cur != ERINA_HUFFMAN_ROOT && (int)st.size() < 0x200) {
        int parent = (int)t->hn[cur].parent;
        if (parent == (int)ERINA_HUFFMAN_NULL || parent < 0 || parent > ERINA_HUFFMAN_ROOT) break;
        UDWORD cc = t->hn[parent].child_code;
        int base = (cc & ERINA_CODE_FLAG) ? (int)(cc & ~ERINA_CODE_FLAG) : (int)cc;
        st.push_back(cur == base + 1 ? 1 : 0);
        cur = parent;
    }
    for (int i = (int)st.size() - 1; i >= 0; i--) bw.putBit(st[i]);
}

struct ErinaCtx {
    std::vector<HuffTree> trees;
    HuffTree* last = nullptr;
    BitWriter* bw = nullptr;
};

static void encPrepare(ErinaCtx& ctx, BitWriter& bw) {
    if (ctx.trees.empty()) ctx.trees.resize(0x101);
    ctx.bw = &bw;
    for (auto& t : ctx.trees) ehtInit(&t);
    ctx.last = &ctx.trees[0];
}

static void encHuffman(ErinaCtx& ctx, HuffTree* tree, int code, bool gamma) {
    int entry = (tree->iEscape != ERINA_HUFFMAN_NULL) ? findEntry(tree, code) : ERINA_HUFFMAN_NULL;
    if (entry != ERINA_HUFFMAN_NULL) {
        emitPath(*ctx.bw, tree, entry);
        ehtIncrease(tree, entry);
        return;
    }
    if (tree->iEscape != ERINA_HUFFMAN_NULL) {
        int esc = tree->iEscape;
        emitPath(*ctx.bw, tree, esc);
        ehtIncrease(tree, esc);
    }
    if (gamma) putGamma(*ctx.bw, code);
    else ctx.bw->putBits((UDWORD)(code & 0xFF), 8);
    ehtAddNew(tree, code);
}

static void encSymbolBytes(ErinaCtx& ctx, const SBYTE* src, ULONG n) {
    HuffTree* tree = ctx.last;
    ULONG i = 0;
    while (i < n) {
        int sym = (int)(BYTE)src[i];
        if (sym == 0) {
            ULONG run = 0;
            while (i + run < n && (BYTE)src[i + run] == 0) run++;
            if (run > 0x7FFFFFFEUL) run = 0x7FFFFFFEUL;
            encHuffman(ctx, tree, 0, false);
            tree = &ctx.trees[0];
            encHuffman(ctx, &ctx.trees[0x100], (int)run, true);
            i += run;
        }
        else {
            encHuffman(ctx, tree, sym, false);
            tree = &ctx.trees[sym & 0xFF];
            i++;
        }
    }
    ctx.last = tree;
}

static int roundF32(float r) {
    return (r >= 0.0f) ? (int)floor(r + 0.5) : (int)ceil(r - 0.5);
}

static void buildWeightTable(MatrixState& ms, float* wt, const double* avg) {
    int n = ms.nDegreeNum;
    int i = 0;
    for (i = 0; i < ms.freqPoint[0]; i++) wt[i] = (float)avg[0];
    for (int j = 1; j < 7; j++) {
        double a = avg[j - 1];
        double k = (avg[j] - a) / (ms.freqPoint[j] - ms.freqPoint[j - 1]);
        while (i < ms.freqPoint[j]) {
            wt[i] = (float)(k * (i - ms.freqPoint[j - 1]) + a);
            i++;
        }
    }
    while (i < n) { wt[i] = (float)avg[6]; i++; }
}

struct MioEncoder {
    MatrixState ms;
    MioInfo h;
    EncParam prm;
    std::vector<float> weightTable;

    void quantumize(SDWORD* dst, const float* src, int n, float scale, UDWORD* pW, UDWORD* pC) {
        double tbl[6];
        double lo = prm.lowWeight, mid = prm.middleWeight;
        double ratio = mid / lo;
        tbl[0] = lo * pow(ratio, 0.25);
        tbl[1] = lo * pow(ratio, 0.75);
        tbl[2] = pow(mid, 6.0 / 7.0);
        tbl[3] = pow(mid, 4.0 / 7.0);
        tbl[4] = pow(mid, 2.0 / 7.0);
        tbl[5] = pow(mid, 0.0 / 7.0);
        double avg[7];
        int exps[6];
        for (int i = 0; i < 7; i++) avg[i] = 1.0;
        for (int i = 0; i < 6; i++) exps[i] = 0;

        double bv[7];
        int idx = 0;
        for (int k = 0; k < 7; k++) {
            int cnt = ms.freqWidth[k];
            double s = 0.0;
            for (int i = 0; i < cnt; i++) {
                double x = (double)src[idx++];
                s += fabs(x);
            }
            double mean = (cnt > 0) ? s / cnt : 0.0;
            bv[k] = mean * mean;
        }
        int refBand = -1;
        for (int k = 6; k >= 0; k--) if (bv[k] >= 1.0) { refBand = k; break; }
        if (refBand >= 0) {
            double ref = bv[refBand];
            for (int k = 0; k < 6; k++) {
                double t = bv[k] / ref;
                if (t <= 0.0) break;
                double x = tbl[k] * pow(t, 0.85);
                if (x < 0.0009765625) x = 0.0009765625;
                if (x > 181.01933598375618) x = 181.01933598375618;
                int ex = -roundF32((float)(2.0 * (log(x) / log(2.0))));
                if (ex < -15) ex = -15;
                if (ex > 16) ex = 16;
                avg[k] = 1.0 / pow(2.0, ex * 0.5);
                exps[k] = ex;
            }
        }
        avg[6] = 1.0;

        weightTable.assign((size_t)n, 0.0f);
        buildWeightTable(ms, weightTable.data(), avg);
        double odd = ((prm.oddWeight & 3) + 2) / 2.0;
        for (int i = 15; i < n; i += 16) weightTable[i] *= (float)odd;

        double maxT = 0.0;
        for (int i = 0; i < n; i++) {
            double t = fabs((double)src[i] * (double)weightTable[i]);
            if (t > maxT) maxT = t;
        }
        double gain = (scale > 0.0f) ? ((double)scale / 0.1640625) : 1.0;
        if (gain < 1.0) gain = 1.0;
        int nCoef = roundF32((float)(maxT * gain / 32767.0));
        if (nCoef < 1) nCoef = 1;
        if (nCoef >= 0x10000) nCoef = 0xFFFF;

        weightTable[n - 1] = (float)nCoef;
        double inv = 1.0 / (double)nCoef;
        for (int i = 0; i < n; i++) {
            double v = (double)src[i] * (double)weightTable[i] * inv;
            int q = roundF32((float)v);
            if (q < -0x8000) q = -0x8000;
            if (q > 0x7FFF) q = 0x7FFF;
            dst[i] = q;
        }
        UDWORD wc = 0;
        for (int i = 0; i < 6; i++) {
            int ex = exps[i];
            if (ex < -15) ex = -15;
            if (ex > 16) ex = 16;
            wc |= ((UDWORD)(ex + 15) & 0x1F) << (5 * i);
        }
        wc |= ((UDWORD)(prm.oddWeight & 3)) << 30;
        *pW = wc;
        *pC = (UDWORD)nCoef;
    }
};

struct Blk {
    int kind = 1;
    int deg = 0;
    int n = 0;
    int nIn = 1;
    int ch = 0;
    int newGrp = 0;
    int dc = 0;
    std::vector<float> pe, po;
};

static bool readWav(const std::string& path, WavData& w) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<BYTE> d((size_t)sz);
    if (fread(d.data(), 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);
    if (sz < 44 || memcmp(d.data(), "RIFF", 4) || memcmp(d.data() + 8, "WAVE", 4)) return false;
    long p = 12;
    UDWORD ch = 0, rate = 0, bits = 0;
    const BYTE* data = nullptr;
    UDWORD dlen = 0;
    while (p + 8 <= sz) {
        UDWORD id = *(UDWORD*)(d.data() + p);
        UDWORD len = *(UDWORD*)(d.data() + p + 4);
        if (id == 0x20746D66 && p + 8 + 16 <= sz) {
            ch = *(WORD*)(d.data() + p + 10);
            rate = *(UDWORD*)(d.data() + p + 12);
            bits = *(WORD*)(d.data() + p + 22);
        }
        else if (id == 0x61746164) {
            data = d.data() + p + 8;
            dlen = len;
            if ((long)(p + 8 + dlen) > sz) dlen = (UDWORD)(sz - p - 8);
        }
        p += 8 + len + (len & 1);
    }
    if (!data || bits != 16 || ch < 1 || ch > 2) return false;
    UDWORD frames = dlen / (2 * ch);
    w.pcm.resize((size_t)frames * ch);
    memcpy(w.pcm.data(), data, sizeof(SWORD) * frames * ch);
    w.nFrames = frames;
    w.channels = ch;
    w.rate = rate;
    return true;
}

static long g_dataSizePos = 0;
static long g_streamSizePos = 0;

static void writeHeader(FILE* f, MioInfo& h, UDWORD allSamples, UDWORD rate) {
    BYTE hdr[0x40];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, "Entis\x1a\x00\x00", 8);
    *(UDWORD*)(hdr + 0x08) = 0x03000100;
    memcpy(hdr + 0x10, "Music Interleaved and Orthogonal", 32);
    fwrite(hdr, 1, 0x40, f);
    g_dataSizePos = 0x38;
    BYTE rec[0x10];
    memset(rec, 0, sizeof(rec));
    memcpy(rec, "Header  ", 8);
    *(uint64_t*)(rec + 8) = 0x24 + 0x38 + 0x12;
    fwrite(rec, 1, 0x10, f);
    BYTE b[0x40];
    memset(b, 0, sizeof(b));
    memcpy(b, "FileHdr ", 8);
    *(uint64_t*)(b + 8) = 0x14;
    *(UDWORD*)(b + 0x10 + 0x00) = 0x00020100;
    *(UDWORD*)(b + 0x10 + 0x04) = 0x00000100;
    *(UDWORD*)(b + 0x10 + 0x08) = 1;
    *(UDWORD*)(b + 0x10 + 0x0c) = 0;
    *(UDWORD*)(b + 0x10 + 0x10) = (UDWORD)((uint64_t)allSamples * 1000 / (rate ? rate : 44100));
    fwrite(b, 1, 0x24, f);
    memset(b, 0, sizeof(b));
    memcpy(b, "SoundInf", 8);
    *(uint64_t*)(b + 8) = 0x28;
    *(UDWORD*)(b + 0x10 + 0x00) = 0x00020300;
    *(UDWORD*)(b + 0x10 + 0x04) = h.fdwTransformation;
    *(UDWORD*)(b + 0x10 + 0x08) = h.dwArchitecture;
    *(UDWORD*)(b + 0x10 + 0x0c) = h.dwChannelCount;
    *(UDWORD*)(b + 0x10 + 0x10) = h.dwSamplesPerSec;
    *(UDWORD*)(b + 0x10 + 0x14) = 0;
    *(UDWORD*)(b + 0x10 + 0x18) = h.dwSubbandDegree;
    *(UDWORD*)(b + 0x10 + 0x1c) = h.dwAllSampleCount;
    *(UDWORD*)(b + 0x10 + 0x20) = h.dwLappedDegree;
    *(UDWORD*)(b + 0x10 + 0x24) = h.dwBitsPerSample;
    fwrite(b, 1, 0x38, f);
    memset(rec, 0, sizeof(rec));
    memcpy(rec, "descript", 8);
    *(uint64_t*)(rec + 8) = 2;
    fwrite(rec, 1, 0x10, f);
    BYTE bom[2] = { 0xFF, 0xFE };
    fwrite(bom, 1, 2, f);
    memset(rec, 0, sizeof(rec));
    memcpy(rec, "Stream  ", 8);
    fwrite(rec, 1, 0x10, f);
    g_streamSizePos = ftell(f) - 8;
}

static void finishHeader(FILE* f) {
    long total = ftell(f);
    uint64_t ds = (uint64_t)total - 0x40;
    uint64_t ss = (uint64_t)(total - (g_streamSizePos + 8));
    fseek(f, g_dataSizePos, SEEK_SET);
    fwrite(&ds, 1, 8, f);
    fseek(f, g_streamSizePos, SEEK_SET);
    fwrite(&ss, 1, 8, f);
    fseek(f, total, SEEK_SET);
}

static void writePacket(FILE* f, bool lead, UDWORD samples, const std::vector<BYTE>& payload) {
    BYTE rec[0x10];
    memset(rec, 0, sizeof(rec));
    memcpy(rec, "SoundStm", 8);
    *(uint64_t*)(rec + 8) = 8 + payload.size();
    fwrite(rec, 1, 0x10, f);
    BYTE hd[8];
    hd[0] = 1;
    hd[1] = (BYTE)(lead ? MIO_LEAD_BLOCK : 0);
    hd[2] = 0;
    hd[3] = 0;
    *(UDWORD*)(hd + 4) = samples;
    fwrite(hd, 1, 8, f);
    fwrite(payload.data(), 1, payload.size(), f);
}

static void encodeFile(const WavData& w, EncParam prm, const std::string& outPath) {
    MioEncoder enc;
    enc.prm = prm;
    enc.h.fdwTransformation = prm.lossless ? CVTYPE_LOSSLESS_ERI
        : ((prm.useMss && w.channels == 2) ? CVTYPE_LOT_ERI_MSS : CVTYPE_LOT_ERI);
    enc.h.dwArchitecture = prm.lossless ? ERI_RUNLENGTH_HUFFMAN : ERI_RUNLENGTH_GAMMA;
    enc.h.dwChannelCount = w.channels;
    enc.h.dwSamplesPerSec = w.rate;
    enc.h.dwSubbandDegree = prm.lossless ? 0 : (UDWORD)prm.degree;
    enc.h.dwAllSampleCount = w.nFrames;
    enc.h.dwLappedDegree = prm.lossless ? 0 : 1;
    enc.h.dwBitsPerSample = 16;

    ErinaCtx ctx;
    int pktSamples = PACKET_SAMPLES;
    UDWORD nPackets = (w.nFrames + pktSamples - 1) / pktSamples;
    if (nPackets == 0) nPackets = 1;

    FILE* fo = fopen(outPath.c_str(), "wb");
    if (!fo) { fprintf(stderr, "cannot open output\n"); exit(1); }
    writeHeader(fo, enc.h, w.nFrames, w.rate);

    if (!prm.lossless) matrixInit(enc.ms, prm.degree);

    for (UDWORD pk = 0; pk < nPackets; pk++) {
        UDWORD start = pk * (UDWORD)pktSamples;
        UDWORD cnt = w.nFrames - start;
        if (cnt > (UDWORD)pktSamples) cnt = (UDWORD)pktSamples;
        bool lead = (pk % LEAD_INTERVAL == 0);
        BitWriter bw;
        if (prm.lossless) {
            UDWORD nBytes = cnt * w.channels * 2;
            std::vector<SBYTE> sym(nBytes);
            for (UDWORD c = 0; c < w.channels; c++) {
                SWORD prev = 0, pd = 0;
                for (UDWORD i = 0; i < cnt; i++) {
                    SWORD cur = w.pcm[(start + i) * w.channels + c];
                    SWORD s = (SWORD)(cur - (SWORD)(prev * 2) + pd);
                    pd = prev;
                    prev = cur;
                    BYTE lo = (BYTE)(s & 0xFF);
                    BYTE hi = (BYTE)((((UDWORD)s) >> 8) & 0xFF) ^ (BYTE)(((SBYTE)lo) >> 7);
                    sym[(size_t)c * cnt * 2 + i] = (SBYTE)hi;
                    sym[(size_t)c * cnt * 2 + cnt + i] = (SBYTE)lo;
                }
            }
            if (lead) encPrepare(ctx, bw);
            encSymbolBytes(ctx, sym.data(), nBytes);
        }
        else {
            int ndw = 1 << prm.degree;
            UDWORD nSampleCount = ((cnt + ndw - 1) / ndw) * ndw;
            UDWORD nSubbandCount = nSampleCount / ndw;
            UDWORD nAllSampleCount = nSampleCount * w.channels;
            bool isMss = (enc.h.fdwTransformation == CVTYPE_LOT_ERI_MSS);
            int nInCh = isMss ? 2 : 1;
            UDWORD unitW = isMss ? (UDWORD)(ndw * 2) : ndw;
            UDWORD unitJ = isMss ? nSubbandCount : (nSubbandCount * w.channels);
            UDWORD nGrp = isMss ? nSubbandCount : (nSubbandCount * w.channels);
            int nChans = isMss ? 1 : (int)w.channels;

            std::vector<SWORD> pad((size_t)nSampleCount * w.channels + 8, 0);
            memcpy(pad.data(), &w.pcm[(size_t)start * w.channels], sizeof(SWORD) * cnt * w.channels);

            std::vector<UDWORD> divs(nGrp + 4, 0);
            for (UDWORD g = 0; g < nGrp; g++) {
                UDWORD ch0 = isMss ? 0 : (g % w.channels);
                UDWORD sb0 = isMss ? g : (g / w.channels);
                double sum = 0.0, mn = 0.0;
                int nb = ndw / 64;
                int fire = 0;
                if (nb <= 0) nb = 1;
                for (int bb = 0; bb < nb; bb++) {
                    double s2 = 0.0;
                    for (int t = 1; t < 64; t++) {
                        long a1 = (long)pad[((size_t)sb0 * ndw + bb * 64 + t) * w.channels + ch0];
                        long a0 = (long)pad[((size_t)sb0 * ndw + bb * 64 + t - 1) * w.channels + ch0];
                        double dd = (double)(a1 - a0);
                        s2 += dd * dd;
                    }
                    double mean = s2 / 63.0;
                    if (bb >= 1 && (double)prm.peThreshold * mn > mean) { fire = 1; break; }
                    sum += mean;
                    mn = sum / (double)(bb + 1);
                }
                divs[g] = fire ? 2 : 0;
            }

            std::vector<Blk> bl;
            std::vector<int> pld(nChans + 2, -1);
            for (UDWORD gi = 0; gi < nGrp; gi++) {
                UDWORD sb0 = isMss ? gi : (gi / w.channels);
                int j2 = isMss ? 0 : (int)(gi % w.channels);
                UDWORD dc = divs[gi];
                bool changed = (pld[j2] != (int)dc);
                int nDivCount = 1 << dc;
                int degB = prm.degree - (int)dc;
                if (changed && sb0 != 0) {
                    int degO = prm.degree - pld[j2];
                    Blk b;
                    b.kind = 2; b.deg = degO; b.n = 1 << degO; b.nIn = nInCh; b.ch = j2;
                    b.newGrp = 1; b.dc = (int)dc;
                    b.pe.assign((size_t)b.n * nInCh, 0.0f);
                    b.po.assign((size_t)b.n * nInCh, 0.0f);
                    bl.push_back(b);
                }
                if (changed) pld[j2] = (int)dc;
                for (int k = 0; k < nDivCount; k++) {
                    Blk b;
                    b.kind = (changed && k == 0) ? 0 : 1;
                    b.deg = degB; b.n = 1 << degB; b.nIn = nInCh; b.ch = j2;
                    b.newGrp = (k == 0 && !(changed && sb0 != 0)) ? 1 : 0;
                    b.dc = (int)dc;
                    b.pe.assign((size_t)b.n * nInCh, 0.0f);
                    b.po.assign((size_t)b.n * nInCh, 0.0f);
                    bl.push_back(b);
                }
            }
            for (int j = 0; j < nChans; j++) {
                int degO = prm.degree - pld[j];
                Blk b;
                b.kind = 2; b.deg = degO; b.n = 1 << degO; b.nIn = nInCh; b.ch = j;
                b.newGrp = 0; b.dc = 0;
                b.pe.assign((size_t)b.n * nInCh, 0.0f);
                b.po.assign((size_t)b.n * nInCh, 0.0f);
                bl.push_back(b);
            }

            std::vector<float> oBuf((size_t)(1 << MAX_DCT_DEGREE) * 2 + 8);
            std::vector<float> gBuf((size_t)(1 << MAX_DCT_DEGREE) * 2 + 8);
            std::vector<float> zBuf((size_t)(1 << MAX_DCT_DEGREE) * 2 + 8);
            std::vector<float> zTmp((size_t)(1 << MAX_DCT_DEGREE) * 2 + 8);
            std::vector<float> pBuf((size_t)(1 << MAX_DCT_DEGREE) * 2 + 8);
            std::vector<SDWORD> qtmp((size_t)(1 << MAX_DCT_DEGREE) * 2 + 8);
            std::vector<SDWORD> qall((size_t)nAllSampleCount + 16, 0);
            UDWORD qn = 0;
            std::vector<UDWORD> wlist, clist, rlist;
            float scale = (float)prm.powerScale;
            std::vector<UDWORD> spos(nChans + 4, 0);

            for (auto& b : bl) {
                int n = b.n;
                if (b.kind == 0) continue;
                matrixInit(enc.ms, b.deg);
                for (int c = 0; c < b.nIn; c++) {
                    int srcCh = isMss ? c : b.ch;
                    for (int t = 0; t < n; t++) {
                        UDWORD si = spos[srcCh] + (UDWORD)t;
                        oBuf[(size_t)c * n + t] = (si < nSampleCount) ? (float)pad[(size_t)si * w.channels + srcCh] : 0.0f;
                    }
                    idctInverse(b.pe.data() + (size_t)c * n, oBuf.data() + (size_t)c * n, b.deg);
                    lotSplit(b.pe.data() + (size_t)c * n, b.pe.data() + (size_t)c * n, b.po.data() + (size_t)c * n, n);
                }
                for (int c = 0; c < b.nIn; c++) {
                    int srcCh = isMss ? c : b.ch;
                    spos[srcCh] += (UDWORD)n;
                }
            }

            for (size_t bi = 0; bi < bl.size(); bi++) {
                Blk& b = bl[bi];
                int n = b.n;
                Blk* nx = nullptr;
                for (size_t bj = bi + 1; bj < bl.size(); bj++) {
                    if (isMss || bl[bj].ch == b.ch) { nx = &bl[bj]; break; }
                }
                if (nx && nx->n != n) nx = nullptr;
                matrixInit(enc.ms, b.deg);
                for (int c = 0; c < b.nIn; c++) {
                    float* p = pBuf.data() + (size_t)c * n;
                    for (int i = 0; i < n; i += 2) {
                        if (b.kind == 2) {
                            p[i] = 0.0f;
                            p[i + 1] = b.po[(size_t)c * n + i + 1];
                        }
                        else {
                            p[i] = nx ? nx->pe[(size_t)c * n + i] : 0.0f;
                            p[i + 1] = (b.kind == 0) ? 0.0f : b.po[(size_t)c * n + i + 1];
                        }
                    }
                    plotInverse(gBuf.data() + (size_t)c * n, p, n);
                    for (int i = 0; i < n; i++) zBuf[(size_t)c * n + i] = gBuf[(size_t)c * n + i];
                    oddGivensForward(zBuf.data() + (size_t)c * n, enc.ms);
                    if (b.kind != 1) {
                        for (int i = 0; i < n; i += 2) zBuf[(size_t)c * n + i] = 0.0f;
                    }
                }
                int nRev = 0;
                if (isMss) {
                    int tries = (b.kind == 1) ? 16 : 4;
                    double bestErr = -1.0;
                    for (int rc = 0; rc < tries; rc++) {
                        for (int i = 0; i < 2 * n; i++) zTmp[i] = zBuf[i];
                        if (b.kind == 1) {
                            int r1 = (rc >> 2) & 3, r2 = rc & 3;
                            revolve2x2Inv(zTmp.data(), zTmp.data() + n, (float)sin(r1 * ERI_PI / 8), (float)cos(r1 * ERI_PI / 8), 2, n / 2);
                            revolve2x2Inv(zTmp.data() + 1, zTmp.data() + 1 + n, (float)sin(r2 * ERI_PI / 8), (float)cos(r2 * ERI_PI / 8), 2, n / 2);
                        }
                        else {
                            revolve2x2Inv(zTmp.data(), zTmp.data() + n, (float)sin(rc * ERI_PI / 8), (float)cos(rc * ERI_PI / 8), 1, n);
                        }
                        double ms2 = sqrt(2.0 / n);
                        for (int i = 0; i < 2 * n; i++) zTmp[i] = (float)(zTmp[i] / ms2);
                        UDWORD ww, cc;
                        enc.quantumize(qtmp.data(), zTmp.data(), n, scale, &ww, &cc);
                        double err = 0.0;
                        for (int i = 0; i < 2 * n; i++) err += (double)qtmp[i] * (double)qtmp[i];
                        if (bestErr < 0.0 || err < bestErr) { bestErr = err; nRev = rc; }
                    }
                    for (int i = 0; i < 2 * n; i++) zTmp[i] = zBuf[i];
                    if (b.kind == 1) {
                        int r1 = (nRev >> 2) & 3, r2 = nRev & 3;
                        revolve2x2Inv(zTmp.data(), zTmp.data() + n, (float)sin(r1 * ERI_PI / 8), (float)cos(r1 * ERI_PI / 8), 2, n / 2);
                        revolve2x2Inv(zTmp.data() + 1, zTmp.data() + 1 + n, (float)sin(r2 * ERI_PI / 8), (float)cos(r2 * ERI_PI / 8), 2, n / 2);
                    }
                    else {
                        revolve2x2Inv(zTmp.data(), zTmp.data() + n, (float)sin(nRev * ERI_PI / 8), (float)cos(nRev * ERI_PI / 8), 1, n);
                    }
                    for (int i = 0; i < 2 * n; i++) zBuf[i] = zTmp[i];
                }
                rlist.push_back((UDWORD)nRev);
                double ms2 = sqrt(2.0 / n);
                for (int i = 0; i < b.nIn * n; i++) zBuf[i] = (float)(zBuf[i] / ms2);
                UDWORD ww, cc;
                enc.quantumize(qtmp.data(), zBuf.data(), n, scale, &ww, &cc);
                wlist.push_back(ww);
                clist.push_back(cc);
                for (int c = 0; c < b.nIn; c++) {
                    if (b.kind == 1) {
                        for (int t = 0; t < n; t++) qall[qn++] = qtmp[(size_t)c * n + t];
                    }
                    else {
                        for (int t = 0; t < n / 2; t++) qall[qn++] = qtmp[(size_t)c * n + t * 2 + 1];
                    }
                }
            }

            if (qn != nAllSampleCount) { fprintf(stderr, "qn=%u expected=%u\n", qn, nAllSampleCount); exit(2); }
            if (false) fprintf(stderr, "qn=%u expected=%u nGrp=%u blocks=%zu dc=", qn, nAllSampleCount, nGrp, bl.size());
            if (qn != nAllSampleCount) { for (UDWORD g = 0;g < nGrp;g++) fprintf(stderr, "%u,", divs[g]); fprintf(stderr, "\n"); for (auto& bb : bl) fprintf(stderr, "  kind=%d n=%d ch=%d\n", bb.kind, bb.n, bb.ch); exit(2); }
            bw.putBit(0);
            size_t pi = 0, ri = 0;
            for (auto& b : bl) {
                if (b.newGrp) bw.putBits((UDWORD)b.dc, 2);
                if (isMss) bw.putBits(rlist[ri], (b.kind == 1) ? 4 : 2);
                ri++;
                bw.putBits(wlist[pi], 32);
                bw.putBits(clist[pi], 16);
                pi++;
            }
            bw.putBit(0);

            std::vector<SBYTE> sym((size_t)nAllSampleCount * 2 + 8, 0);
            UDWORD halfAll = nAllSampleCount;
            for (UDWORD k = 0; k < nAllSampleCount; k++) {
                SDWORD v = (k < qn) ? qall[k] : 0;
                BYTE lo = (BYTE)(v & 0xFF);
                BYTE hi = (BYTE)((((UDWORD)v) >> 8) & 0xFF) ^ (BYTE)(((SBYTE)lo) >> 7);
                UDWORD i2 = k % unitW;
                UDWORD j2 = k / unitW;
                UDWORD m = i2 * unitJ + j2;
                sym[m] = (SBYTE)hi;
                sym[halfAll + m] = (SBYTE)lo;
            }
            if (lead) encPrepare(ctx, bw);
            encSymbolBytes(ctx, sym.data(), nAllSampleCount * 2);
        }
        bw.flush();
        writePacket(fo, lead, cnt, bw.buf);
    }
    finishHeader(fo);
    fclose(fo);
}

static void loadPreset(int idx, EncParam& p) {
    static const double ps[9] = {
        256.0 / 256.0, 140.0 / 256.0, 108.0 / 256.0, 95.0 / 256.0, 85.0 / 256.0,
        75.0 / 256.0, 57.0 / 256.0, 49.0 / 256.0, 42.0 / 256.0
    };
    static const int ow[9] = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    static const int pe[9] = { 2, 2, 2, 2, 2, 2, 4, 6, 8 };
    static const int dg[9] = { 9, 10, 10, 10, 10, 10, 11, 11, 12 };
    static const int ms[9] = { 0, 0, 1, 1, 1, 1, 1, 1, 1 };
    p.lowWeight = 4.0;
    p.middleWeight = 3.0;
    p.powerScale = ps[idx];
    p.oddWeight = ow[idx];
    p.peThreshold = pe[idx];
    p.degree = dg[idx];
    p.useMss = ms[idx];
    p.lossless = false;
}

static void loadParamsXml(const std::string& path, EncParam& p) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open parameter file\n"); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string t((size_t)sz, 0);
    if (fread(&t[0], 1, (size_t)sz, f) != (size_t)sz) { fclose(f); exit(1); }
    fclose(f);
    size_t pos = 0;
    while ((pos = t.find('<', pos)) != std::string::npos) {
        size_t q = pos + 1;
        std::string name;
        while (q < t.size() && t[q] != '>' && t[q] != ' ' && name.size() < 63) name += t[q++];
        if (q >= t.size() || t[q] != '>') { pos = q; continue; }
        size_t e = t.find("</", q);
        if (e == std::string::npos) break;
        std::string val = t.substr(q + 1, e - q - 1);
        if (name == "degree") p.degree = atoi(val.c_str());
        else if (name == "use_mss") p.useMss = atoi(val.c_str());
        else if (name == "pe_threshold") p.peThreshold = atoi(val.c_str());
        else if (name == "odd_weight") p.oddWeight = atoi(val.c_str());
        else if (name == "power_scale") p.powerScale = atof(val.c_str());
        else if (name == "middle_weight") p.middleWeight = atof(val.c_str());
        else if (name == "low_weight") p.lowWeight = atof(val.c_str());
        else if (name == "lossless_mode") p.lossless = atoi(val.c_str()) != 0;
        pos = e + 2;
    }
}

static void printUsage() {
    printf("wav2mio - WAV to MIO encoder\n\n");
    printf("usage:\n");
    printf("  wav2mio -e <in.wav> <out.mio>\n");
}

static std::string deriveOut(const std::string& src, const std::string& dst) {
    if (dst.size() > 4 && dst.substr(dst.size() - 4) == ".mio") return dst;
    size_t b = src.find_last_of("/\\");
    std::string base = (b == std::string::npos) ? src : src.substr(b + 1);
    if (base.size() > 4 && base.substr(base.size() - 4) == ".wav") base.resize(base.size() - 4);
    return dst + "/" + base + ".mio";
}

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[1]) != "-e") {
        printUsage();
        return 1;
    }

    EncParam prm;
    prm.lossless = true;

    std::string src = argv[2];
    std::string dst = argv[3];

    initMatrix();

    WavData w;
    if (!readWav(src, w)) {
        fprintf(stderr, "cannot read source wav: %s\n", src.c_str());
        return 1;
    }

    encodeFile(w, prm, dst);
    printf("encoded %s -> %s\n", src.c_str(), dst.c_str());
    return 0;
}
