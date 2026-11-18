// CPU reference for LUT-defined approximate INT8 products.  This is an
// experiment accelerator, not synthesizable hardware or an energy model.
#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define NIBBLE_EXPORT extern "C" __declspec(dllexport)
#else
#define NIBBLE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Vectorized eight-output gather.  All accumulation stays in signed int32;
// the Python wrapper proves K * max(abs(product_lut)) <= INT32_MAX first.
static inline void dot_row(const int32_t *a, const int32_t *w,
                           const int32_t *lut, int64_t *out,
                           int k_count, int n_count) {
    int n = 0;
#if defined(__AVX2__)
    for (; n + 32 <= n_count; n += 32) {
        __m256i s0 = _mm256_setzero_si256();
        __m256i s1 = _mm256_setzero_si256();
        __m256i s2 = _mm256_setzero_si256();
        __m256i s3 = _mm256_setzero_si256();
        for (int k = 0; k < k_count; ++k) {
            const __m256i base = _mm256_set1_epi32(a[k]);
            const int32_t *row = w + static_cast<size_t>(k) * n_count + n;
            const __m256i i0 = _mm256_add_epi32(base, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row)));
            const __m256i i1 = _mm256_add_epi32(base, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + 8)));
            const __m256i i2 = _mm256_add_epi32(base, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + 16)));
            const __m256i i3 = _mm256_add_epi32(base, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + 24)));
            s0 = _mm256_add_epi32(s0, _mm256_i32gather_epi32(lut, i0, 4));
            s1 = _mm256_add_epi32(s1, _mm256_i32gather_epi32(lut, i1, 4));
            s2 = _mm256_add_epi32(s2, _mm256_i32gather_epi32(lut, i2, 4));
            s3 = _mm256_add_epi32(s3, _mm256_i32gather_epi32(lut, i3, 4));
        }
        int32_t sums[32];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sums), s0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sums + 8), s1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sums + 16), s2);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sums + 24), s3);
        for (int j = 0; j < 32; ++j) out[n + j] = sums[j];
    }
    for (; n + 8 <= n_count; n += 8) {
        __m256i sum = _mm256_setzero_si256();
        for (int k = 0; k < k_count; ++k) {
            const __m256i base = _mm256_set1_epi32(a[k]);
            const __m256i weights = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + static_cast<size_t>(k) * n_count + n));
            sum = _mm256_add_epi32(sum, _mm256_i32gather_epi32(lut, _mm256_add_epi32(base, weights), 4));
        }
        int32_t sums[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(sums), sum);
        for (int j = 0; j < 8; ++j) out[n + j] = sums[j];
    }
#endif
    for (; n < n_count; ++n) {
        int32_t sum = 0;
        for (int k = 0; k < k_count; ++k)
            sum += lut[a[k] + w[static_cast<size_t>(k) * n_count + n]];
        out[n] = sum;
    }
}

NIBBLE_EXPORT const char *nibble_backend_features() {
#if defined(__AVX2__)
    return "AVX2-gather/OpenMP/int32-dot-int64-output";
#else
    return "scalar/OpenMP/int32-dot-int64-output";
#endif
}

// Inputs are contiguous int64 qcodes in [-127,127], LUT axis [activation,
// weight].  counts[s-1] counts multiplication pairs ending at stage s.
// When collect_stats == 0 the stage LUT is not read and counts stays zero.
NIBBLE_EXPORT int nibble_matmul(
    const int64_t *a, const int64_t *w, const int32_t *product_lut,
    const uint8_t *stage_lut, int64_t *out, int64_t *counts,
    int m_count, int k_count, int n_count, int threads, int collect_stats) {
    if (m_count < 0 || k_count < 0 || n_count < 0 || threads < 1) return 1;
    std::fill(counts, counts + 4, int64_t(0));
    std::vector<int32_t> aa(static_cast<size_t>(m_count) * k_count);
    std::vector<int32_t> ww(static_cast<size_t>(k_count) * n_count);
    int bad = 0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:bad)
    for (int64_t i = 0; i < static_cast<int64_t>(aa.size()); ++i) {
        const int64_t q = a[i];
        if (q < -127 || q > 127) bad |= 1;
        aa[static_cast<size_t>(i)] = static_cast<int32_t>((q + 127) * 255);
    }
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:bad)
    for (int64_t i = 0; i < static_cast<int64_t>(ww.size()); ++i) {
        const int64_t q = w[i];
        if (q < -127 || q > 127) bad |= 1;
        ww[static_cast<size_t>(i)] = static_cast<int32_t>(q + 127);
    }
    if (bad) return 2;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int m = 0; m < m_count; ++m)
        dot_row(aa.data() + static_cast<size_t>(m) * k_count, ww.data(),
                product_lut, out + static_cast<size_t>(m) * n_count,
                k_count, n_count);

    if (collect_stats) {
        // Exact counts without allocating an M*K*N tensor.  At fixed k,
        // observed pair frequencies factor into activation and weight histograms.
        int64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        int invalid_stage = 0;
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:c0,c1,c2,c3) reduction(|:invalid_stage)
        for (int k = 0; k < k_count; ++k) {
            int64_t ah[255] = {}, wh[255] = {};
            int ai[255], wi[255], an = 0, wn = 0;
            for (int m = 0; m < m_count; ++m)
                ++ah[aa[static_cast<size_t>(m) * k_count + k] / 255];
            for (int n = 0; n < n_count; ++n)
                ++wh[ww[static_cast<size_t>(k) * n_count + n]];
            for (int i = 0; i < 255; ++i) {
                if (ah[i]) ai[an++] = i;
                if (wh[i]) wi[wn++] = i;
            }
            for (int i = 0; i < an; ++i) {
                const int av = ai[i];
                for (int j = 0; j < wn; ++j) {
                    const int wv = wi[j];
                    const int64_t multiplicity = ah[av] * wh[wv];
                    switch (stage_lut[av * 255 + wv]) {
                        case 1: c0 += multiplicity; break;
                        case 2: c1 += multiplicity; break;
                        case 3: c2 += multiplicity; break;
                        case 4: c3 += multiplicity; break;
                        default: invalid_stage |= 1; break;
                    }
                }
            }
        }
        if (invalid_stage) return 3;
        counts[0] = c0; counts[1] = c1; counts[2] = c2; counts[3] = c3;
    }
    return 0;
}
