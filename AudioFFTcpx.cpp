// ==================================================================================
// Copyright (c) 2017 HiFi-LoFi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ==================================================================================

#include "AudioFFTcpx.h"

#include <cassert>
#include <cmath>
#include <cstring>

#define AUDIOFFT_INTEL_IPP

#if defined(AUDIOFFT_INTEL_IPP)
#define AUDIOFFT_INTEL_IPP_USED
#include <ipp.h>
#elif defined(AUDIOFFT_APPLE_ACCELERATE)
#define AUDIOFFT_APPLE_ACCELERATE_USED
#include <Accelerate/Accelerate.h>
#include <vector>
#elif defined(AUDIOFFT_FFTW3)
#define AUDIOFFT_FFTW3_USED
#include <fftw3.h>
#else
#if !defined(AUDIOFFT_OOURA)
#define AUDIOFFT_OOURA
#endif
#define AUDIOFFT_OOURA_USED
#include <vector>
#endif

namespace audiofft {

namespace detail {

class AudioFFTcpxImpl {
public:
  AudioFFTcpxImpl() = default;
  AudioFFTcpxImpl(const AudioFFTcpxImpl &) = delete;
  AudioFFTcpxImpl &operator=(const AudioFFTcpxImpl &) = delete;
  virtual ~AudioFFTcpxImpl() = default;
  virtual void init(size_t size) = 0;
  virtual void fft(const float *in_re, const float *in_im, float *out_re,
                   float *out_im) = 0;
  virtual void ifft(float *out_re, float *out_im, const float *in_re,
                    const float *in_im) = 0;
};

constexpr bool IsPowerOf2(size_t val) {
  return (val == 1 || (val & (val - 1)) == 0);
}

template <typename TypeDest, typename TypeSrc>
void ConvertBuffer(TypeDest *dest, const TypeSrc *src, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    dest[i] = static_cast<TypeDest>(src[i]);
  }
}

template <typename TypeDest, typename TypeSrc, typename TypeFactor>
void ScaleBuffer(TypeDest *dest, const TypeSrc *src, const TypeFactor factor,
                 size_t len) {
  for (size_t i = 0; i < len; ++i) {
    dest[i] = static_cast<TypeDest>(static_cast<TypeFactor>(src[i]) * factor);
  }
}

} // End of namespace detail

// ================================================================

#ifdef AUDIOFFT_OOURA_USED

/**
 * @internal
 * @class OouraFFT
 * @brief FFT implementation based on the great radix-4 routines by Takuya Ooura
 */
class OouraFFT : public detail::AudioFFTcpxImpl {
public:
  OouraFFT() : detail::AudioFFTcpxImpl(), _size(0), _ip(), _w(), _buffer() {}

  OouraFFT(const OouraFFT &) = delete;
  OouraFFT &operator=(const OouraFFT &) = delete;

  virtual void init(size_t size) override {
    if (_size != size) {
      assert(detail::IsPowerOf2(size));
      _size = size;

      _ip.resize(2 + std::ceil(std::sqrt(size / 2.0f)));
      _w.resize(size / 2);
      _buffer.resize(size * 2);
      const size_t size4 = size / 2;
      makewt(size4, _ip.data(), _w.data());
    }
  }

  virtual void fft(const float *in_re, const float *in_im, float *out_re,
                   float *out_im) override {
    for (size_t i = 0; i < _size; ++i) {
      _buffer[2 * i] = in_re[i];
      _buffer[2 * i + 1] = in_im[i];
    }
    cdft(_size * 2, 1, _buffer.data(), _ip.data(), _w.data());
    out_re[0] = _buffer[0];
    out_im[0] = _buffer[1];
    for (size_t i = 1; i < _size; ++i) {
      out_re[_size - i] = _buffer[i * 2];
      out_im[_size - i] = _buffer[i * 2 + 1];
    }
  }

  virtual void ifft(float *out_re, float *out_im, const float *in_re,
                    const float *in_im) override {
    _buffer[0] = in_re[0];
    _buffer[1] = in_im[0];
    for (size_t i = 1; i < _size; ++i) {
      _buffer[2 * i] = in_re[_size - i];
      _buffer[2 * i + 1] = in_im[_size - i];
    }
    cdft(_size * 2, -1, _buffer.data(), _ip.data(), _w.data());
    const float gain = 1.0f / _size;
    for (size_t i = 0; i < _size; ++i) {
      out_re[i] = _buffer[i * 2] * gain;
      out_im[i] = _buffer[i * 2 + 1] * gain;
    }
  }

private:
  size_t _size;
  std::vector<int> _ip;
  std::vector<float> _w;
  std::vector<float> _buffer;

  static void cftmdl(int n, int l, float *a, float *w) noexcept {
    int j, j1, j2, j3, k, k1, k2, m, m2;
    float wk1r, wk1i, wk2r, wk2i, wk3r, wk3i;
    float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

    m = l << 2;
    for (j = 0; j < l; j += 2) {
      j1 = j + l;
      j2 = j1 + l;
      j3 = j2 + l;
      x0r = a[j] + a[j1];
      x0i = a[j + 1] + a[j1 + 1];
      x1r = a[j] - a[j1];
      x1i = a[j + 1] - a[j1 + 1];
      x2r = a[j2] + a[j3];
      x2i = a[j2 + 1] + a[j3 + 1];
      x3r = a[j2] - a[j3];
      x3i = a[j2 + 1] - a[j3 + 1];
      a[j] = x0r + x2r;
      a[j + 1] = x0i + x2i;
      a[j2] = x0r - x2r;
      a[j2 + 1] = x0i - x2i;
      a[j1] = x1r - x3i;
      a[j1 + 1] = x1i + x3r;
      a[j3] = x1r + x3i;
      a[j3 + 1] = x1i - x3r;
    }
    wk1r = w[2];
    for (j = m; j < l + m; j += 2) {
      j1 = j + l;
      j2 = j1 + l;
      j3 = j2 + l;
      x0r = a[j] + a[j1];
      x0i = a[j + 1] + a[j1 + 1];
      x1r = a[j] - a[j1];
      x1i = a[j + 1] - a[j1 + 1];
      x2r = a[j2] + a[j3];
      x2i = a[j2 + 1] + a[j3 + 1];
      x3r = a[j2] - a[j3];
      x3i = a[j2 + 1] - a[j3 + 1];
      a[j] = x0r + x2r;
      a[j + 1] = x0i + x2i;
      a[j2] = x2i - x0i;
      a[j2 + 1] = x0r - x2r;
      x0r = x1r - x3i;
      x0i = x1i + x3r;
      a[j1] = wk1r * (x0r - x0i);
      a[j1 + 1] = wk1r * (x0r + x0i);
      x0r = x3i + x1r;
      x0i = x3r - x1i;
      a[j3] = wk1r * (x0i - x0r);
      a[j3 + 1] = wk1r * (x0i + x0r);
    }
    k1 = 0;
    m2 = 2 * m;
    for (k = m2; k < n; k += m2) {
      k1 += 2;
      k2 = 2 * k1;
      wk2r = w[k1];
      wk2i = w[k1 + 1];
      wk1r = w[k2];
      wk1i = w[k2 + 1];
      wk3r = wk1r - 2 * wk2i * wk1i;
      wk3i = 2 * wk2i * wk1r - wk1i;
      for (j = k; j < l + k; j += 2) {
        j1 = j + l;
        j2 = j1 + l;
        j3 = j2 + l;
        x0r = a[j] + a[j1];
        x0i = a[j + 1] + a[j1 + 1];
        x1r = a[j] - a[j1];
        x1i = a[j + 1] - a[j1 + 1];
        x2r = a[j2] + a[j3];
        x2i = a[j2 + 1] + a[j3 + 1];
        x3r = a[j2] - a[j3];
        x3i = a[j2 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        x0r -= x2r;
        x0i -= x2i;
        a[j2] = wk2r * x0r - wk2i * x0i;
        a[j2 + 1] = wk2r * x0i + wk2i * x0r;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j1] = wk1r * x0r - wk1i * x0i;
        a[j1 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r - wk3i * x0i;
        a[j3 + 1] = wk3r * x0i + wk3i * x0r;
      }
      wk1r = w[k2 + 2];
      wk1i = w[k2 + 3];
      wk3r = wk1r - 2 * wk2r * wk1i;
      wk3i = 2 * wk2r * wk1r - wk1i;
      for (j = k + m; j < l + (k + m); j += 2) {
        j1 = j + l;
        j2 = j1 + l;
        j3 = j2 + l;
        x0r = a[j] + a[j1];
        x0i = a[j + 1] + a[j1 + 1];
        x1r = a[j] - a[j1];
        x1i = a[j + 1] - a[j1 + 1];
        x2r = a[j2] + a[j3];
        x2i = a[j2 + 1] + a[j3 + 1];
        x3r = a[j2] - a[j3];
        x3i = a[j2 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        x0r -= x2r;
        x0i -= x2i;
        a[j2] = -wk2i * x0r - wk2r * x0i;
        a[j2 + 1] = -wk2i * x0i + wk2r * x0r;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j1] = wk1r * x0r - wk1i * x0i;
        a[j1 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r - wk3i * x0i;
        a[j3 + 1] = wk3r * x0i + wk3i * x0r;
      }
    }
  }

  static void cft1st(int n, float *a, float *w) noexcept {
    int j, k1, k2;
    float wk1r, wk1i, wk2r, wk2i, wk3r, wk3i;
    float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

    x0r = a[0] + a[2];
    x0i = a[1] + a[3];
    x1r = a[0] - a[2];
    x1i = a[1] - a[3];
    x2r = a[4] + a[6];
    x2i = a[5] + a[7];
    x3r = a[4] - a[6];
    x3i = a[5] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[2] = x1r - x3i;
    a[3] = x1i + x3r;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;
    wk1r = w[2];
    x0r = a[8] + a[10];
    x0i = a[9] + a[11];
    x1r = a[8] - a[10];
    x1i = a[9] - a[11];
    x2r = a[12] + a[14];
    x2i = a[13] + a[15];
    x3r = a[12] - a[14];
    x3i = a[13] - a[15];
    a[8] = x0r + x2r;
    a[9] = x0i + x2i;
    a[12] = x2i - x0i;
    a[13] = x0r - x2r;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[10] = wk1r * (x0r - x0i);
    a[11] = wk1r * (x0r + x0i);
    x0r = x3i + x1r;
    x0i = x3r - x1i;
    a[14] = wk1r * (x0i - x0r);
    a[15] = wk1r * (x0i + x0r);
    k1 = 0;
    for (j = 16; j < n; j += 16) {
      k1 += 2;
      k2 = 2 * k1;
      wk2r = w[k1];
      wk2i = w[k1 + 1];
      wk1r = w[k2];
      wk1i = w[k2 + 1];
      wk3r = wk1r - 2 * wk2i * wk1i;
      wk3i = 2 * wk2i * wk1r - wk1i;
      x0r = a[j] + a[j + 2];
      x0i = a[j + 1] + a[j + 3];
      x1r = a[j] - a[j + 2];
      x1i = a[j + 1] - a[j + 3];
      x2r = a[j + 4] + a[j + 6];
      x2i = a[j + 5] + a[j + 7];
      x3r = a[j + 4] - a[j + 6];
      x3i = a[j + 5] - a[j + 7];
      a[j] = x0r + x2r;
      a[j + 1] = x0i + x2i;
      x0r -= x2r;
      x0i -= x2i;
      a[j + 4] = wk2r * x0r - wk2i * x0i;
      a[j + 5] = wk2r * x0i + wk2i * x0r;
      x0r = x1r - x3i;
      x0i = x1i + x3r;
      a[j + 2] = wk1r * x0r - wk1i * x0i;
      a[j + 3] = wk1r * x0i + wk1i * x0r;
      x0r = x1r + x3i;
      x0i = x1i - x3r;
      a[j + 6] = wk3r * x0r - wk3i * x0i;
      a[j + 7] = wk3r * x0i + wk3i * x0r;
      wk1r = w[k2 + 2];
      wk1i = w[k2 + 3];
      wk3r = wk1r - 2 * wk2r * wk1i;
      wk3i = 2 * wk2r * wk1r - wk1i;
      x0r = a[j + 8] + a[j + 10];
      x0i = a[j + 9] + a[j + 11];
      x1r = a[j + 8] - a[j + 10];
      x1i = a[j + 9] - a[j + 11];
      x2r = a[j + 12] + a[j + 14];
      x2i = a[j + 13] + a[j + 15];
      x3r = a[j + 12] - a[j + 14];
      x3i = a[j + 13] - a[j + 15];
      a[j + 8] = x0r + x2r;
      a[j + 9] = x0i + x2i;
      x0r -= x2r;
      x0i -= x2i;
      a[j + 12] = -wk2i * x0r - wk2r * x0i;
      a[j + 13] = -wk2i * x0i + wk2r * x0r;
      x0r = x1r - x3i;
      x0i = x1i + x3r;
      a[j + 10] = wk1r * x0r - wk1i * x0i;
      a[j + 11] = wk1r * x0i + wk1i * x0r;
      x0r = x1r + x3i;
      x0i = x1i - x3r;
      a[j + 14] = wk3r * x0r - wk3i * x0i;
      a[j + 15] = wk3r * x0i + wk3i * x0r;
    }
  }

  static void cftbsub(int n, float *a, float *w) noexcept {
    int j, j1, j2, j3, l;
    float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

    l = 2;
    if (n > 8) {
      cft1st(n, a, w);
      l = 8;
      while ((l << 2) < n) {
        cftmdl(n, l, a, w);
        l <<= 2;
      }
    }
    if ((l << 2) == n) {
      for (j = 0; j < l; j += 2) {
        j1 = j + l;
        j2 = j1 + l;
        j3 = j2 + l;
        x0r = a[j] + a[j1];
        x0i = -a[j + 1] - a[j1 + 1];
        x1r = a[j] - a[j1];
        x1i = -a[j + 1] + a[j1 + 1];
        x2r = a[j2] + a[j3];
        x2i = a[j2 + 1] + a[j3 + 1];
        x3r = a[j2] - a[j3];
        x3i = a[j2 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i - x2i;
        a[j2] = x0r - x2r;
        a[j2 + 1] = x0i + x2i;
        a[j1] = x1r - x3i;
        a[j1 + 1] = x1i - x3r;
        a[j3] = x1r + x3i;
        a[j3 + 1] = x1i + x3r;
      }
    } else {
      for (j = 0; j < l; j += 2) {
        j1 = j + l;
        x0r = a[j] - a[j1];
        x0i = -a[j + 1] + a[j1 + 1];
        a[j] += a[j1];
        a[j + 1] = -a[j + 1] - a[j1 + 1];
        a[j1] = x0r;
        a[j1 + 1] = x0i;
      }
    }
  }

  static void rftbsub(int n, float *a, int nc, float *c) noexcept {
    int j, k, kk, ks, m;
    float wkr, wki, xr, xi, yr, yi;

    a[1] = -a[1];
    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    for (j = 2; j < m; j += 2) {
      k = n - j;
      kk += ks;
      wkr = 0.5 - c[nc - kk];
      wki = c[kk];
      xr = a[j] - a[k];
      xi = a[j + 1] + a[k + 1];
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;
      a[j] -= yr;
      a[j + 1] = yi - a[j + 1];
      a[k] += yr;
      a[k + 1] = yi - a[k + 1];
    }
    a[m + 1] = -a[m + 1];
  }

  static void rftfsub(int n, float *a, int nc, float *c) noexcept {
    int j, k, kk, ks, m;
    float wkr, wki, xr, xi, yr, yi;

    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    for (j = 2; j < m; j += 2) {
      k = n - j;
      kk += ks;
      wkr = 0.5 - c[nc - kk];
      wki = c[kk];
      xr = a[j] - a[k];
      xi = a[j + 1] + a[k + 1];
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      a[j] -= yr;
      a[j + 1] -= yi;
      a[k] += yr;
      a[k + 1] -= yi;
    }
  }

  static void cdft(int n, int isgn, float *a, int *ip, float *w) noexcept {
    if (n > 4) {
      if (isgn >= 0) {
        bitrv2(n, ip + 2, a);
        cftfsub(n, a, w);
      } else {
        bitrv2conj(n, ip + 2, a);
        cftbsub(n, a, w);
      }
    } else if (n == 4) {
      cftfsub(n, a, w);
    }
  }

  static void bitrv2conj(int n, int *ip, float *a) noexcept {
    int j, j1, k, k1, l, m, m2;
    float xr, xi, yr, yi;

    ip[0] = 0;
    l = n;
    m = 1;
    while ((m << 3) < l) {
      l >>= 1;
      for (j = 0; j < m; j++) {
        ip[m + j] = ip[j] + l;
      }
      m <<= 1;
    }
    m2 = 2 * m;
    if ((m << 3) == l) {
      for (k = 0; k < m; k++) {
        for (j = 0; j < k; j++) {
          j1 = 2 * j + ip[k];
          k1 = 2 * k + ip[j];
          xr = a[j1];
          xi = -a[j1 + 1];
          yr = a[k1];
          yi = -a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 += 2 * m2;
          xr = a[j1];
          xi = -a[j1 + 1];
          yr = a[k1];
          yi = -a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 -= m2;
          xr = a[j1];
          xi = -a[j1 + 1];
          yr = a[k1];
          yi = -a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 += 2 * m2;
          xr = a[j1];
          xi = -a[j1 + 1];
          yr = a[k1];
          yi = -a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
        }
        k1 = 2 * k + ip[k];
        a[k1 + 1] = -a[k1 + 1];
        j1 = k1 + m2;
        k1 = j1 + m2;
        xr = a[j1];
        xi = -a[j1 + 1];
        yr = a[k1];
        yi = -a[k1 + 1];
        a[j1] = yr;
        a[j1 + 1] = yi;
        a[k1] = xr;
        a[k1 + 1] = xi;
        k1 += m2;
        a[k1 + 1] = -a[k1 + 1];
      }
    } else {
      a[1] = -a[1];
      a[m2 + 1] = -a[m2 + 1];
      for (k = 1; k < m; k++) {
        for (j = 0; j < k; j++) {
          j1 = 2 * j + ip[k];
          k1 = 2 * k + ip[j];
          xr = a[j1];
          xi = -a[j1 + 1];
          yr = a[k1];
          yi = -a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 += m2;
          xr = a[j1];
          xi = -a[j1 + 1];
          yr = a[k1];
          yi = -a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
        }
        k1 = 2 * k + ip[k];
        a[k1 + 1] = -a[k1 + 1];
        a[k1 + m2 + 1] = -a[k1 + m2 + 1];
      }
    }
  }

  static void cftfsub(int n, float *a, float *w) noexcept {
    int j, j1, j2, j3, l;
    float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

    l = 2;
    if (n > 8) {
      cft1st(n, a, w);
      l = 8;
      while ((l << 2) < n) {
        cftmdl(n, l, a, w);
        l <<= 2;
      }
    }
    if ((l << 2) == n) {
      for (j = 0; j < l; j += 2) {
        j1 = j + l;
        j2 = j1 + l;
        j3 = j2 + l;
        x0r = a[j] + a[j1];
        x0i = a[j + 1] + a[j1 + 1];
        x1r = a[j] - a[j1];
        x1i = a[j + 1] - a[j1 + 1];
        x2r = a[j2] + a[j3];
        x2i = a[j2 + 1] + a[j3 + 1];
        x3r = a[j2] - a[j3];
        x3i = a[j2 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j2] = x0r - x2r;
        a[j2 + 1] = x0i - x2i;
        a[j1] = x1r - x3i;
        a[j1 + 1] = x1i + x3r;
        a[j3] = x1r + x3i;
        a[j3 + 1] = x1i - x3r;
      }
    } else {
      for (j = 0; j < l; j += 2) {
        j1 = j + l;
        x0r = a[j] - a[j1];
        x0i = a[j + 1] - a[j1 + 1];
        a[j] += a[j1];
        a[j + 1] += a[j1 + 1];
        a[j1] = x0r;
        a[j1 + 1] = x0i;
      }
    }
  }

  static void makect(int nc, int *ip, float *c) noexcept {
    int j, nch;
    float delta;

    ip[1] = nc;
    if (nc > 1) {
      nch = nc >> 1;
      delta = atan(1.0) / nch;
      c[0] = cos(delta * nch);
      c[nch] = 0.5 * c[0];
      for (j = 1; j < nch; j++) {
        c[j] = 0.5 * cos(delta * j);
        c[nc - j] = 0.5 * sin(delta * j);
      }
    }
  }

  static void makewt(int nw, int *ip, float *w) noexcept {
    int j, nwh;
    float delta, x, y;

    ip[0] = nw;
    ip[1] = 1;
    if (nw > 2) {
      nwh = nw >> 1;
      delta = atan(1.0) / nwh;
      w[0] = 1;
      w[1] = 0;
      w[nwh] = cos(delta * nwh);
      w[nwh + 1] = w[nwh];
      if (nwh > 2) {
        for (j = 2; j < nwh; j += 2) {
          x = cos(delta * j);
          y = sin(delta * j);
          w[j] = x;
          w[j + 1] = y;
          w[nw - j] = y;
          w[nw - j + 1] = x;
        }
        bitrv2(nw, ip + 2, w);
      }
    }
  }

  static void bitrv2(int n, int *ip, float *a) noexcept {
    int j, j1, k, k1, l, m, m2;
    float xr, xi, yr, yi;

    ip[0] = 0;
    l = n;
    m = 1;
    while ((m << 3) < l) {
      l >>= 1;
      for (j = 0; j < m; j++) {
        ip[m + j] = ip[j] + l;
      }
      m <<= 1;
    }
    m2 = 2 * m;
    if ((m << 3) == l) {
      for (k = 0; k < m; k++) {
        for (j = 0; j < k; j++) {
          j1 = 2 * j + ip[k];
          k1 = 2 * k + ip[j];
          xr = a[j1];
          xi = a[j1 + 1];
          yr = a[k1];
          yi = a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 += 2 * m2;
          xr = a[j1];
          xi = a[j1 + 1];
          yr = a[k1];
          yi = a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 -= m2;
          xr = a[j1];
          xi = a[j1 + 1];
          yr = a[k1];
          yi = a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 += 2 * m2;
          xr = a[j1];
          xi = a[j1 + 1];
          yr = a[k1];
          yi = a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
        }
        j1 = 2 * k + m2 + ip[k];
        k1 = j1 + m2;
        xr = a[j1];
        xi = a[j1 + 1];
        yr = a[k1];
        yi = a[k1 + 1];
        a[j1] = yr;
        a[j1 + 1] = yi;
        a[k1] = xr;
        a[k1 + 1] = xi;
      }
    } else {
      for (k = 1; k < m; k++) {
        for (j = 0; j < k; j++) {
          j1 = 2 * j + ip[k];
          k1 = 2 * k + ip[j];
          xr = a[j1];
          xi = a[j1 + 1];
          yr = a[k1];
          yi = a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
          j1 += m2;
          k1 += m2;
          xr = a[j1];
          xi = a[j1 + 1];
          yr = a[k1];
          yi = a[k1 + 1];
          a[j1] = yr;
          a[j1 + 1] = yi;
          a[k1] = xr;
          a[k1 + 1] = xi;
        }
      }
    }
  }
};

/**
 * @internal
 * @brief Concrete FFT implementation
 */
typedef OouraFFT AudioFFTImplementation;

#endif // AUDIOFFT_OOURA_USED

// ================================================================

#ifdef AUDIOFFT_INTEL_IPP_USED

/**
 * @internal
 * @class IntelIppFFT
 * @brief FFT implementation using the Intel Integrated Performance Primitives
 */
class IntelIppFFT : public detail::AudioFFTcpxImpl {
public:
  IntelIppFFT() : detail::AudioFFTcpxImpl() {}

  IntelIppFFT(const IntelIppFFT &) = delete;
  IntelIppFFT &operator=(const IntelIppFFT &) = delete;

  virtual ~IntelIppFFT() {
    if (p_spec_) {
      ippsFree(p_spec_);
      p_spec_ = nullptr;
    }
    if (p_buffer_) {
      ippsFree(p_buffer_);
      p_buffer_ = nullptr;
    }
  }

  virtual void init(size_t fft_size) override {
    fft_size_ = fft_size;

    int order = std::countr_zero(fft_size);
    int p_spec_size{};
    int p_spec_buffer_size{};
    int p_buffer_size{};
    int const flag = IPP_FFT_DIV_INV_BY_N;
    ippsFFTGetSize_C_32f(order, flag, ippAlgHintFast, &p_spec_size,
                         &p_spec_buffer_size, &p_buffer_size);

    if (p_spec_) {
      ippsFree(p_spec_);
      p_spec_ = nullptr;
    }
    p_spec_ = ippsMalloc_8u(p_spec_size);
    if (p_spec_buffer_) {
      ippsFree(p_spec_buffer_);
      p_spec_buffer_ = nullptr;
    }
    p_spec_buffer_ =
        p_spec_buffer_size > 0 ? ippsMalloc_8u(p_spec_buffer_size) : nullptr;
    if (p_buffer_) {
      ippsFree(p_buffer_);
      p_buffer_ = nullptr;
    }
    p_buffer_ = p_buffer_size > 0 ? ippsMalloc_8u(p_buffer_size) : nullptr;

    ippsFFTInit_C_32f(&p_fft_spec_, order, flag, ippAlgHintFast, p_spec_,
                      p_spec_buffer_);
    if (p_spec_buffer_) {
      ippsFree(p_spec_buffer_);
      p_spec_buffer_ = nullptr;
    }
  }

  virtual void fft(const float *in_re, const float *in_im, float *out_re,
                   float *out_im) override {
    ippsFFTFwd_CToC_32f(in_re, in_im, out_re, out_im, p_fft_spec_, p_buffer_);
  }

  virtual void ifft(float *out_re, float *out_im, const float *in_re,
                    const float *in_im) override {
    ippsFFTInv_CToC_32f(in_re, in_im, out_re, out_im, p_fft_spec_, p_buffer_);
  }

private:
  size_t fft_size_{};
  Ipp8u *p_spec_{};
  Ipp8u *p_spec_buffer_{};
  Ipp8u *p_buffer_{};
  IppsFFTSpec_C_32f *p_fft_spec_{};
};

/**
 * @internal
 * @brief Concrete FFT implementation
 */
typedef IntelIppFFT AudioFFTImplementation;

#endif // AUDIOFFT_INTEL_IPP_USED

// ================================================================

#ifdef AUDIOFFT_APPLE_ACCELERATE_USED

#error "not implemened"

/**
 * @internal
 * @class AppleAccelerateFFT
 * @brief FFT implementation using the Apple Accelerate framework internally
 */
class AppleAccelerateFFT : public detail::AudioFFTImpl {
public:
  AppleAccelerateFFT()
      : detail::AudioFFTImpl(), _size(0), _powerOf2(0), _fftSetup(0), _re(),
        _im() {}

  AppleAccelerateFFT(const AppleAccelerateFFT &) = delete;
  AppleAccelerateFFT &operator=(const AppleAccelerateFFT &) = delete;

  virtual ~AppleAccelerateFFT() { init(0); }

  virtual void init(size_t size) override {
    if (_fftSetup) {
      vDSP_destroy_fftsetup(_fftSetup);
      _size = 0;
      _powerOf2 = 0;
      _fftSetup = 0;
      _re.clear();
      _im.clear();
    }

    if (size > 0) {
      _size = size;
      _powerOf2 = 0;
      while ((1 << _powerOf2) < _size) {
        ++_powerOf2;
      }
      _fftSetup = vDSP_create_fftsetup(_powerOf2, FFT_RADIX2);
      _re.resize(_size / 2);
      _im.resize(_size / 2);
    }
  }

  virtual void fft(const float *data, float *re, float *im) override {
    const size_t size2 = _size / 2;
    DSPSplitComplex splitComplex;
    splitComplex.realp = re;
    splitComplex.imagp = im;
    vDSP_ctoz(reinterpret_cast<const COMPLEX *>(data), 2, &splitComplex, 1,
              size2);
    vDSP_fft_zrip(_fftSetup, &splitComplex, 1, _powerOf2, FFT_FORWARD);
    const float factor = 0.5f;
    vDSP_vsmul(re, 1, &factor, re, 1, size2);
    vDSP_vsmul(im, 1, &factor, im, 1, size2);
    re[size2] = im[0];
    im[0] = 0.0f;
    im[size2] = 0.0f;
  }

  virtual void ifft(float *data, const float *re, const float *im) override {
    const size_t size2 = _size / 2;
    ::memcpy(_re.data(), re, size2 * sizeof(float));
    ::memcpy(_im.data(), im, size2 * sizeof(float));
    _im[0] = re[size2];
    DSPSplitComplex splitComplex;
    splitComplex.realp = _re.data();
    splitComplex.imagp = _im.data();
    vDSP_fft_zrip(_fftSetup, &splitComplex, 1, _powerOf2, FFT_INVERSE);
    vDSP_ztoc(&splitComplex, 1, reinterpret_cast<COMPLEX *>(data), 2, size2);
    const float factor = 1.0f / static_cast<float>(_size);
    vDSP_vsmul(data, 1, &factor, data, 1, _size);
  }

private:
  size_t _size;
  size_t _powerOf2;
  FFTSetup _fftSetup;
  std::vector<float> _re;
  std::vector<float> _im;
};

/**
 * @internal
 * @brief Concrete FFT implementation
 */
typedef AppleAccelerateFFT AudioFFTImplementation;

#endif // AUDIOFFT_APPLE_ACCELERATE_USED

// ================================================================

#ifdef AUDIOFFT_FFTW3_USED

#error "not implemened"

/**
 * @internal
 * @class FFTW3FFT
 * @brief FFT implementation using FFTW3 internally (see fftw.org)
 */
class FFTW3FFT : public detail::AudioFFTImpl {
public:
  FFTW3FFT()
      : detail::AudioFFTImpl(), _size(0), _complexSize(0), _planForward(0),
        _planBackward(0), _data(0), _re(0), _im(0) {}

  FFTW3FFT(const FFTW3FFT &) = delete;
  FFTW3FFT &operator=(const FFTW3FFT &) = delete;

  virtual ~FFTW3FFT() { init(0); }

  virtual void init(size_t size) override {
    if (_size != size) {
      if (_size > 0) {
        fftwf_destroy_plan(_planForward);
        fftwf_destroy_plan(_planBackward);
        _planForward = 0;
        _planBackward = 0;
        _size = 0;
        _complexSize = 0;

        if (_data) {
          fftwf_free(_data);
          _data = 0;
        }

        if (_re) {
          fftwf_free(_re);
          _re = 0;
        }

        if (_im) {
          fftwf_free(_im);
          _im = 0;
        }
      }

      if (size > 0) {
        _size = size;
        _complexSize = AudioFFT::ComplexSize(_size);
        const size_t complexSize = AudioFFT::ComplexSize(_size);
        _data = reinterpret_cast<float *>(fftwf_malloc(_size * sizeof(float)));
        _re = reinterpret_cast<float *>(
            fftwf_malloc(complexSize * sizeof(float)));
        _im = reinterpret_cast<float *>(
            fftwf_malloc(complexSize * sizeof(float)));

        fftw_iodim dim;
        dim.n = static_cast<int>(size);
        dim.is = 1;
        dim.os = 1;
        _planForward = fftwf_plan_guru_split_dft_r2c(1, &dim, 0, 0, _data, _re,
                                                     _im, FFTW_MEASURE);
        _planBackward = fftwf_plan_guru_split_dft_c2r(1, &dim, 0, 0, _re, _im,
                                                      _data, FFTW_MEASURE);
      }
    }
  }

  virtual void fft(const float *data, float *re, float *im) override {
    ::memcpy(_data, data, _size * sizeof(float));
    fftwf_execute_split_dft_r2c(_planForward, _data, _re, _im);
    ::memcpy(re, _re, _complexSize * sizeof(float));
    ::memcpy(im, _im, _complexSize * sizeof(float));
  }

  virtual void ifft(float *data, const float *re, const float *im) override {
    ::memcpy(_re, re, _complexSize * sizeof(float));
    ::memcpy(_im, im, _complexSize * sizeof(float));
    fftwf_execute_split_dft_c2r(_planBackward, _re, _im, _data);
    detail::ScaleBuffer(data, _data, 1.0f / static_cast<float>(_size), _size);
  }

private:
  size_t _size;
  size_t _complexSize;
  fftwf_plan _planForward;
  fftwf_plan _planBackward;
  float *_data;
  float *_re;
  float *_im;
};

/**
 * @internal
 * @brief Concrete FFT implementation
 */
typedef FFTW3FFT AudioFFTImplementation;

#endif // AUDIOFFT_FFTW3_USED

// =============================================================

AudioFFTcpx::AudioFFTcpx() : _impl(new AudioFFTImplementation()) {}

AudioFFTcpx::~AudioFFTcpx() {}

void AudioFFTcpx::init(size_t size) {
  assert(detail::IsPowerOf2(size));
  _impl->init(size);
}

void AudioFFTcpx::fft(const float *in_re, const float *in_im, float *out_re,
                      float *out_im) {
  _impl->fft(in_re, in_im, out_re, out_im);
}

void AudioFFTcpx::ifft(float *out_re, float *out_im, const float *in_re,
                       const float *in_im) {
  _impl->ifft(out_re, out_im, in_re, in_im);
}

} // namespace audiofft
