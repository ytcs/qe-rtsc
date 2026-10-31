// rtsc: pruned fused local operator on cuFFTDx (groups/compute/FFT-LOG.md). RTSC_STH_FFTDX=1.
//
// Same mathematics and the same separable, pruned structure as rtsc/sth/sth_pfft.f90 (pfft), with the 1D transforms
// replaced by size-specialised, register-resident cuFFTDx thread FFTs (one thread per line):
//   K1 kz_inv  one thread per (z-stick, column): gather the stick's sphere coefficients (0 elsewhere), inverse z-FFT
//              in registers, write the stick array S(stick + nst*z, j);
//   K2 kxy     one block per (z-plane, column): scatter the plane's sticks into a shared (n1 x n2) plane, inverse
//              y-FFT on the occupied x-columns only, then per row: inverse x-FFT, *V(r), forward x-FFT (fused in
//              registers), forward y-FFT on the occupied x-columns, write the sticks back to S;
//   K3 kz_fwd  one thread per (z-stick, column): forward z-FFT and the k_gather_h formula on the sphere.
// For a column x on the k+q sphere this gives y = g2kin x + scale * FFT_fwd[V FFT_inv[x]] - e x (rows > npwq zeroed),
// the local part of the engine's apply_op. FP64 (scored path) and FP32 (the MP inner solve) instances.
// FP64 size 54 is not a native cuFFTDx size in MathDx 25.6: it is built as 2 x 27 (two native thread FFTs and one
// radix-2 combine with __constant__ twiddles).
//
// Grid sizes are compile-time (FDX_GRIDS below); other grids return "unsupported" and the caller keeps pfft/cuFFT.
// Host entry points (extern "C", callable from Fortran BIND(C)):
//   fdx_setup(nr, nrx, nksq, npwx, npwq, idxkq, chunk) -> 0 ok, > 0 unsupported
//   fdx_local_d / fdx_local_s (device pointers, Fortran 1-based list/idx contents), fdx_free, fdx_info
#include <cufftdx.hpp>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef FDX_SM
#define FDX_SM 860
#endif

namespace fdx {

template <class T> struct alignas(2 * sizeof(T)) cplx { T x, y; };

template <class T> __device__ __forceinline__ cplx<T> cmul(cplx<T> a, cplx<T> b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// ---------------------------------------------------------------------------------------------------------------
// 1D line FFT in registers, natural order in and out, unnormalised. INV = exp(+i...) (cuFFT inverse).
// ---------------------------------------------------------------------------------------------------------------
#ifdef FDX_EMU
// Harness-only precision emulation of the FP32 instance (FFT-LOG scope items 2 and 3): d_emu = 1: each 1D transform
// is a dense DFT with TF32-rounded operands (inputs and DFT matrix) and FP32 accumulation, as a TF32 tensor-core GEMM
// would do it; 2 / 3: native FFT, data rounded to FP16 / BF16 at every transform input and output (storage in half
// precision with ideal scaling; FP32 arithmetic, so a lower bound on the error of a real half-precision FFT).
__device__ int d_emu = 0;
__constant__ double c_twE[540][2];
__device__ __forceinline__ float tf32r(float f) {
  unsigned u = __float_as_uint(f); u += 0x1000u; u &= 0xFFFFE000u; return __uint_as_float(u);
}
__device__ __forceinline__ float rnd_emu(float f, int m) {
  if (m == 2) return __half2float(__float2half_rn(f));
  if (m == 3) return __bfloat162float(__float2bfloat16_rn(f));
  return f;
}
#endif
template <class T, int N, bool INV> struct Line {
  using FFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<T>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                       cufftdx::Direction<INV ? cufftdx::fft_direction::inverse : cufftdx::fft_direction::forward>() +
                       cufftdx::SM<FDX_SM>() + cufftdx::Thread());
  using V = typename FFT::value_type;
  static_assert(FFT::storage_size == N, "thread FFT storage must equal N");
  __device__ __forceinline__ static void run(cplx<T> (&v)[N]) {
#ifdef FDX_EMU
    if constexpr (sizeof(T) == 4) {
      const int m = d_emu;
      if (m == 1) {
        static_assert(540 % N == 0, "emulation table");
        cplx<T> in[N];
#pragma unroll
        for (int j = 0; j < N; j++) in[j] = {tf32r(v[j].x), tf32r(v[j].y)};
        for (int k = 0; k < N; k++) {
          float ax = 0, ay = 0;
#pragma unroll
          for (int j = 0; j < N; j++) {
            int e = (j * k) % N;
            float wx = tf32r((float)c_twE[e * (540 / N)][0]), wy = tf32r((float)c_twE[e * (540 / N)][1]);
            if (!INV) wy = -wy;
            ax += in[j].x * wx - in[j].y * wy; ay += in[j].x * wy + in[j].y * wx;
          }
          v[k] = {ax, ay};   // dynamic index: local memory, acceptable for an accuracy-only mode
        }
        return;
      }
      if (m >= 2) {
#pragma unroll
        for (int j = 0; j < N; j++) v[j] = {rnd_emu(v[j].x, m), rnd_emu(v[j].y, m)};
        FFT().execute(reinterpret_cast<V*>(&v[0]));
#pragma unroll
        for (int j = 0; j < N; j++) v[j] = {rnd_emu(v[j].x, m), rnd_emu(v[j].y, m)};
        return;
      }
    }
#endif
    V* p = reinterpret_cast<V*>(&v[0]);
    FFT().execute(p);
  }
};

__constant__ double c_tw54[27][2];   // exp(+2 pi i k / 54), k = 0..26

template <bool INV> struct Line<double, 54, INV> {
  __device__ __forceinline__ static void run(cplx<double> (&v)[54]) {
    cplx<double> e[27], o[27];
#pragma unroll
    for (int i = 0; i < 27; i++) { e[i] = v[2 * i]; o[i] = v[2 * i + 1]; }
    Line<double, 27, INV>::run(e);
    Line<double, 27, INV>::run(o);
#pragma unroll
    for (int k = 0; k < 27; k++) {
      cplx<double> w = {c_tw54[k][0], INV ? c_tw54[k][1] : -c_tw54[k][1]};
      cplx<double> t = cmul(o[k], w);
      v[k] = {e[k].x + t.x, e[k].y + t.y};
      v[k + 27] = {e[k].x - t.x, e[k].y - t.y};
    }
  }
};

// ---------------------------------------------------------------------------------------------------------------
// geometry (device), set by fdx_setup
// ---------------------------------------------------------------------------------------------------------------
struct Geo {
  int n1 = 0, n2 = 0, n3 = 0, nst = 0, nxa = 0, nzst = 0, nksq = 0, chunk = 0, gid = -1, split = 0;
  int *zmap = nullptr, *sx = nullptr, *sy = nullptr, *xa = nullptr;
  void *S = nullptr; size_t Sbytes = 0;
};
static Geo g;

// ---------------------------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------------------------
constexpr int TPB_Z = 128;

template <class T, int N3>
__global__ void __launch_bounds__(TPB_Z) kz_inv(cplx<T>* __restrict__ S, const cplx<T>* __restrict__ x,
                                                const int* __restrict__ list, int s0, int nslot, int npert, int npwx,
                                                int nst, const int* __restrict__ zmap, int nzst) {
  const int j = blockIdx.y;
  const int s = blockIdx.x * TPB_Z + threadIdx.x;
  if (s >= nst) return;
  const int col = list[s0 + j] - 1;
  const int k = col / (nslot * npert);
  const int* zm = zmap + (size_t)nzst * k + s;
  const cplx<T>* xc = x + (size_t)npwx * col;
  cplx<T> v[N3];
#pragma unroll
  for (int z = 0; z < N3; z++) {
    int ig = __ldg(zm + (size_t)nst * z);
    v[z] = ig > 0 ? xc[ig - 1] : cplx<T>{0, 0};
  }
  Line<T, N3, true>::run(v);
  cplx<T>* Sj = S + (size_t)nzst * j + s;
#pragma unroll
  for (int z = 0; z < N3; z++) Sj[(size_t)nst * z] = v[z];
}

template <class T, int N1, int N2>
__global__ void kxy(cplx<T>* __restrict__ S, const T* __restrict__ vr, int nst, int nzst, const int* __restrict__ sx,
                    const int* __restrict__ sy, const int* __restrict__ xa, int nxa) {
  constexpr int LD = (N1 % 2 == 0) ? N1 + 1 : N1;
  extern __shared__ __align__(16) unsigned char smraw[];
  cplx<T>* P = reinterpret_cast<cplx<T>*>(smraw);
  const int z = blockIdx.x, j = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
  cplx<T>* Sj = S + (size_t)nzst * j + (size_t)nst * z;
  for (int p = tid; p < LD * N2; p += nt) P[p] = cplx<T>{0, 0};
  __syncthreads();
  for (int p = tid; p < nst; p += nt) P[sy[p] * LD + sx[p]] = Sj[p];
  __syncthreads();
  for (int c = tid; c < nxa; c += nt) {           // inverse y on occupied x-columns
    const int xx = xa[c];
    cplx<T> v[N2];
#pragma unroll
    for (int y = 0; y < N2; y++) v[y] = P[y * LD + xx];
    Line<T, N2, true>::run(v);
#pragma unroll
    for (int y = 0; y < N2; y++) P[y * LD + xx] = v[y];
  }
  __syncthreads();
  const T* vz = vr + (size_t)N1 * N2 * z;
  for (int r = tid; r < N2; r += nt) {            // inverse x, *V, forward x
    cplx<T> v[N1];
#pragma unroll
    for (int xx = 0; xx < N1; xx++) v[xx] = P[r * LD + xx];
    Line<T, N1, true>::run(v);
#pragma unroll
    for (int xx = 0; xx < N1; xx++) {
      T w = __ldg(vz + r * N1 + xx);
      v[xx].x *= w; v[xx].y *= w;
    }
    Line<T, N1, false>::run(v);
#pragma unroll
    for (int xx = 0; xx < N1; xx++) P[r * LD + xx] = v[xx];
  }
  __syncthreads();
  for (int c = tid; c < nxa; c += nt) {           // forward y on occupied x-columns
    const int xx = xa[c];
    cplx<T> v[N2];
#pragma unroll
    for (int y = 0; y < N2; y++) v[y] = P[y * LD + xx];
    Line<T, N2, false>::run(v);
#pragma unroll
    for (int y = 0; y < N2; y++) P[y * LD + xx] = v[y];
  }
  __syncthreads();
  for (int p = tid; p < nst; p += nt) Sj[p] = P[sy[p] * LD + sx[p]];
}

template <class T, class TG, int N3>
__global__ void __launch_bounds__(TPB_Z) kz_fwd(cplx<T>* __restrict__ y, const cplx<T>* __restrict__ x,
                                                const cplx<T>* __restrict__ S, const int* __restrict__ list, int s0,
                                                int nslot, int npert, int npwx, int nbnd, int nst,
                                                const int* __restrict__ zmap, int nzst, const int* __restrict__ npwq,
                                                const TG* __restrict__ g2, const double* __restrict__ etk, T scale) {
  const int j = blockIdx.y;
  const int s = blockIdx.x * TPB_Z + threadIdx.x;
  const int col = list[s0 + j] - 1;
  const int k = col / (nslot * npert);
  const int n = col % nslot;
  if (blockIdx.x == 0) {                           // rows > npwq(k) zeroed
    for (int ig = npwq[k] + threadIdx.x; ig < npwx; ig += TPB_Z) y[(size_t)npwx * col + ig] = cplx<T>{0, 0};
  }
  if (s >= nst) return;
  const cplx<T>* Sj = S + (size_t)nzst * j + s;
  cplx<T> v[N3];
#pragma unroll
  for (int z = 0; z < N3; z++) v[z] = Sj[(size_t)nst * z];
  Line<T, N3, false>::run(v);
  const int* zm = zmap + (size_t)nzst * k + s;
  const cplx<T>* xc = x + (size_t)npwx * col;
  cplx<T>* yc = y + (size_t)npwx * col;
  const TG* g2k = g2 + (size_t)npwx * k;
  const T e = (T)etk[(size_t)nbnd * k + n];
#pragma unroll
  for (int z = 0; z < N3; z++) {
    int ig = __ldg(zm + (size_t)nst * z);
    if (ig > 0) {
      cplx<T> xv = xc[ig - 1];
      T gg = (T)g2k[ig - 1];
      yc[ig - 1] = {gg * xv.x + v[z].x * scale - e * xv.x, gg * xv.y + v[z].y * scale - e * xv.y};
    }
  }
}


// ---------------------------------------------------------------------------------------------------------------
// split-line plane kernel: two threads per line (lanes 2q, 2q+1), N = 2 x N/2 with native N/2-point thread FFTs.
//   inverse: radix-2 decimation in time   (thread h loads samples 2i+h, ends holding outputs i + h N/2);
//   forward: radix-2 decimation in frequency (thread h loads samples i + h N/2, ends holding outputs 2k + h).
// The pair exchanges N/2 values once per transform by shuffle. Twice the threads, half the registers per thread.
// ---------------------------------------------------------------------------------------------------------------
constexpr int TWL = 540;                  // lcm of the half-split sizes' parents (30, 36, 54)
__constant__ double c_twL[TWL][2];        // exp(+2 pi i m / TWL)

template <class T, int N> __device__ __forceinline__ cplx<T> twp(int m) {   // exp(+2 pi i m / N)
  static_assert(TWL % N == 0, "add N to TWL");
  return {(T)c_twL[m * (TWL / N)][0], (T)c_twL[m * (TWL / N)][1]};
}
template <class T> __device__ __forceinline__ cplx<T> shx(cplx<T> a, unsigned mask) {
  return {__shfl_xor_sync(mask, a.x, 1), __shfl_xor_sync(mask, a.y, 1)};
}
template <class T, int N>
__device__ __forceinline__ void inv_dit2(cplx<T> (&v)[N / 2], int h, unsigned mask) {
  Line<T, N / 2, true>::run(v);
#pragma unroll
  for (int i = 0; i < N / 2; i++) {
    cplx<T> o = shx(v[i], mask);
    cplx<T> E = h ? o : v[i], O = h ? v[i] : o;
    cplx<T> t = cmul(O, twp<T, N>(i));
    v[i] = h ? cplx<T>{E.x - t.x, E.y - t.y} : cplx<T>{E.x + t.x, E.y + t.y};
  }
}
template <class T, int N>
__device__ __forceinline__ void fwd_dif2(cplx<T> (&v)[N / 2], int h, unsigned mask) {
#pragma unroll
  for (int i = 0; i < N / 2; i++) {
    cplx<T> o = shx(v[i], mask);
    cplx<T> a = h ? o : v[i], b = h ? v[i] : o;
    if (h) {
      cplx<T> w = twp<T, N>(i); w.y = -w.y;
      v[i] = cmul(cplx<T>{a.x - b.x, a.y - b.y}, w);
    } else {
      v[i] = {a.x + b.x, a.y + b.y};
    }
  }
  Line<T, N / 2, false>::run(v);
}

template <class T, int N1, int N2>
__global__ void kxy2(cplx<T>* __restrict__ S, const T* __restrict__ vr, int nst, int nzst, const int* __restrict__ sx,
                     const int* __restrict__ sy, const int* __restrict__ xa, int nxa) {
  static_assert(N1 % 2 == 0 && N2 % 2 == 0, "split kernel needs even n1, n2");
  constexpr int LD = N1 + 1, H1 = N1 / 2, H2 = N2 / 2;
  extern __shared__ __align__(16) unsigned char smraw[];
  cplx<T>* P = reinterpret_cast<cplx<T>*>(smraw);
  const int z = blockIdx.x, j = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
  const int q = tid >> 1, h = tid & 1, nq = nt >> 1;
  cplx<T>* Sj = S + (size_t)nzst * j + (size_t)nst * z;
  for (int p = tid; p < LD * N2; p += nt) P[p] = cplx<T>{0, 0};
  __syncthreads();
  for (int p = tid; p < nst; p += nt) P[sy[p] * LD + sx[p]] = Sj[p];
  __syncthreads();
  for (int c = q; c < nxa; c += nq) {             // inverse y on occupied x-columns
    const unsigned mask = __activemask();
    const int xx = xa[c];
    cplx<T> v[H2];
#pragma unroll
    for (int i = 0; i < H2; i++) v[i] = P[(2 * i + h) * LD + xx];
    inv_dit2<T, N2>(v, h, mask);
#pragma unroll
    for (int i = 0; i < H2; i++) P[(i + h * H2) * LD + xx] = v[i];
  }
  __syncthreads();
  const T* vz = vr + (size_t)N1 * N2 * z;
  for (int r = q; r < N2; r += nq) {              // inverse x, *V, forward x
    const unsigned mask = __activemask();
    cplx<T> v[H1];
#pragma unroll
    for (int i = 0; i < H1; i++) v[i] = P[r * LD + 2 * i + h];
    inv_dit2<T, N1>(v, h, mask);
#pragma unroll
    for (int i = 0; i < H1; i++) {
      T w = __ldg(vz + r * N1 + i + h * H1);
      v[i].x *= w; v[i].y *= w;
    }
    fwd_dif2<T, N1>(v, h, mask);
#pragma unroll
    for (int i = 0; i < H1; i++) P[r * LD + 2 * i + h] = v[i];
  }
  __syncthreads();
  for (int c = q; c < nxa; c += nq) {             // forward y on occupied x-columns
    const unsigned mask = __activemask();
    const int xx = xa[c];
    cplx<T> v[H2];
#pragma unroll
    for (int i = 0; i < H2; i++) v[i] = P[(i + h * H2) * LD + xx];
    fwd_dif2<T, N2>(v, h, mask);
#pragma unroll
    for (int i = 0; i < H2; i++) P[(2 * i + h) * LD + xx] = v[i];
  }
  __syncthreads();
  for (int p = tid; p < nst; p += nt) Sj[p] = P[sy[p] * LD + sx[p]];
}

template <class T, int N1, int N2> constexpr int plane_bytes() {
  return ((N1 % 2 == 0) ? N1 + 1 : N1) * N2 * (int)sizeof(cplx<T>);
}

// ---------------------------------------------------------------------------------------------------------------
// apply_dpot / incdrhoscf on the pruned path (RTSC_STH_FDXDP; groups/compute/AMDAHL-LOG.md), FP64.
//   dpot: kxy_fwd  one block per (z-plane, column): plane of psi_kn(r) * dv(r, p) (read once, coalesced), forward x
//                  on every row, forward y on the occupied x-columns, write the k+q sticks;
//         kz_fwd_add one thread per (stick, column): forward z, d(ig) += scale * v on the k+q sphere (k_gather_add).
//   drho: kz_inv (above) on dpsi, then kxy_inv: inverse y on the occupied x-columns, inverse x on every row, write the
//         full plane into box(:, j); the engine's k_drho then forms conj(psi) * box as before.
// Box layout as the engine's: box(nnr, *) with nnr = n1 n2 n3 (checked by the caller), plane z at n1 n2 z.
// ---------------------------------------------------------------------------------------------------------------
template <class T, int N1, int N2>
__global__ void kxy_fwd(cplx<T>* __restrict__ S, const cplx<T>* __restrict__ psir, const cplx<T>* __restrict__ dv,
                        const int* __restrict__ list, int s0, int nslot, int npert, int nnr, int pj, int jb, int nst,
                        int nzst, const int* __restrict__ sx, const int* __restrict__ sy, const int* __restrict__ xa,
                        int nxa) {
  constexpr int LD = (N1 % 2 == 0) ? N1 + 1 : N1;
  extern __shared__ __align__(16) unsigned char smraw[];
  cplx<T>* P = reinterpret_cast<cplx<T>*>(smraw);
  const int z = blockIdx.x, j = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
  const int col = list[s0 + j] - 1;
  const int k = col / (nslot * npert), n = col % nslot, p = (col / nslot) % npert;
  const size_t b = pj ? (size_t)(jb + j) : (size_t)n + (size_t)nslot * k;
  const cplx<T>* ps = psir + (size_t)nnr * b + (size_t)N1 * N2 * z;
  const cplx<T>* dz = dv + (size_t)nnr * p + (size_t)N1 * N2 * z;
  for (int q = tid; q < N1 * N2; q += nt) P[(q / N1) * LD + (q % N1)] = cmul(ps[q], dz[q]);
  __syncthreads();
  for (int r = tid; r < N2; r += nt) {            // forward x on every row
    cplx<T> v[N1];
#pragma unroll
    for (int xx = 0; xx < N1; xx++) v[xx] = P[r * LD + xx];
    Line<T, N1, false>::run(v);
#pragma unroll
    for (int xx = 0; xx < N1; xx++) P[r * LD + xx] = v[xx];
  }
  __syncthreads();
  for (int c = tid; c < nxa; c += nt) {           // forward y on occupied x-columns
    const int xx = xa[c];
    cplx<T> v[N2];
#pragma unroll
    for (int y = 0; y < N2; y++) v[y] = P[y * LD + xx];
    Line<T, N2, false>::run(v);
#pragma unroll
    for (int y = 0; y < N2; y++) P[y * LD + xx] = v[y];
  }
  __syncthreads();
  cplx<T>* Sj = S + (size_t)nzst * j + (size_t)nst * z;
  for (int q = tid; q < nst; q += nt) Sj[q] = P[sy[q] * LD + sx[q]];
}

template <class T, int N3>
__global__ void __launch_bounds__(TPB_Z) kz_fwd_add(cplx<T>* __restrict__ d, const cplx<T>* __restrict__ S,
                                                    const int* __restrict__ list, int s0, int nslot, int npert,
                                                    int npwx, int nst, const int* __restrict__ zmap, int nzst,
                                                    T scale) {
  const int j = blockIdx.y;
  const int s = blockIdx.x * TPB_Z + threadIdx.x;
  if (s >= nst) return;
  const int col = list[s0 + j] - 1;
  const int k = col / (nslot * npert);
  const cplx<T>* Sj = S + (size_t)nzst * j + s;
  cplx<T> v[N3];
#pragma unroll
  for (int z = 0; z < N3; z++) v[z] = Sj[(size_t)nst * z];
  Line<T, N3, false>::run(v);
  const int* zm = zmap + (size_t)nzst * k + s;
  cplx<T>* dc = d + (size_t)npwx * col;
#pragma unroll
  for (int z = 0; z < N3; z++) {
    int ig = __ldg(zm + (size_t)nst * z);
    if (ig > 0) {
      cplx<T> a = dc[ig - 1];
      dc[ig - 1] = {a.x + v[z].x * scale, a.y + v[z].y * scale};
    }
  }
}

template <class T, int N1, int N2>
__global__ void kxy_inv(const cplx<T>* __restrict__ S, cplx<T>* __restrict__ box, int nnr, int jb, int nst, int nzst,
                        const int* __restrict__ sx, const int* __restrict__ sy, const int* __restrict__ xa, int nxa) {
  constexpr int LD = (N1 % 2 == 0) ? N1 + 1 : N1;
  extern __shared__ __align__(16) unsigned char smraw[];
  cplx<T>* P = reinterpret_cast<cplx<T>*>(smraw);
  const int z = blockIdx.x, j = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
  const cplx<T>* Sj = S + (size_t)nzst * j + (size_t)nst * z;
  for (int q = tid; q < LD * N2; q += nt) P[q] = cplx<T>{0, 0};
  __syncthreads();
  for (int q = tid; q < nst; q += nt) P[sy[q] * LD + sx[q]] = Sj[q];
  __syncthreads();
  for (int c = tid; c < nxa; c += nt) {           // inverse y on occupied x-columns
    const int xx = xa[c];
    cplx<T> v[N2];
#pragma unroll
    for (int y = 0; y < N2; y++) v[y] = P[y * LD + xx];
    Line<T, N2, true>::run(v);
#pragma unroll
    for (int y = 0; y < N2; y++) P[y * LD + xx] = v[y];
  }
  __syncthreads();
  for (int r = tid; r < N2; r += nt) {            // inverse x on every row
    cplx<T> v[N1];
#pragma unroll
    for (int xx = 0; xx < N1; xx++) v[xx] = P[r * LD + xx];
    Line<T, N1, true>::run(v);
#pragma unroll
    for (int xx = 0; xx < N1; xx++) P[r * LD + xx] = v[xx];
  }
  __syncthreads();
  cplx<T>* bz = box + (size_t)nnr * (jb + j) + (size_t)N1 * N2 * z;
  for (int q = tid; q < N1 * N2; q += nt) bz[q] = P[(q / N1) * LD + (q % N1)];
}

template <int N1, int N2, int N3>
static int run_dpot(cplx<double>* d, const cplx<double>* psir, const cplx<double>* dv, const int* list, int s0, int nb,
                    int nslot, int npert, int npwx, int nnr, int pj, double scale, cudaStream_t st, int tpb) {
  static bool attr = false;
  constexpr int shm = plane_bytes<double, N1, N2>();
  if (!attr) {
    if (cudaFuncSetAttribute(kxy_fwd<double, N1, N2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm) != cudaSuccess)
      return 5;
    attr = true;
  }
  cplx<double>* S = reinterpret_cast<cplx<double>*>(g.S);
  const int chunk = (int)std::min<size_t>(g.chunk, g.Sbytes / (sizeof(cplx<double>) * (size_t)g.nzst));
  for (int c0 = 0; c0 < nb; c0 += chunk) {
    int m = std::min(chunk, nb - c0);
    kxy_fwd<double, N1, N2><<<dim3(N3, m), tpb, shm, st>>>(S, psir, dv, list, s0 + c0, nslot, npert, nnr, pj, c0,
                                                          g.nst, g.nzst, g.sx, g.sy, g.xa, g.nxa);
    kz_fwd_add<double, N3><<<dim3((g.nst + TPB_Z - 1) / TPB_Z, m), TPB_Z, 0, st>>>(d, S, list, s0 + c0, nslot, npert,
                                                                                 npwx, g.nst, g.zmap, g.nzst, scale);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "fdx dpot: launch failed: %s\n", cudaGetErrorString(e)); return 6; }
  return 0;
}

template <int N1, int N2, int N3>
static int run_inv(cplx<double>* box, const cplx<double>* x, const int* list, int s0, int nb, int nslot, int npert,
                   int npwx, int nnr, cudaStream_t st, int tpb) {
  static bool attr = false;
  constexpr int shm = plane_bytes<double, N1, N2>();
  if (!attr) {
    if (cudaFuncSetAttribute(kxy_inv<double, N1, N2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm) != cudaSuccess)
      return 5;
    attr = true;
  }
  cplx<double>* S = reinterpret_cast<cplx<double>*>(g.S);
  const int chunk = (int)std::min<size_t>(g.chunk, g.Sbytes / (sizeof(cplx<double>) * (size_t)g.nzst));
  for (int c0 = 0; c0 < nb; c0 += chunk) {
    int m = std::min(chunk, nb - c0);
    kz_inv<double, N3><<<dim3((g.nst + TPB_Z - 1) / TPB_Z, m), TPB_Z, 0, st>>>(S, x, list, s0 + c0, nslot, npert, npwx,
                                                                             g.nst, g.zmap, g.nzst);
    kxy_inv<double, N1, N2><<<dim3(N3, m), tpb, shm, st>>>(S, box, nnr, c0, g.nst, g.nzst, g.sx, g.sy, g.xa, g.nxa);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "fdx inv: launch failed: %s\n", cudaGetErrorString(e)); return 6; }
  return 0;
}


// rtsc b2x (RTSC_STH_DRF, class E): incdrhoscf fused into the inverse plane transform. One block per (z-plane, group of
// G consecutive columns of the chunk): each column's plane is built as in kxy_inv, then A += w_k conj(psi_kn(r)) P is
// accumulated in shared memory (per run of equal pert); the full box is never written. Partial sums per (group, pert)
// go to part (exclusive per block: deterministic), and krho_reduce adds them to drho in group order.
template <class T, int N1, int N2>
__global__ void kxy_inv_rho(const cplx<T>* __restrict__ S, const cplx<T>* __restrict__ psir,
                            const int* __restrict__ list, int s0, int nslot, int npert, int nnr, int pj, int jb, int m,
                            int G, const double* __restrict__ wgt, cplx<T>* __restrict__ part, int nst, int nzst,
                            const int* __restrict__ sx, const int* __restrict__ sy, const int* __restrict__ xa,
                            int nxa) {
  constexpr int LD = (N1 % 2 == 0) ? N1 + 1 : N1;
  constexpr int NP = N1 * N2;
  extern __shared__ __align__(16) unsigned char smraw[];
  cplx<T>* P = reinterpret_cast<cplx<T>*>(smraw);
  cplx<T>* A = P + LD * N2;
  const int z = blockIdx.x, gr = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
  const int j0 = gr * G, j1 = min(j0 + G, m);
  int pcur = -1;
  for (int q = tid; q < NP; q += nt) A[q] = cplx<T>{0, 0};
  for (int j = j0; j < j1; j++) {
    const int col = list[s0 + j] - 1;
    const int k = col / (nslot * npert), n = col % nslot, p = (col / nslot) % npert;
    if (p != pcur) {
      __syncthreads();
      if (pcur >= 0) {
        cplx<T>* dst = part + ((size_t)gr * npert + pcur) * nnr + (size_t)NP * z;
        for (int q = tid; q < NP; q += nt) { cplx<T> a = dst[q]; dst[q] = {a.x + A[q].x, a.y + A[q].y}; A[q] = cplx<T>{0, 0}; }
      }
      pcur = p;
    }
    const cplx<T>* Sj = S + (size_t)nzst * j + (size_t)nst * z;
    for (int q = tid; q < LD * N2; q += nt) P[q] = cplx<T>{0, 0};
    __syncthreads();
    for (int q = tid; q < nst; q += nt) P[sy[q] * LD + sx[q]] = Sj[q];
    __syncthreads();
    for (int c = tid; c < nxa; c += nt) {         // inverse y on occupied x-columns
      const int xx = xa[c];
      cplx<T> v[N2];
#pragma unroll
      for (int y = 0; y < N2; y++) v[y] = P[y * LD + xx];
      Line<T, N2, true>::run(v);
#pragma unroll
      for (int y = 0; y < N2; y++) P[y * LD + xx] = v[y];
    }
    __syncthreads();
    for (int r = tid; r < N2; r += nt) {          // inverse x on every row
      cplx<T> v[N1];
#pragma unroll
      for (int xx = 0; xx < N1; xx++) v[xx] = P[r * LD + xx];
      Line<T, N1, true>::run(v);
#pragma unroll
      for (int xx = 0; xx < N1; xx++) P[r * LD + xx] = v[xx];
    }
    __syncthreads();
    const size_t b = pj ? (size_t)(jb + j) : (size_t)n + (size_t)nslot * k;
    const cplx<T>* ps = psir + (size_t)nnr * b + (size_t)NP * z;
    const T w = (T)wgt[k];
    for (int q = tid; q < NP; q += nt) {
      const cplx<T> c = ps[q], d = P[(q / N1) * LD + (q % N1)];
      const cplx<T> t = {c.x * d.x + c.y * d.y, c.x * d.y - c.y * d.x};   // conj(psi) * dpsi
      A[q] = {A[q].x + w * t.x, A[q].y + w * t.y};
    }
    __syncthreads();                                // P is rewritten by the next column
  }
  if (pcur >= 0) {
    cplx<T>* dst = part + ((size_t)gr * npert + pcur) * nnr + (size_t)NP * z;
    for (int q = tid; q < NP; q += nt) { cplx<T> a = dst[q]; dst[q] = {a.x + A[q].x, a.y + A[q].y}; }
  }
}

// drho(r, p) += sum over groups (ascending) of part(r, p, group)
template <class T>
__global__ void krho_reduce(cplx<T>* __restrict__ drho, const cplx<T>* __restrict__ part, int nnr, int npert, int ngr) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)nnr * npert) return;
  const size_t r = i % nnr, p = i / nnr;
  cplx<T> a = {0, 0};
  for (int g2 = 0; g2 < ngr; g2++) { cplx<T> v = part[((size_t)g2 * npert + p) * nnr + r]; a = {a.x + v.x, a.y + v.y}; }
  cplx<T> d = drho[i];
  drho[i] = {d.x + a.x, d.y + a.y};
}

static void* g_part = nullptr; static size_t g_partb = 0;

template <int N1, int N2, int N3>
static int run_inv_rho(cplx<double>* drho, const cplx<double>* x, const cplx<double>* psir, const int* list, int s0,
                       int nb, int nslot, int npert, int npwx, int nnr, int pj, const double* wgt, cudaStream_t st,
                       int tpb) {
  static bool attr = false;
  constexpr int shm = plane_bytes<double, N1, N2>() + (int)sizeof(cplx<double>) * N1 * N2;
  if (!attr) {
    if (cudaFuncSetAttribute(kxy_inv_rho<double, N1, N2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm) != cudaSuccess)
      return 5;
    attr = true;
  }
  int G = 16;
  if (const char* e = getenv("FDX_RHO_G")) G = std::max(1, atoi(e));
  cplx<double>* S = reinterpret_cast<cplx<double>*>(g.S);
  const int chunk = (int)std::min<size_t>(g.chunk, g.Sbytes / (sizeof(cplx<double>) * (size_t)g.nzst));
  const int ngmax = (std::min(chunk, nb) + G - 1) / G;
  const size_t pb = sizeof(cplx<double>) * (size_t)nnr * npert * ngmax;
  if (pb > g_partb) { cudaFree(g_part); g_partb = 0; if (cudaMalloc(&g_part, pb)) return 3; g_partb = pb; }
  cplx<double>* part = reinterpret_cast<cplx<double>*>(g_part);
  for (int c0 = 0; c0 < nb; c0 += chunk) {
    int m = std::min(chunk, nb - c0);
    int ngr = (m + G - 1) / G;
    cudaMemsetAsync(part, 0, sizeof(cplx<double>) * (size_t)nnr * npert * ngr, st);
    kz_inv<double, N3><<<dim3((g.nst + TPB_Z - 1) / TPB_Z, m), TPB_Z, 0, st>>>(S, x, list, s0 + c0, nslot, npert, npwx,
                                                                             g.nst, g.zmap, g.nzst);
    kxy_inv_rho<double, N1, N2><<<dim3(N3, ngr), tpb, shm, st>>>(S, psir, list, s0 + c0, nslot, npert, nnr, pj, c0, m, G,
                                                                wgt, part, g.nst, g.nzst, g.sx, g.sy, g.xa, g.nxa);
    const size_t nr = (size_t)nnr * npert;
    krho_reduce<double><<<(unsigned)((nr + 255) / 256), 256, 0, st>>>(drho, part, nnr, npert, ngr);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "fdx inv_rho: launch failed: %s\n", cudaGetErrorString(e)); return 6; }
  return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// supported grids and dispatch
// ---------------------------------------------------------------------------------------------------------------
#define FDX_GRIDS(X) X(0, 30, 30, 36) X(1, 54, 54, 54)


template <class T, class TG, int N1, int N2, int N3>
static int run_local(const cplx<T>* x, cplx<T>* y, const int* list, int nl, int nslot, int npert, int npwx,
                     int nbnd, const T* vrs, const int* npwq, const TG* g2, const double* etk, T scale,
                     cudaStream_t st, int tpb_xy) {
  static bool attr = false;
  constexpr int shm = plane_bytes<T, N1, N2>();
  if (!attr) {
    if (cudaFuncSetAttribute(kxy<T, N1, N2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm) != cudaSuccess) return 5;
    if (cudaFuncSetAttribute(kxy2<T, N1, N2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm) != cudaSuccess) return 5;
    attr = true;
  }
  cplx<T>* S = reinterpret_cast<cplx<T>*>(g.S);
  const int chunk = (int)std::min<size_t>(g.chunk, g.Sbytes / (sizeof(cplx<T>) * (size_t)g.nzst));
  static const int only = getenv("FDX_ONLY") ? atoi(getenv("FDX_ONLY")) : 0;   // timing ablation only
  for (int s0 = 0; s0 < nl; s0 += chunk) {
    int nb = std::min(chunk, nl - s0);
    dim3 gz((g.nst + TPB_Z - 1) / TPB_Z, nb);
    if (!only || only == 1)
      kz_inv<T, N3><<<gz, TPB_Z, 0, st>>>(S, x, list, s0, nslot, npert, npwx, g.nst, g.zmap, g.nzst);
    if ((!only || only == 2) && g.split)
      kxy2<T, N1, N2><<<dim3(N3, nb), tpb_xy, shm, st>>>(S, vrs, g.nst, g.nzst, g.sx, g.sy, g.xa, g.nxa);
    else if (!only || only == 2)
      kxy<T, N1, N2><<<dim3(N3, nb), tpb_xy, shm, st>>>(S, vrs, g.nst, g.nzst, g.sx, g.sy, g.xa, g.nxa);
    if (!only || only == 3)
      kz_fwd<T, TG, N3><<<gz, TPB_Z, 0, st>>>(y, x, S, list, s0, nslot, npert, npwx, nbnd, g.nst, g.zmap, g.nzst, npwq,
                                           g2, etk, scale);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "fdx: launch failed: %s\n", cudaGetErrorString(e)); return 6; }
  return 0;
}

static int tpb_xy_default() { return (((g.split ? 2 : 1) * std::max(g.n2, g.nxa) + 31) / 32) * 32; }

template <class T, class TG>
static int dispatch(const cplx<T>* x, cplx<T>* y, const int* list, int nl, int nslot, int npert, int npwx, int nbnd,
                    const T* vrs, const int* npwq, const TG* g2, const double* etk, T scale, cudaStream_t st) {
  int tpb = tpb_xy_default();
  if (const char* e = getenv("FDX_TPB_XY")) tpb = atoi(e);
  switch (g.gid) {
#define X(id, a, b, c) \
  case id: return run_local<T, TG, a, b, c>(x, y, list, nl, nslot, npert, npwx, nbnd, vrs, npwq, g2, etk, scale, st, tpb);
    FDX_GRIDS(X)
#undef X
  }
  return 7;
}

// occupancy report for the harness
template <class T, int N1, int N2, int N3>
static void occ_one(int tpb_xy, int out[6]) {
  cudaFuncAttributes a;
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kz_inv<T, N3>, TPB_Z, 0);
  cudaFuncGetAttributes(&a, kz_inv<T, N3>);
  out[0] = nb * TPB_Z / 32; out[1] = a.numRegs;
  auto kp = g.split ? kxy2<T, N1, N2> : kxy<T, N1, N2>;
  cudaFuncSetAttribute(kp, cudaFuncAttributeMaxDynamicSharedMemorySize, plane_bytes<T, N1, N2>());
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kp, tpb_xy, plane_bytes<T, N1, N2>());
  cudaFuncGetAttributes(&a, kp);
  out[2] = nb * ((tpb_xy + 31) / 32); out[3] = a.numRegs;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kz_fwd<T, T, N3>, TPB_Z, 0);
  cudaFuncGetAttributes(&a, kz_fwd<T, T, N3>);
  out[4] = nb * TPB_Z / 32; out[5] = a.numRegs;
  (void)a.localSizeBytes;
}

}  // namespace fdx

using namespace fdx;

extern "C" {

void fdx_free() {
  cudaFree(g.zmap); cudaFree(g.sx); cudaFree(g.sy); cudaFree(g.xa); cudaFree(g.S);
  cudaFree(g_part); g_part = nullptr; g_partb = 0;
  g = Geo();
}

// nr, nrx: FFT dims and leading dims; idxkq(npwx, nksq): 1-based linear FFT index of each k+q plane wave.
// Returns 0 when ready, > 0 when unsupported (the caller keeps its other path).
int fdx_setup(const int* nr, const int* nrx, int nksq, int npwx, const int* npwq, const int* idxkq, int chunk) {
  fdx_free();
  const int n1 = nr[0], n2 = nr[1], n3 = nr[2];
  if (nrx[0] != n1 || nrx[1] != n2) return 1;
  int gid = -1;
#define X(id, a, b, c) if (n1 == a && n2 == b && n3 == c) gid = id;
  FDX_GRIDS(X)
#undef X
  if (gid < 0) return 2;
  std::vector<int> smap(n1 * n2, 0);
  std::vector<char> on(n1 * n2, 0);
  for (int k = 0; k < nksq; k++)
    for (int ig = 0; ig < npwq[k]; ig++) {
      int l = idxkq[(size_t)npwx * k + ig] - 1;
      on[(l % nrx[0]) + n1 * ((l / nrx[0]) % nrx[1])] = 1;
    }
  std::vector<int> sx, sy, xa;
  std::vector<char> xon(n1, 0);
  for (int i2 = 0; i2 < n2; i2++)           // sticks in (y, x) scan order
    for (int i1 = 0; i1 < n1; i1++)
      if (on[i1 + n1 * i2]) { sx.push_back(i1); sy.push_back(i2); smap[i1 + n1 * i2] = (int)sx.size(); xon[i1] = 1; }
  for (int i1 = 0; i1 < n1; i1++) if (xon[i1]) xa.push_back(i1);
  const int nst = (int)sx.size(), nzst = nst * n3;
  std::vector<int> zm((size_t)nzst * nksq, 0);
  for (int k = 0; k < nksq; k++)
    for (int ig = 0; ig < npwq[k]; ig++) {
      int l = idxkq[(size_t)npwx * k + ig] - 1;
      int i1 = l % nrx[0], i2 = (l / nrx[0]) % nrx[1], i3 = l / (nrx[0] * nrx[1]);
      zm[(size_t)nzst * k + (smap[i1 + n1 * i2] - 1) + (size_t)nst * i3] = ig + 1;
    }
  g.n1 = n1; g.n2 = n2; g.n3 = n3; g.nst = nst; g.nxa = (int)xa.size(); g.nzst = nzst; g.nksq = nksq; g.gid = gid;
  size_t fb, tb;
  cudaMemGetInfo(&fb, &tb);
  g.chunk = (int)std::max<size_t>(1, std::min<size_t>(chunk, (fb / 8) / (16 * (size_t)nzst)));
  g.Sbytes = (size_t)16 * nzst * g.chunk;
  if (cudaMalloc(&g.zmap, sizeof(int) * zm.size()) || cudaMalloc(&g.sx, sizeof(int) * nst) ||
      cudaMalloc(&g.sy, sizeof(int) * nst) || cudaMalloc(&g.xa, sizeof(int) * g.nxa) || cudaMalloc(&g.S, g.Sbytes)) {
    fdx_free(); return 3;
  }
  cudaMemcpy(g.zmap, zm.data(), sizeof(int) * zm.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(g.sx, sx.data(), sizeof(int) * nst, cudaMemcpyHostToDevice);
  cudaMemcpy(g.sy, sy.data(), sizeof(int) * nst, cudaMemcpyHostToDevice);
  cudaMemcpy(g.xa, xa.data(), sizeof(int) * g.nxa, cudaMemcpyHostToDevice);
  double tw[27][2];
  for (int k = 0; k < 27; k++) { tw[k][0] = cos(2 * M_PI * k / 54.0); tw[k][1] = sin(2 * M_PI * k / 54.0); }
  cudaMemcpyToSymbol(c_tw54, tw, sizeof(tw));
  static double twl[TWL][2];
  for (int m = 0; m < TWL; m++) { twl[m][0] = cos(2 * M_PI * m / TWL); twl[m][1] = sin(2 * M_PI * m / TWL); }
  cudaMemcpyToSymbol(c_twL, twl, sizeof(twl));
  g.split = getenv("FDX_SPLIT") ? atoi(getenv("FDX_SPLIT")) : 0;   // 1: kxy2 (slower on the A40; H100 option)
  if (n1 % 2 || n2 % 2) g.split = 0;
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

void fdx_info(int* nst, int* nxa, int* chunk) { *nst = g.nst; *nxa = g.nxa; *chunk = g.chunk; }

// rtsc amdahl (EPW coarse g, RTSC_EPW_FDX): geometry of ONE sphere (nksq = 1), rebuilt per call with grow-only device
// buffers (no cudaMalloc/cudaFree in the steady state). ncols = columns the stick buffer must hold. 0 = ok.
static size_t gz_cap = 0, gs_cap = 0, gx_cap = 0;
int fdx_geom1(const int* nr, const int* nrx, int npwq, const int* idxkq, int ncols) {
  const int n1 = nr[0], n2 = nr[1], n3 = nr[2];
  if (nrx[0] != n1 || nrx[1] != n2) return 1;
  int gid = -1;
#define X(id, a, b, c) if (n1 == a && n2 == b && n3 == c) gid = id;
  FDX_GRIDS(X)
#undef X
  if (gid < 0) return 2;
  static std::vector<int> smap, sx, sy, xa, zm;
  static std::vector<char> on, xon;
  static bool tw = false;
  smap.assign(n1 * n2, 0); on.assign(n1 * n2, 0); xon.assign(n1, 0);
  sx.clear(); sy.clear(); xa.clear();
  for (int ig = 0; ig < npwq; ig++) {
    int l = idxkq[ig] - 1;
    on[(l % nrx[0]) + n1 * ((l / nrx[0]) % nrx[1])] = 1;
  }
  for (int i2 = 0; i2 < n2; i2++)
    for (int i1 = 0; i1 < n1; i1++)
      if (on[i1 + n1 * i2]) { sx.push_back(i1); sy.push_back(i2); smap[i1 + n1 * i2] = (int)sx.size(); xon[i1] = 1; }
  for (int i1 = 0; i1 < n1; i1++) if (xon[i1]) xa.push_back(i1);
  const int nst = (int)sx.size(), nzst = nst * n3;
  zm.assign((size_t)nzst, 0);
  for (int ig = 0; ig < npwq; ig++) {
    int l = idxkq[ig] - 1;
    int i1 = l % nrx[0], i2 = (l / nrx[0]) % nrx[1], i3 = l / (nrx[0] * nrx[1]);
    zm[(smap[i1 + n1 * i2] - 1) + (size_t)nst * i3] = ig + 1;
  }
  if ((size_t)nzst > gz_cap) { cudaFree(g.zmap); gz_cap = (size_t)nzst * 5 / 4; if (cudaMalloc(&g.zmap, 4 * gz_cap)) return 3; }
  if ((size_t)nst > gx_cap) {
    cudaFree(g.sx); cudaFree(g.sy); cudaFree(g.xa); gx_cap = std::max<size_t>((size_t)nst * 5 / 4, n1);
    if (cudaMalloc(&g.sx, 4 * gx_cap) || cudaMalloc(&g.sy, 4 * gx_cap) || cudaMalloc(&g.xa, 4 * gx_cap)) return 3;
  }
  const size_t sb = (size_t)16 * nzst * ncols;
  if (sb > gs_cap) { cudaFree(g.S); gs_cap = sb * 5 / 4; if (cudaMalloc(&g.S, gs_cap)) return 3; }
  cudaMemcpy(g.zmap, zm.data(), 4 * (size_t)nzst, cudaMemcpyHostToDevice);
  cudaMemcpy(g.sx, sx.data(), 4 * (size_t)nst, cudaMemcpyHostToDevice);
  cudaMemcpy(g.sy, sy.data(), 4 * (size_t)nst, cudaMemcpyHostToDevice);
  cudaMemcpy(g.xa, xa.data(), 4 * xa.size(), cudaMemcpyHostToDevice);
  if (!tw) {
    double t54[27][2];
    for (int k = 0; k < 27; k++) { t54[k][0] = cos(2 * M_PI * k / 54.0); t54[k][1] = sin(2 * M_PI * k / 54.0); }
    cudaMemcpyToSymbol(c_tw54, t54, sizeof(t54));
    tw = true;
  }
  g.n1 = n1; g.n2 = n2; g.n3 = n3; g.nst = nst; g.nxa = (int)xa.size(); g.nzst = nzst; g.nksq = 1; g.gid = gid;
  g.split = 0; g.Sbytes = gs_cap; g.chunk = ncols;
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

// RTSC_STH_FDXDP (FP64). dpot: d(npwx, *) += scale * gather_kq(FFT_fwd(psir(:, b) * dv(:, p))) for the nb columns
// list(s0+1 .. s0+nb); b = the column's (k, n) slot of the psi(r) cache (pj = 0) or the chunk column j (pj = 1).
int fdx_dpot_d(void* d, const void* psir, const void* dv, const int* list, int s0, int nb, int nslot, int npert,
               int npwx, int nnr, int pj, double scale, cudaStream_t st) {
  if (nnr != g.n1 * g.n2 * g.n3) return 8;
  int tpb = tpb_xy_default();
  switch (g.gid) {
#define X(id, a, b, c) \
  case id: return run_dpot<a, b, c>((cplx<double>*)d, (const cplx<double>*)psir, (const cplx<double>*)dv, list, s0, nb, \
                                    nslot, npert, npwx, nnr, pj, scale, st, tpb);
    FDX_GRIDS(X)
#undef X
  }
  return 7;
}

// drho: box(:, j) = FFT_inv(x on the k+q sphere) for the nb columns list(s0+1 .. s0+nb), j = 1..nb (full plane)
int fdx_inv_d(void* box, const void* x, const int* list, int s0, int nb, int nslot, int npert, int npwx, int nnr,
              cudaStream_t st) {
  if (nnr != g.n1 * g.n2 * g.n3) return 8;
  int tpb = tpb_xy_default();
  switch (g.gid) {
#define X(id, a, b, c) \
  case id: return run_inv<a, b, c>((cplx<double>*)box, (const cplx<double>*)x, list, s0, nb, nslot, npert, npwx, nnr, \
                                   st, tpb);
    FDX_GRIDS(X)
#undef X
  }
  return 7;
}

// RTSC_STH_DRF: drho(:, p) += sum_j wgt(k_j) conj(psir(:, b_j)) FFT_inv(x on the k+q sphere)(:, j) for the nb columns
// list(s0+1 .. s0+nb), without the full box (b_j = the (k, n) slot of the cache, pj = 0, or the chunk column, pj = 1)
int fdx_inv_rho_d(void* drho, const void* x, const void* psir, const int* list, int s0, int nb, int nslot, int npert,
                  int npwx, int nnr, int pj, const double* wgt, cudaStream_t st) {
  if (nnr != g.n1 * g.n2 * g.n3) return 8;
  int tpb = tpb_xy_default();
  if (const char* e = getenv("FDX_RHO_TPB")) tpb = atoi(e);
  switch (g.gid) {
#define X(id, a, b, c) \
  case id: return run_inv_rho<a, b, c>((cplx<double>*)drho, (const cplx<double>*)x, (const cplx<double>*)psir, list, s0, \
                                       nb, nslot, npert, npwx, nnr, pj, wgt, st, tpb);
    FDX_GRIDS(X)
#undef X
  }
  return 7;
}

// FP64: x, y complex(npwx, *); vrs(nnr); g2(npwx, nksq); etk(nbnd, nksq); list 1-based columns
int fdx_local_d(const void* x, void* y, const int* list, int nl, int nslot, int npert, int npwx, int nksq, int nbnd,
                const double* vrs, const int* npwq, const double* g2, const double* etk, double scale, cudaStream_t st) {
  (void)nksq;
  return dispatch<double, double>((const cplx<double>*)x, (cplx<double>*)y, list, nl, nslot, npert, npwx, nbnd, vrs,
                                  npwq, g2, etk, scale, st);
}

// FP32 (MP inner solve): x, y complex(sp); vrs, g2 real(sp); etk real(dp)
int fdx_local_s(const void* x, void* y, const int* list, int nl, int nslot, int npert, int npwx, int nksq, int nbnd,
                const float* vrs, const int* npwq, const float* g2, const double* etk, float scale, cudaStream_t st) {
  (void)nksq;
  return dispatch<float, float>((const cplx<float>*)x, (cplx<float>*)y, list, nl, nslot, npert, npwx, nbnd, vrs, npwq,
                                g2, etk, scale, st);
}

#ifdef FDX_EMU
void fdx_set_emu(int m) {
  static double t[540][2];
  for (int i = 0; i < 540; i++) { t[i][0] = cos(2 * M_PI * i / 540); t[i][1] = sin(2 * M_PI * i / 540); }
  cudaMemcpyToSymbol(c_twE, t, sizeof(t));
  cudaMemcpyToSymbol(d_emu, &m, sizeof(int));
}
#endif

// occupancy: out[0..5] = (warps/SM, regs) of kz_inv, kxy, kz_fwd for the current grid; prec 8 or 4
void fdx_occupancy(int prec, int out[6]) {
  int tpb = tpb_xy_default();
  if (const char* e = getenv("FDX_TPB_XY")) tpb = atoi(e);
  switch (g.gid) {
#define X(id, a, b, c) \
  case id: if (prec == 8) occ_one<double, a, b, c>(tpb, out); else occ_one<float, a, b, c>(tpb, out); break;
    FDX_GRIDS(X)
#undef X
  }
}

}  // extern "C"
