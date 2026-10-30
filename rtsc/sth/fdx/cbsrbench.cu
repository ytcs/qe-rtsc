// cbsrbench: CBSR Stage 0 (groups/theory/returns/M4/S.md; FFT-LOG.md "CBSR Stage 0").
//   zheevd (cuSOLVER, FP64 complex Hermitian) wall time and device memory, values only (jobz=N) and with vectors (jobz=V);
//   cublasZgemm FP64 throughput for C(n x N) = A(n x n) B(n x N); pinned H2D streaming of a 1.16 GB block.
//   ./cbsrbench [n ...]   (default 1535 6000 8500)
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void k_fill(cuDoubleComplex* a, int n, unsigned seed) {   // A = B + B^H, B hashed uniform(-0.5, 0.5)
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)n * n) return;
  int r = i % n, c = i / n;
  auto h = [&](unsigned x) { x ^= seed; x *= 0x9E3779B1u; x ^= x >> 15; x *= 0x85EBCA77u; x ^= x >> 13; return (x & 0xFFFFFF) / 16777216.0 - 0.5; };
  unsigned ij = (unsigned)(r * 7919u + c * 104729u), ji = (unsigned)(c * 7919u + r * 104729u);
  double re = h(ij) + h(ji), im = h(ij + 1u) - h(ji + 1u);
  if (r == c) im = 0;
  a[i] = make_cuDoubleComplex(re + (r == c ? 0.01 * r : 0.0), im);
}

int main(int argc, char** argv) {
  std::vector<int> ns;
  for (int i = 1; i < argc; i++) ns.push_back(atoi(argv[i]));
  if (ns.empty()) ns = {1535, 6000, 8500};
  cusolverDnHandle_t sh; cusolverDnCreate(&sh);
  cublasHandle_t bh; cublasCreate(&bh);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float ms;
  for (int n : ns) {
    cuDoubleComplex *A, *A0, *work;
    double* W;
    int* info;
    size_t nn = (size_t)n * n;
    cudaMalloc(&A, 16 * nn); cudaMalloc(&A0, 16 * nn); cudaMalloc(&W, 8 * n); cudaMalloc(&info, 4);
    k_fill<<<(nn + 255) / 256, 256>>>(A0, n, 12345u);
    for (int jv = 0; jv < 2; jv++) {
      cusolverEigMode_t jobz = jv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
      int lw = 0;
      cusolverDnZheevd_bufferSize(sh, jobz, CUBLAS_FILL_MODE_LOWER, n, A, n, W, &lw);
      cudaMalloc(&work, 16 * (size_t)lw);
      size_t f0, t0; cudaMemGetInfo(&f0, &t0);
      double best = 1e30;
      int nrep = n > 5000 ? 2 : 4;
      for (int r = 0; r <= nrep; r++) {   // first call = warm-up
        cudaMemcpy(A, A0, 16 * nn, cudaMemcpyDeviceToDevice);
        cudaEventRecord(e0);
        cusolverDnZheevd(sh, jobz, CUBLAS_FILL_MODE_LOWER, n, A, n, W, work, lw, info);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
        if (r > 0 && ms < best) best = ms;
      }
      int hinfo; cudaMemcpy(&hinfo, info, 4, cudaMemcpyDeviceToHost);
      printf("zheevd n %5d jobz=%s: %9.1f ms (best of %d)  info %d  matrix %.0f MB + workspace %.0f MB  (device free %.1f GB)\n", n,
             jv ? "V" : "N", best, nrep, hinfo, 16.0 * nn / 1e6, 16.0 * lw / 1e6, f0 / 1e9);
      cudaFree(work);
    }
    // ZGEMM C(n x N) = A(n x n) B(n x N)
    for (int N : {16, 20, 64, 128, 256, 512, 1024}) {
      cuDoubleComplex *B, *C;
      cudaMalloc(&B, 16 * (size_t)n * N); cudaMalloc(&C, 16 * (size_t)n * N);
      cudaMemset(B, 0, 16 * (size_t)n * N);
      cuDoubleComplex one = {1, 0}, zero = {0, 0};
      int nrep = 20;
      for (int r = 0; r <= nrep; r++) {
        if (r == 1) cudaEventRecord(e0);
        cublasZgemm(bh, CUBLAS_OP_N, CUBLAS_OP_N, n, N, n, &one, A0, n, B, n, &zero, C, n);
      }
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
      double t = ms * 1e-3 / nrep;
      printf("zgemm  n %5d N %5d: %9.3f ms  %6.2f TFLOP/s  (%.2f us per column)\n", n, N, t * 1e3, 8.0 * n * n * N / t / 1e12,
             t * 1e6 / N);
      cudaFree(B); cudaFree(C);
    }
    cudaFree(A); cudaFree(A0); cudaFree(W); cudaFree(info);
  }
  // pinned H2D streaming of one 1.16 GB basis
  size_t bytes = 1160000000ull;
  void *h, *d;
  cudaMallocHost(&h, bytes); cudaMalloc(&d, bytes);
  memset(h, 1, bytes);
  for (int r = 0; r < 3; r++) {
    cudaEventRecord(e0); cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); cudaEventRecord(e1);
    cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
    printf("H2D pinned 1.16 GB: %.1f ms  %.1f GB/s\n", ms, bytes / (ms * 1e-3) / 1e9);
  }
  return 0;
}
