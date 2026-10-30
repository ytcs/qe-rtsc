// gemmbench: FFT-LOG item (7), dense local potential V(G-G') on the k+q sphere as a batched ZGEMM / CGEMM.
// Y(npw, ncol) = V_k(npw, npw) X(npw, ncol) for the columns sharing one k, materialised V_k (the best case for cuBLAS;
// an on-the-fly V(G-G') lookup would need a custom GEMM). Reports us per column and achieved TFLOP/s (8 npw^2 per column).
//   ./gemmbench  (npw 1535 = B1, 6000 = B2; ncol 8 .. 256)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main() {
  cublasHandle_t h; cublasCreate(&h);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int npw : {1535, 6000}) for (int prec : {8, 4}) for (int nc : {8, 24, 64, 256}) {
    size_t es = prec == 8 ? 16 : 8;
    void *V, *X, *Y;
    cudaMalloc(&V, es * (size_t)npw * npw); cudaMalloc(&X, es * (size_t)npw * nc); cudaMalloc(&Y, es * (size_t)npw * nc);
    cudaMemset(V, 0, es * (size_t)npw * npw); cudaMemset(X, 0, es * (size_t)npw * nc);
    int nrep = npw > 3000 ? 20 : 100;
    for (int r = 0; r <= nrep; r++) {
      if (r == 1) cudaEventRecord(a);
      if (prec == 8) {
        cuDoubleComplex one = {1, 0}, zero = {0, 0};
        cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, npw, nc, npw, &one, (cuDoubleComplex*)V, npw, (cuDoubleComplex*)X, npw,
                    &zero, (cuDoubleComplex*)Y, npw);
      } else {
        cuComplex one = {1, 0}, zero = {0, 0};
        cublasCgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, npw, nc, npw, &one, (cuComplex*)V, npw, (cuComplex*)X, npw, &zero,
                    (cuComplex*)Y, npw);
      }
    }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    double t = ms * 1e-3 / nrep, us = t * 1e6 / nc, tf = 8.0 * npw * npw * nc / t / 1e12;
    printf("npw %5d %s ncol %4d: %8.3f us/column  %6.2f TFLOP/s  (V_k %.1f MB)\n", npw, prec == 8 ? "Z" : "C", nc, us, tf,
           es * (double)npw * npw / 1e6);
    cudaFree(V); cudaFree(X); cudaFree(Y);
  }
  return 0;
}
