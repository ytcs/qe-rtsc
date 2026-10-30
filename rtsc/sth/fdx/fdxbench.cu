// fdxbench: per-application cost of the engine's local-operator FFT chain (groups/compute/FFT-LOG.md).
//   y = g2kin x + (1/N) FFT_fwd[ V(r) FFT_inv[ scatter(x) ] ](sphere) - e x   for nl columns, batched.
// Variants, each FP64 and FP32 (the MP inner path):
//   A  the engine's cuFFT path (k_scatter_full, cufftPlanMany 3D Z2Z/C2C batch 256, k_vmul, 3D, k_gather_h);
//   P  pfft (rtsc/sth/sth_pfft.f90 / sth_pfft_s.f90, via pfwrap.f90);
//   D  fftdx (sth_fftdx.cu, cuFFTDx thread FFTs, pruned and fused).
// Geometry exactly as the engine sees it: the k+q spheres |k+q+G|^2 (2 pi/a)^2 <= ecutwfc of the frozen cells
// (B1 MgB2 hex, 60 Ry, 30x30x36; B2 Mg2IrH6 fcc, 80 Ry, 54^3) for nksq k-points drawn from the MP grid; the stick set
// is the union over those k (as pfft_setup / fdx_setup build it).
//   ./fdxbench B1|B2 [nl=1024] [nslot=8] [nrep=20] [nksq: B1 666, B2 189]
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <array>

extern "C" {
int fdx_setup(const int*, const int*, int, int, const int*, const int*, int);
void fdx_info(int*, int*, int*);
void fdx_occupancy(int, int*);

int fdx_local_d(const void*, void*, const int*, int, int, int, int, int, int, const double*, const int*,
                const double*, const double*, double, cudaStream_t);
int fdx_local_s(const void*, void*, const int*, int, int, int, int, int, int, const float*, const int*, const float*,
                const double*, float, cudaStream_t);
void pfw_setup(int, const int*, const int*, int, int, const int*, const int*, int, int*);
void pfw_info(int*, int*, int*);
void pfw_local_d(const void*, void*, const int*, int, int, int, int, int, int, const double*, const int*,
                 const double*, const double*, double, cudaStream_t);
void pfw_local_s(const void*, void*, const int*, int, int, int, int, int, int, const float*, const int*, const float*,
                 const double*, float, cudaStream_t);
}

#ifdef FDX_EMU
extern "C" void fdx_set_emu(int);
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

template <class T> struct C2 { T x, y; };

// ---- the engine's cuFFT-path kernels (sth_engine.f90: k_scatter_full, k_vmul, k_gather_h), C++ transcription ----
template <class T>
__global__ void k_scatter_full(C2<T>* box, const C2<T>* x, const int* list, int s0, int nnr, int npwx, int nslot,
                               int npert, const int* inv) {
  int j = blockIdx.y, s = list[s0 + j] - 1, k = s / (nslot * npert);
  int ir = blockIdx.x * blockDim.x + threadIdx.x;
  if (ir < nnr) {
    int ig = inv[(size_t)nnr * k + ir];
    box[(size_t)nnr * j + ir] = ig > 0 ? x[(size_t)npwx * s + ig - 1] : C2<T>{0, 0};
  }
}
template <class T>
__global__ void k_vmul(C2<T>* box, const T* v, int nnr) {
  int j = blockIdx.y, ir = blockIdx.x * blockDim.x + threadIdx.x;
  if (ir < nnr) { C2<T> b = box[(size_t)nnr * j + ir]; T w = v[ir]; box[(size_t)nnr * j + ir] = {b.x * w, b.y * w}; }
}
template <class T>
__global__ void k_gather_h(C2<T>* y, const C2<T>* x, const C2<T>* box, const int* list, int s0, int nnr, int npwx,
                           int nslot, int npert, int nbnd, const int* npwq, const int* idxq, const T* g2,
                           const double* etk, T scale) {
  int j = blockIdx.y, s = list[s0 + j] - 1, k = s / (nslot * npert), n = s % nslot;
  int ig = blockIdx.x * blockDim.x + threadIdx.x;
  if (ig < npwx) {
    size_t o = (size_t)npwx * s + ig;
    if (ig < npwq[k]) {
      C2<T> xv = x[o], b = box[(size_t)nnr * j + idxq[(size_t)npwx * k + ig] - 1];
      T gg = g2[(size_t)npwx * k + ig], e = (T)etk[(size_t)nbnd * k + n];
      y[o] = {gg * xv.x + b.x * scale - e * xv.x, gg * xv.y + b.y * scale - e * xv.y};
    } else y[o] = {0, 0};
  }
}

struct Geom {
  int n[3], nnr, nksq, npwx;
  std::vector<int> npwq, idx, inv;   // idx(npwx, nksq) 1-based linear FFT index; inv(nnr, nksq) 1-based pw index
  double box_frac_sphere;
};

static Geom make_geom(const char* sys, int nksq, unsigned seed) {
  Geom G;
  double b[3][3], ecut, alat;
  int mp;
  if (!strcmp(sys, "B1")) {   // ibrav 4, celldm(1) 5.8260252227888, c/a 1.1420694129095, ecutwfc 60, k 18^3
    alat = 5.8260252227888; double c = 1.1420694129095; ecut = 60; mp = 18;
    G.n[0] = 30; G.n[1] = 30; G.n[2] = 36;
    double s3 = sqrt(3.0);
    double bb[3][3] = {{1, 1 / s3, 0}, {0, 2 / s3, 0}, {0, 0, 1 / c}};
    memcpy(b, bb, sizeof b);
  } else {                    // ibrav 2, celldm(1) 12.569178, ecutwfc 80, k 12^3
    alat = 12.569178; ecut = 80; mp = 12;
    G.n[0] = G.n[1] = G.n[2] = 54;
    double bb[3][3] = {{-1, -1, 1}, {1, 1, 1}, {-1, 1, -1}};
    memcpy(b, bb, sizeof b);
  }
  const double tpiba2 = pow(2 * M_PI / alat, 2);
  G.nnr = G.n[0] * G.n[1] * G.n[2];
  // all MP k (fractional in [-0.5, 0.5), as kpoint_grid stores them), nksq of them at random
  std::vector<std::array<double, 3>> kall;
  for (int i = 0; i < mp; i++) for (int j = 0; j < mp; j++) for (int l = 0; l < mp; l++) {
    double f[3] = {(double)i / mp, (double)j / mp, (double)l / mp};
    for (double& v : f) if (v >= 0.5) v -= 1;
    kall.push_back({f[0] * b[0][0] + f[1] * b[1][0] + f[2] * b[2][0], f[0] * b[0][1] + f[1] * b[1][1] + f[2] * b[2][1],
                    f[0] * b[0][2] + f[1] * b[1][2] + f[2] * b[2][2]});
  }
  std::mt19937 rng(seed);
  std::shuffle(kall.begin(), kall.end(), rng);
  G.nksq = std::min<int>(nksq, kall.size());
  std::vector<std::vector<int>> lists(G.nksq);
  int mmax[3];
  for (int d = 0; d < 3; d++) mmax[d] = G.n[d] / 2;
  G.npwx = 0;
  for (int k = 0; k < G.nksq; k++) {
    for (int m1 = -mmax[0]; m1 <= mmax[0]; m1++) for (int m2 = -mmax[1]; m2 <= mmax[1]; m2++)
      for (int m3 = -mmax[2]; m3 <= mmax[2]; m3++) {
        double gx = kall[k][0] + m1 * b[0][0] + m2 * b[1][0] + m3 * b[2][0];
        double gy = kall[k][1] + m1 * b[0][1] + m2 * b[1][1] + m3 * b[2][1];
        double gz = kall[k][2] + m1 * b[0][2] + m2 * b[1][2] + m3 * b[2][2];
        if ((gx * gx + gy * gy + gz * gz) * tpiba2 <= ecut) {
          int i1 = (m1 + G.n[0]) % G.n[0], i2 = (m2 + G.n[1]) % G.n[1], i3 = (m3 + G.n[2]) % G.n[2];
          lists[k].push_back(1 + i1 + G.n[0] * (i2 + G.n[1] * i3));
        }
      }
    std::shuffle(lists[k].begin(), lists[k].end(), rng);   // QE orders by |G|: not by FFT index
    G.npwx = std::max<int>(G.npwx, lists[k].size());
  }
  G.npwq.resize(G.nksq);
  G.idx.assign((size_t)G.npwx * G.nksq, 1);
  G.inv.assign((size_t)G.nnr * G.nksq, 0);
  double sumf = 0;
  for (int k = 0; k < G.nksq; k++) {
    G.npwq[k] = lists[k].size();
    for (int ig = 0; ig < G.npwq[k]; ig++) {
      G.idx[(size_t)G.npwx * k + ig] = lists[k][ig];
      G.inv[(size_t)G.nnr * k + lists[k][ig] - 1] = ig + 1;
    }
    sumf += (double)G.npwq[k] / G.nnr;
  }
  G.box_frac_sphere = sumf / G.nksq;
  return G;
}

template <class T> struct Prec;
template <> struct Prec<double> { static constexpr cufftType t = CUFFT_Z2Z; static const char* name() { return "FP64"; } };
template <> struct Prec<float> { static constexpr cufftType t = CUFFT_C2C; static const char* name() { return "FP32"; } };

static void fft_exec(cufftHandle p, C2<double>* b, int dir) { cufftExecZ2Z(p, (cufftDoubleComplex*)b, (cufftDoubleComplex*)b, dir); }
static void fft_exec(cufftHandle p, C2<float>* b, int dir) { cufftExecC2C(p, (cufftComplex*)b, (cufftComplex*)b, dir); }
static void fdx_local(const C2<double>* x, C2<double>* y, const int* l, int nl, int ns, int np, int npwx, int nk, int nb,
                      const double* v, const int* npwq, const double* g2, const double* e, double sc, cudaStream_t st) {
  int r = fdx_local_d(x, y, l, nl, ns, np, npwx, nk, nb, v, npwq, g2, e, sc, st); if (r) { printf("fdx err %d\n", r); exit(1); }
}
static void fdx_local(const C2<float>* x, C2<float>* y, const int* l, int nl, int ns, int np, int npwx, int nk, int nb,
                      const float* v, const int* npwq, const float* g2, const double* e, float sc, cudaStream_t st) {
  int r = fdx_local_s(x, y, l, nl, ns, np, npwx, nk, nb, v, npwq, g2, e, sc, st); if (r) { printf("fdx err %d\n", r); exit(1); }
}
static void pf_local(const C2<double>* x, C2<double>* y, const int* l, int nl, int ns, int np, int npwx, int nk, int nb,
                     const double* v, const int* npwq, const double* g2, const double* e, double sc, cudaStream_t st) {
  pfw_local_d(x, y, l, nl, ns, np, npwx, nk, nb, v, npwq, g2, e, sc, st);
}
static void pf_local(const C2<float>* x, C2<float>* y, const int* l, int nl, int ns, int np, int npwx, int nk, int nb,
                     const float* v, const int* npwq, const float* g2, const double* e, float sc, cudaStream_t st) {
  pfw_local_s(x, y, l, nl, ns, np, npwx, nk, nb, v, npwq, g2, e, sc, st);
}

struct Res { double us[3]; std::vector<C2<double>> y[3]; };

// run the three variants in precision T on the same data; y returned in double for comparisons
template <class T>
static Res run_prec(const Geom& G, int nl, int nslot, int nrep, int nbnd, const std::vector<double>& hx,
                    const std::vector<double>& hv, const std::vector<double>& hg2, const std::vector<double>& het,
                    const std::vector<int>& hlist, bool dopf, int nst_fdx) {
  Res R;
  const int npwx = G.npwx, nnr = G.nnr, nk = G.nksq;
  const size_t nx = (size_t)npwx * nl;
  std::vector<C2<T>> xt(nx);
  for (size_t i = 0; i < nx; i++) xt[i] = {(T)hx[2 * i], (T)hx[2 * i + 1]};
  std::vector<T> vt(hv.begin(), hv.end()), gt(hg2.begin(), hg2.end());
  C2<T> *x, *y, *box;
  T *v, *g2;
  double* et;
  int *list, *npwq, *idx, *inv;
  const int B = 256;
  CK(cudaMalloc(&x, sizeof(C2<T>) * nx)); CK(cudaMalloc(&y, sizeof(C2<T>) * nx));
  CK(cudaMalloc(&box, sizeof(C2<T>) * (size_t)nnr * B));
  CK(cudaMalloc(&v, sizeof(T) * nnr)); CK(cudaMalloc(&g2, sizeof(T) * gt.size())); CK(cudaMalloc(&et, sizeof(double) * het.size()));
  CK(cudaMalloc(&list, sizeof(int) * nl)); CK(cudaMalloc(&npwq, sizeof(int) * nk));
  CK(cudaMalloc(&idx, sizeof(int) * G.idx.size())); CK(cudaMalloc(&inv, sizeof(int) * G.inv.size()));
  cudaMemcpy(x, xt.data(), sizeof(C2<T>) * nx, cudaMemcpyHostToDevice);
  cudaMemcpy(v, vt.data(), sizeof(T) * nnr, cudaMemcpyHostToDevice);
  cudaMemcpy(g2, gt.data(), sizeof(T) * gt.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(et, het.data(), sizeof(double) * het.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(list, hlist.data(), sizeof(int) * nl, cudaMemcpyHostToDevice);
  cudaMemcpy(npwq, G.npwq.data(), sizeof(int) * nk, cudaMemcpyHostToDevice);
  cudaMemcpy(idx, G.idx.data(), sizeof(int) * G.idx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(inv, G.inv.data(), sizeof(int) * G.inv.size(), cudaMemcpyHostToDevice);
  cudaStream_t st; cudaStreamCreate(&st);
  cufftHandle plan;
  int n[3] = {G.n[2], G.n[1], G.n[0]};
  if (cufftPlanMany(&plan, 3, n, n, 1, nnr, n, 1, nnr, Prec<T>::t, B) != CUFFT_SUCCESS) { printf("plan failed\n"); exit(1); }
  cufftSetStream(plan, st);
  const T scale = (T)(1.0 / nnr);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int var = 0; var < 3; var++) {
    if (var == 1 && !dopf) { R.us[1] = 0; continue; }
    cudaMemset(y, 0, sizeof(C2<T>) * nx);
    for (int rep = 0; rep <= nrep; rep++) {
      if (rep == 1) cudaEventRecord(e0, st);
      if (var == 0) {
        for (int s0 = 0; s0 < nl; s0 += B) {
          int nb = std::min(B, nl - s0);
          k_scatter_full<T><<<dim3((nnr + 127) / 128, nb), 128, 0, st>>>(box, x, list, s0, nnr, npwx, nslot, 1, inv);
          fft_exec(plan, box, CUFFT_INVERSE);
          k_vmul<T><<<dim3((nnr + 127) / 128, nb), 128, 0, st>>>(box, v, nnr);
          fft_exec(plan, box, CUFFT_FORWARD);
          k_gather_h<T><<<dim3((npwx + 127) / 128, nb), 128, 0, st>>>(y, x, box, list, s0, nnr, npwx, nslot, 1, nbnd,
                                                                      npwq, idx, g2, et, scale);
        }
      } else if (var == 1) {
        pf_local(x, y, list, nl, nslot, 1, npwx, nk, nbnd, v, npwq, g2, et, scale, st);
      } else {
        fdx_local(x, y, list, nl, nslot, 1, npwx, nk, nbnd, v, npwq, g2, et, scale, st);
      }
    }
    cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    R.us[var] = 1e3 * ms / ((double)nrep * nl);
    CK(cudaGetLastError());
    std::vector<C2<T>> h(nx);
    cudaMemcpy(h.data(), y, sizeof(C2<T>) * nx, cudaMemcpyDeviceToHost);
    R.y[var].resize(nx);
    for (size_t i = 0; i < nx; i++) R.y[var][i] = {(double)h[i].x, (double)h[i].y};
  }
  cufftDestroy(plan); cudaStreamDestroy(st);
  cudaFree(x); cudaFree(y); cudaFree(box); cudaFree(v); cudaFree(g2); cudaFree(et); cudaFree(list); cudaFree(npwq);
  cudaFree(idx); cudaFree(inv);
  (void)nst_fdx;
  return R;
}

static void cmp(const std::vector<C2<double>>& a, const std::vector<C2<double>>& b, double& mx, double& l2) {
  double d = 0, dm = 0, s = 0, sm = 0;
  for (size_t i = 0; i < a.size(); i++) {
    double dx = a[i].x - b[i].x, dy = a[i].y - b[i].y, ax = hypot(a[i].x, a[i].y);
    d += dx * dx + dy * dy; dm = std::max(dm, hypot(dx, dy)); s += ax * ax; sm = std::max(sm, ax);
  }
  mx = dm / sm; l2 = sqrt(d / s);
}

int main(int argc, char** argv) {
  const char* sys = argc > 1 ? argv[1] : "B1";
  int nl = argc > 2 ? atoi(argv[2]) : 1024;
  int nslot = argc > 3 ? atoi(argv[3]) : 8;
  int nrep = argc > 4 ? atoi(argv[4]) : 20;
  int nksq = argc > 5 ? atoi(argv[5]) : (!strcmp(sys, "B1") ? 666 : 189);
  const bool dopf = !getenv("NOPF");
  Geom G = make_geom(sys, nksq, 12345);
  int nbnd = nslot;
  if ((nl + nslot - 1) / nslot > G.nksq) { printf("too many columns for nksq\n"); return 1; }
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  // data
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(-0.5, 0.5);
  std::vector<double> hx((size_t)2 * G.npwx * nl, 0.0), hv(G.nnr), hg2((size_t)G.npwx * G.nksq), het((size_t)nbnd * G.nksq);
  std::vector<int> hlist(nl);
  for (int c = 0; c < nl; c++) {
    hlist[c] = c + 1;   // column c: k = c / nslot, band c % nslot
    int k = c / nslot;
    for (int ig = 0; ig < G.npwq[k]; ig++) { hx[2 * ((size_t)G.npwx * c + ig)] = U(rng); hx[2 * ((size_t)G.npwx * c + ig) + 1] = U(rng); }
  }
  for (auto& w : hv) w = U(rng);
  for (auto& w : hg2) w = 5 * (U(rng) + 0.5);
  for (auto& w : het) w = U(rng) + 0.5;
  if (getenv("FDX_VONLY")) {   // potential part alone (g2 = e = 0): the undiluted error of the transforms
    std::fill(hg2.begin(), hg2.end(), 0.0); std::fill(het.begin(), het.end(), 0.0);
    printf("g2 = e = 0: y = (1/N) FFT[V FFT^-1 x] only\n");
  }
  // geometry for pfft and fftdx
  int nst = 0, nxa = 0, pc = 0;
  int ier = fdx_setup(G.n, G.n, G.nksq, G.npwx, G.npwq.data(), G.idx.data(), 1024);
  if (ier) { printf("fdx_setup ierr %d\n", ier); return 1; }
#ifdef FDX_EMU
  if (getenv("FDX_EMU")) { fdx_set_emu(atoi(getenv("FDX_EMU"))); printf("FP32 instance emulates mode %s (1 TF32 dense DFT, 2 FP16, 3 BF16)\n", getenv("FDX_EMU")); }
#endif
  fdx_info(&nst, &nxa, &pc);
  int nsph = 0;
  for (int k = 0; k < G.nksq; k++) nsph += G.npwq[k];
  printf("%s grid %dx%dx%d  nksq %d  npwx %d  <npw> %.0f (sphere %.4f of box)  sticks %d of %d (%.3f)  occupied x %d of %d"
         "  chunk %d  cols %d  GPU %s\n", sys, G.n[0], G.n[1], G.n[2], G.nksq, G.npwx, (double)nsph / G.nksq,
         G.box_frac_sphere, nst, G.n[0] * G.n[1], (double)nst / (G.n[0] * G.n[1]), nxa, G.n[0], pc, nl, prop.name);
  for (int prec : {8, 4}) {
    if (dopf) {
      pfw_setup(prec, G.n, G.n, G.nksq, G.npwx, G.npwq.data(), G.idx.data(), 1024, &ier);
      if (ier) { printf("pfft setup ierr %d\n", ier); return 1; }
    }
    Res R = prec == 8 ? run_prec<double>(G, nl, nslot, nrep, nbnd, hx, hv, hg2, het, hlist, dopf, nst)
                      : run_prec<float>(G, nl, nslot, nrep, nbnd, hx, hv, hg2, het, hlist, dopf, nst);
    static std::vector<C2<double>> ref;   // FP64 cuFFT chain
    if (prec == 8) ref = R.y[0];
    // modelled bytes per column (B per element = 16 FP64 / 8 FP32)
    const double eb = prec == 8 ? 16 : 8, rb = prec;
    const double box = (double)G.nnr;
    const double bytesA = box * eb * (1 + 2 * 3 * 2 + 2) + box * 4 + box * rb;   // scatter w, 2 FFT x 3 passes r+w, vmul r+w; inv; V
    const double nz = (double)nst * G.n[2];
    const double bytesD = nz * eb * 4 + nz * 4 * 2 + (double)nsph / G.nksq * (eb * 3 + rb);   // S w,r,w,r; zmap x2; x x2, y, g2
    int occ[6] = {0};
    fdx_occupancy(prec, occ);
    printf("\n%s  (us per column application; modelled bytes A %.2f MB, D %.2f MB)\n", prec == 8 ? "FP64" : "FP32",
           bytesA / 1e6, bytesD / 1e6);
    const char* nm[3] = {"A cuFFT chain", "P pfft", "D fftdx"};
    for (int v = 0; v < 3; v++) {
      if (v == 1 && !dopf) continue;
      double mx, l2, mxr, l2r;
      cmp(R.y[0], R.y[v], mx, l2);
      cmp(ref, R.y[v], mxr, l2r);
      double bw = (v == 0 ? bytesA : bytesD) / (R.us[v] * 1e-6) / 1e9;
      printf("  %-14s %8.2f us  x%5.2f vs A   %7.0f GB/s   vs A(same prec) max %.2e L2 %.2e   vs A64 max %.2e L2 %.2e\n",
             nm[v], R.us[v], R.us[0] / R.us[v], v == 1 ? 0.0 : bw, mx, l2, mxr, l2r);
    }
    printf("  fftdx occupancy (warps/SM of %d, regs): kz_inv %d/%d  kxy %d/%d  kz_fwd %d/%d\n",
           prop.maxThreadsPerMultiProcessor / 32, occ[0], occ[1], occ[2], occ[3], occ[4], occ[5]);
  }
  return 0;
}
