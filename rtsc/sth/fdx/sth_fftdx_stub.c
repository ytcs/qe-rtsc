/* rtsc: stand-in for sth_fftdx.cu when the build has no MathDx/CUTLASS headers. fdx_setup reports
   "unsupported" (99), so the engine keeps pfft / cuFFT and the other entry points are never reached. */
#include <stddef.h>
int fdx_setup(const int* nr, const int* nrx, int nksq, int npwx, const int* npwq, const int* idxkq, int chunk) {
  (void)nr; (void)nrx; (void)nksq; (void)npwx; (void)npwq; (void)idxkq; (void)chunk;
  return 99;
}
void fdx_free(void) {}
void fdx_info(int* nst, int* nxa, int* chunk) { *nst = 0; *nxa = 0; *chunk = 0; }
int fdx_local_d(void) { return 99; }
int fdx_local_s(void) { return 99; }
int fdx_dpot_d(void) { return 99; }
int fdx_inv_d(void) { return 99; }
int fdx_inv_rho_d(void) { return 99; }
int fdx_geom1(void) { return 99; }
