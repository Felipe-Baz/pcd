/* GEMM sequencial instrumentado (baseline).
 * Uso: gemm_seq_bench N REPS [saida.csv]
 * Faz 1 execucao de aquecimento (descartada) e REPS medicoes de cada variante.
 * Metricas por medicao: tempo (parede), tempo de CPU, MFLOPS, utilizacao de CPU.
 * CSV: versao,n,threads,sched,chunk,rep,tempo_s,cpu_s,mflops,util
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif

static double wall_now (void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

/* tempo de CPU acumulado do processo (usuario + kernel) */
static double cpu_now (void) {
#ifdef _WIN32
  FILETIME c, e, k, u;
  GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
  ULONGLONG kk = ((ULONGLONG)k.dwHighDateTime << 32) | k.dwLowDateTime;
  ULONGLONG uu = ((ULONGLONG)u.dwHighDateTime << 32) | u.dwLowDateTime;
  return (double)(kk + uu) * 100e-9;
#else
  struct timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
#endif
}

void gemm_ijk (int n, const double *A, const double *B, double *C) {
  int i, j, k;
  double sum;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      sum = 0.0;
      for (k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

void gemm_ikj (int n, const double *A, const double *B, double *C) {
  int i, j, k;
  double r;
  for (i = 0; i < n * n; i++) {
    C[i] = 0.0;
  }
  for (i = 0; i < n; i++) {
    for (k = 0; k < n; k++) {
      r = A[i * n + k];
      for (j = 0; j < n; j++) {
        C[i * n + j] += r * B[k * n + j];
      }
    }
  }
}

/* compara as n*n posicoes (erro relativo) */
int validar (int n, const double *C1, const double *C2) {
  size_t i, total = (size_t) n * n;
  for (i = 0; i < total; i++) {
    double d = fabs(C1[i] - C2[i]);
    double m = fmax(fabs(C1[i]), fabs(C2[i]));
    if (d > 1e-9 * (m > 1.0 ? m : 1.0)) {
      fprintf(stderr, "[ERRO] Divergencia no indice %zu: %f vs %f\n", i, C1[i], C2[i]);
      return 0;
    }
  }
  return 1;
}

int main (int argc, char *argv[]) {
  int n = argc > 1 ? atoi(argv[1]) : 1024;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  const char *saida = argc > 3 ? argv[3] : "resultados_seq.csv";
  size_t bytes = (size_t) n * n * sizeof(double), i;
  double *A = malloc(bytes), *B = malloc(bytes), *C1 = malloc(bytes), *C2 = malloc(bytes);
  if (!A || !B || !C1 || !C2) { fprintf(stderr, "Erro de alocacao!\n"); return 1; }
  for (i = 0; i < (size_t) n * n; i++) {
    A[i] = (double)(i % 100) / 10.0;
    B[i] = (double)((i * 2) % 100) / 10.0;
  }

  FILE *f = fopen(saida, "a");
  if (!f) { fprintf(stderr, "Erro ao abrir %s\n", saida); return 1; }
  fseek(f, 0, SEEK_END);
  if (ftell(f) == 0) fprintf(f, "versao,n,threads,sched,chunk,rep,tempo_s,cpu_s,mflops,util\n");
  double flops = 2.0 * n * (double) n * n;

  gemm_ijk(n, A, B, C1);  /* aquecimento */
  gemm_ikj(n, A, B, C2);
  if (!validar(n, C1, C2)) return 2;

  for (int v = 0; v < 2; v++) {
    for (int r = 0; r < reps; r++) {
      double w0 = wall_now(), c0 = cpu_now();
      if (v == 0) gemm_ijk(n, A, B, C1); else gemm_ikj(n, A, B, C2);
      double w = wall_now() - w0, c = cpu_now() - c0;
      fprintf(f, "%s,%d,1,seq,0,%d,%.6f,%.6f,%.2f,%.4f\n",
              v == 0 ? "ijk" : "ikj", n, r, w, c, flops / w / 1e6, c / w);
    }
  }
  fclose(f);
  printf("seq n=%d ok\n", n);
  free(A); free(B); free(C1); free(C2);
  return 0;
}
