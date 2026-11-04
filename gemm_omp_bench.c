/* GEMM OpenMP instrumentado - Linux.
 * Uso: gemm_omp_bench N THREADS REPS SCHED CHUNK [saida.csv]
 *   SCHED = static | dynamic | guided ; CHUNK = 0 usa o padrao do OpenMP
 * Faz 1 execucao de aquecimento (descartada) e REPS medicoes de cada variante.
 * Metricas por medicao, todas sobre O MESMO trecho (so a multiplicacao):
 *   tempo (parede), tempo de CPU, MFLOPS, utilizacao de CPU (cpu/(parede*threads))
 *   e ENERGIA (J) do pacote, lida do contador RAPL (powercap) antes e depois.
 * Ao final, verificacao completa contra a referencia sequencial.
 * Energia: le $RAPL_PATH/energy_uj (padrao /sys/class/powercap/intel-rapl:0).
 *   Se o arquivo nao for legivel (ou RAPL_PATH for vazio), grava nan.
 * CSV: versao,n,threads,sched,chunk,rep,tempo_s,cpu_s,mflops,util,energia_j
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

static double wall_now (void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

/* tempo de CPU acumulado do processo (todas as threads) */
static double cpu_now (void) {
  struct timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

/* ---- energia: contador RAPL do pacote (microjoules, cumulativo) ---- */
static char rapl_energy[512];
static unsigned long long rapl_max = 0;
static int rapl_ok = 0;

static int rapl_ler (unsigned long long *uj) {
  FILE *f = fopen(rapl_energy, "r");
  if (!f) return 0;
  int ok = fscanf(f, "%llu", uj) == 1;
  fclose(f);
  return ok;
}

static void rapl_init (void) {
  const char *dir = getenv("RAPL_PATH");
  if (!dir) dir = "/sys/class/powercap/intel-rapl:0";
  if (dir[0] == '\0') return;                 /* energia desligada de proposito */
  snprintf(rapl_energy, sizeof rapl_energy, "%s/energy_uj", dir);
  char p[600];
  snprintf(p, sizeof p, "%s/max_energy_range_uj", dir);
  FILE *f = fopen(p, "r");
  if (f) { if (fscanf(f, "%llu", &rapl_max) != 1) rapl_max = 0; fclose(f); }
  unsigned long long e;
  rapl_ok = rapl_ler(&e);
  if (!rapl_ok) fprintf(stderr, "[aviso] RAPL ilegivel em %s: energia sera nan\n", rapl_energy);
}

/* diferenca em joules, tratando a volta do contador */
static double rapl_delta_j (unsigned long long e0, unsigned long long e1) {
  unsigned long long d = e1 >= e0 ? e1 - e0 : (rapl_max ? e1 + rapl_max - e0 : 0);
  return d / 1e6;
}

/* schedule(runtime): o escalonamento e escolhido por omp_set_schedule */
void gemm_ijk (int n, const double *A, const double *B, double *C) {
  int i, j, k;
  double sum;
  #pragma omp parallel for private(j, k, sum) schedule(runtime)
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
  #pragma omp parallel for schedule(runtime)
  for (i = 0; i < n * n; i++) {
    C[i] = 0.0;
  }
  #pragma omp parallel for private(k, j, r) schedule(runtime)
  for (i = 0; i < n; i++) {
    for (k = 0; k < n; k++) {
      r = A[i * n + k];
      for (j = 0; j < n; j++) {
        C[i * n + j] += r * B[k * n + j];
      }
    }
  }
}

/* referencia sequencial (ikj) para verificar a correcao do paralelo */
static void gemm_ref (int n, const double *A, const double *B, double *C) {
  memset(C, 0, (size_t) n * n * sizeof(double));
  for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
      double r = A[i * n + k];
      for (int j = 0; j < n; j++) C[i * n + j] += r * B[k * n + j];
    }
}

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
  if (argc < 6) {
    fprintf(stderr, "Uso: %s N THREADS REPS SCHED CHUNK [saida.csv]\n", argv[0]);
    return 1;
  }
  int n = atoi(argv[1]), threads = atoi(argv[2]), reps = atoi(argv[3]);
  const char *sched = argv[4];
  int chunk = atoi(argv[5]);
  const char *saida = argc > 6 ? argv[6] : "resultados_omp.csv";

  omp_sched_t kind = omp_sched_static;
  if (!strcmp(sched, "dynamic")) kind = omp_sched_dynamic;
  else if (!strcmp(sched, "guided")) kind = omp_sched_guided;
  omp_set_num_threads(threads);
  omp_set_schedule(kind, chunk);

  size_t bytes = (size_t) n * n * sizeof(double), i;
  double *A = malloc(bytes), *B = malloc(bytes), *C1 = malloc(bytes), *C2 = malloc(bytes), *R = malloc(bytes);
  if (!A || !B || !C1 || !C2 || !R) { fprintf(stderr, "Erro de alocacao!\n"); return 1; }
  for (i = 0; i < (size_t) n * n; i++) {
    A[i] = (double)(i % 100) / 10.0;
    B[i] = (double)((i * 2) % 100) / 10.0;
  }
  gemm_ref(n, A, B, R);
  rapl_init();

  FILE *f = fopen(saida, "a");
  if (!f) { fprintf(stderr, "Erro ao abrir %s\n", saida); return 1; }
  fseek(f, 0, SEEK_END);
  if (ftell(f) == 0) fprintf(f, "versao,n,threads,sched,chunk,rep,tempo_s,cpu_s,mflops,util,energia_j\n");
  double flops = 2.0 * n * (double) n * n;

  gemm_ijk(n, A, B, C1);  /* aquecimento */
  gemm_ikj(n, A, B, C2);

  for (int v = 0; v < 2; v++) {
    for (int r = 0; r < reps; r++) {
      unsigned long long e0 = 0, e1 = 0;
      int lido = rapl_ok && rapl_ler(&e0);
      double w0 = wall_now(), c0 = cpu_now();
      if (v == 0) gemm_ijk(n, A, B, C1); else gemm_ikj(n, A, B, C2);
      double w = wall_now() - w0, c = cpu_now() - c0;
      lido = lido && rapl_ler(&e1);
      char en[32];
      if (lido) snprintf(en, sizeof en, "%.6f", rapl_delta_j(e0, e1)); else snprintf(en, sizeof en, "nan");
      fprintf(f, "%s,%d,%d,%s,%d,%d,%.6f,%.6f,%.2f,%.4f,%s\n",
              v == 0 ? "ijk" : "ikj", n, threads, sched, chunk, r, w, c,
              flops / w / 1e6, c / w / threads, en);
    }
  }
  fclose(f);
  int ok = validar(n, C1, R) && validar(n, C2, R);
  printf("omp n=%d threads=%d sched=%s chunk=%d %s\n", n, threads, sched, chunk, ok ? "ok" : "FALHOU");
  free(A); free(B); free(C1); free(C2); free(R);
  return ok ? 0 : 2;
}
