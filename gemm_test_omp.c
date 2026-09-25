#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#define _POSIX_C_SOURCE 199309L

void info_threads (void) {
  int num_procs = omp_get_num_procs();
  int max_threads = omp_get_max_threads();
  printf("Nucleos/processadores disponiveis: %d\n", num_procs);
  printf("Threads disponiveis (OpenMP): %d\n", max_threads);
}

void gemm_ijk (int n, const double *A, const double *B, double *C) {
	int i, j, k;
	double sum;
	#pragma omp parallel for private(j, k, sum)
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
	#pragma omp parallel for private(k, j, r)
	for (i = 0; i < n; i++) {
		for (k = 0; k < n; k++) {
			r = A[i * n + k];
			for (j = 0; j < n; j++) {
				C[i * n + j] += r * B[k * n + j];
			}
		}
	}
}

int validar_resultados (int n, const double *C1, const double *C2) {
	int i;
	double epsilon = 1e-9;
	for (i = 0; i < n; i++) {
		if (fabs(C1[i] - C2[i]) > epsilon) {
			printf("[ERRO] Divergencia no Indice %d: ijk=%f, ikj=%f", i, C1[i], C2[i]);
			return 0;
		}
	}
	return 1;
}

int main (int argc, char *argv[]) {
	int i;
	int n = 1024;
	if (argc > 1) {
		n = atoi(argv[1]);
	}
  info_threads();
	printf ("Iniciando teste com matriz %d x %d...\n", n, n);
  size_t bytes = (size_t) n * n * sizeof(double);
  double *A = (double *) malloc(bytes);
  double *B = (double *) malloc(bytes);
  double *C_ijk = (double *) malloc(bytes);
  double *C_ikj = (double *) malloc(bytes);
	if (!A || !B || !C_ijk || !C_ikj) {
	  fprintf(stderr, "Erro de alocacao de memoria!");
    return 1;
  }
  for (i = 0; i < n * n; i++) {
    A[i] = (double)(i % 100) / 10.0;
    B[i] = (double)((i * 2) % 100) / 10.0;
  }

	//medicao do ijk
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	gemm_ijk(n, A, B, C_ijk);
	clock_gettime(CLOCK_MONOTONIC, &end);
	double tempo_ijk = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  //medicao do ikj
  clock_gettime(CLOCK_MONOTONIC, &start);
  gemm_ikj(n, A, B, C_ikj);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double tempo_ikj = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  //calculo dos MFLOPS
  double flops = 2.0 * (double)n * n * n;
  double mflops_ijk = (flops / tempo_ijk) / 1e6;
  double mflops_ikj = (flops / tempo_ikj) / 1e6;

  //validacao
	if (validar_resultados(n, C_ijk, C_ikj)) {
    printf("[OK] Ambos os algoritmos produziram resultados identicos!\n\n");
  }
  printf("Resultados (N = %d):\n", n);
  printf("ijk: %f s | %.2f MFLOPS\n", tempo_ijk, mflops_ijk);
  printf("ikj: %f s | %.2f MFLOPS\n", tempo_ikj, mflops_ikj);
  printf("Speedup de ikj sobre ijk: %2.fx\n", tempo_ijk / tempo_ikj);
  free(A);
  free(B);
  free(C_ijk);
  free(C_ikj);
  return 0;
}
