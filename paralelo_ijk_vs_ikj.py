# -*- coding: utf-8 -*-
"""Paralelo ijk vs ikj (OpenMP)

Versao local (fora do Colab): compila gemm_test_omp.c com gcc
(-fopenmp) e executa via subprocess, definindo OMP_NUM_THREADS
no ambiente do processo.
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FONTE_C = BASE_DIR / "gemm_test_omp.c"
EXECUTAVEL = BASE_DIR / ("gemm_test_omp.exe" if sys.platform == "win32" else "gemm_test_omp")
TAMANHO = 1500
NUM_THREADS = 2


def compilar():
    subprocess.run(
        ["gcc", "-O3", "-Wall", "-fopenmp", str(FONTE_C), "-o", str(EXECUTAVEL), "-lm"],
        check=True,
    )


def rodar_teste():
    ambiente = os.environ.copy()
    ambiente["OMP_NUM_THREADS"] = str(NUM_THREADS)
    subprocess.run(
        [str(EXECUTAVEL), str(TAMANHO)],
        cwd=BASE_DIR,
        env=ambiente,
        check=True,
    )


if __name__ == "__main__":
    compilar()
    rodar_teste()
