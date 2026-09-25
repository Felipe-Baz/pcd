# -*- coding: utf-8 -*-
"""Sequencial ijk vs ikj

Versao local (fora do Colab): compila gemm_test.c com gcc e executa
via subprocess para varios tamanhos de matriz, depois le o CSV
gerado e plota os resultados.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
FONTE_C = BASE_DIR / "gemm_test.c"
EXECUTAVEL = BASE_DIR / ("gemm_test.exe" if sys.platform == "win32" else "gemm_test")
CSV_RESULTADOS = BASE_DIR / "resultados_iniciais.csv"
TAMANHOS = [128, 256, 512, 1500]


def compilar():
    subprocess.run(
        ["gcc", "-O3", "-Wall", str(FONTE_C), "-o", str(EXECUTAVEL), "-lm"],
        check=True,
    )


def rodar_testes():
    if CSV_RESULTADOS.exists():
        CSV_RESULTADOS.unlink()
    for tamanho in TAMANHOS:
        subprocess.run([str(EXECUTAVEL), str(tamanho)], cwd=BASE_DIR, check=True)


def carregar_resultados():
    colunas = ["Ordem", "Tamanho", "Tempo_s", "MFLOPS"]
    return pd.read_csv(CSV_RESULTADOS, names=colunas)


def plotar(tabela):
    tabela_mediana = tabela.groupby(["Ordem", "Tamanho"]).median().reset_index()

    df_ijk = tabela_mediana[tabela_mediana["Ordem"] == "ijk"]
    df_ikj = tabela_mediana[tabela_mediana["Ordem"] == "ikj"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df_ijk["Tamanho"], df_ijk["Tempo_s"], marker='o', color='#e74c3c', label="Caótico (ijk)")
    plt.plot(df_ikj["Tamanho"], df_ikj["Tempo_s"], marker='o', color='#2ecc71', label="Organizado (ikj)")
    plt.title("Tempo de Execução pela Mediana")
    plt.xlabel("Tamanho da Matriz")
    plt.ylabel("Tempo (segundos)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_ijk["Tamanho"], df_ijk["MFLOPS"], marker='o', color='#c0392b', label="Caótico (ijk)")
    plt.plot(df_ikj["Tamanho"], df_ikj["MFLOPS"], marker='o', color='#27ae60', label="Organizado (ikj)")
    plt.title("Velocidade (MFLOPS) pela Mediana")
    plt.xlabel("Tamanho da Matriz")
    plt.ylabel("MFLOPS")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compilar()
    rodar_testes()
    tabela = carregar_resultados()
    print(tabela)
    plotar(tabela)
