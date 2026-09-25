# -*- coding: utf-8 -*-
"""Protocolo experimental (T1 - PCD, secao 9).

Compila as versoes sequencial e OpenMP instrumentadas e executa:
  A) varredura de threads   (N fixo, escalonamento static)
  B) comparacao de escalonamentos (static/dynamic/guided, varios chunks)
  C) varredura do tamanho da matriz (seq, 1 thread e todos os threads)

Cada configuracao gera 10 medicoes (2 rodadas x 5 repeticoes, apos 1 execucao
de aquecimento por processo). A ordem das configuracoes e embaralhada a cada
rodada para diluir efeitos de temperatura/turbo.

Uso: python run_experimentos.py
Saida: resultados/*.csv, resultados/ambiente.txt
"""

import random
import subprocess
import sys
import platform
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
EXE = ".exe" if sys.platform == "win32" else ""
SEQ = BASE / f"gemm_seq_bench{EXE}"
OMP = BASE / f"gemm_omp_bench{EXE}"
FLAGS = ["-O3", "-Wall"]

RODADAS = 2
REPS = 5            # 2 x 5 = 10 medicoes por configuracao
N_THREADS = 1024    # tamanho da varredura de threads
THREADS = [1, 2, 3, 4, 6, 8, 12, 16, 24]
NT_FIXO = 12        # nucleos logicos da maquina de teste
TAM_TAMANHOS = [128, 256, 512, 1024]
SCHEDS = [("static", 0), ("static", 1), ("static", 16), ("dynamic", 1),
          ("dynamic", 16), ("dynamic", 64), ("guided", 0), ("guided", 16)]
NT_SCHED = [6, 12]


def compilar():
    subprocess.run(["gcc", *FLAGS, str(BASE / "gemm_seq_bench.c"), "-o", str(SEQ), "-lm"], check=True)
    subprocess.run(["gcc", *FLAGS, "-fopenmp", str(BASE / "gemm_omp_bench.c"), "-o", str(OMP), "-lm"], check=True)


def registrar_ambiente():
    gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    linhas = [f"SO: {platform.platform()}", f"Python: {platform.python_version()}",
              f"Compilador: {gcc}", f"Flags: {' '.join(FLAGS)} -fopenmp",
              f"Processador: {platform.processor()}",
              "CPU: AMD Ryzen 5 5600X, 6 nucleos / 12 threads, 4.2 GHz (base), 32 GB RAM"]
    (RES / "ambiente.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def executar(grupo, configs):
    """configs: lista de (rotulo, comando, csv). Embaralha a cada rodada."""
    for rodada in range(RODADAS):
        ordem = configs[:]
        random.shuffle(ordem)
        for i, (rotulo, cmd) in enumerate(ordem, 1):
            print(f"[{grupo}] rodada {rodada + 1}/{RODADAS} ({i}/{len(ordem)}) {rotulo}", flush=True)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    RES.mkdir(exist_ok=True)
    for f in RES.glob("*.csv"):
        f.unlink()
    compilar()
    registrar_ambiente()
    random.seed(42)

    # A) threads
    csv = str(RES / "threads.csv")
    cfg = [("seq", [str(SEQ), str(N_THREADS), str(REPS), csv])]
    for t in THREADS:
        cfg.append((f"omp t={t}", [str(OMP), str(N_THREADS), str(t), str(REPS), "static", "0", csv]))
    executar("A-threads", cfg)

    # B) escalonamento
    csv = str(RES / "escalonamento.csv")
    cfg = []
    for t in NT_SCHED:
        for s, c in SCHEDS:
            cfg.append((f"t={t} {s},{c}", [str(OMP), str(N_THREADS), str(t), str(REPS), s, str(c), csv]))
    executar("B-sched", cfg)

    # C) tamanho
    csv = str(RES / "tamanhos.csv")
    cfg = []
    for n in TAM_TAMANHOS:
        reps = REPS * 4 if n <= 256 else REPS   # entradas pequenas: mais repeticoes
        cfg.append((f"seq n={n}", [str(SEQ), str(n), str(reps), csv]))
        for t in (1, NT_FIXO):
            cfg.append((f"omp n={n} t={t}", [str(OMP), str(n), str(t), str(reps), "static", "0", csv]))
    executar("C-tamanhos", cfg)
    print("Concluido.")


if __name__ == "__main__":
    main()
