# -*- coding: utf-8 -*-
"""Protocolo experimental (T1 - PCD, secao 9).

Compila as versoes sequencial e OpenMP instrumentadas e executa:
  A) varredura de threads   (N fixo, escalonamento static)
  B) comparacao de escalonamentos (static/dynamic/guided, varios chunks)
  C) varredura do tamanho da matriz (seq, 1 thread e todos os threads)

Cada configuracao gera 10 medicoes (2 rodadas x 5 repeticoes, apos 1 execucao
de aquecimento por processo). A ordem das configuracoes e embaralhada a cada
rodada para diluir efeitos de temperatura/turbo.

Cada chamada de processo tambem e monitorada com psutil (utilizacao de CPU do
processo, amostrada em intervalos curtos) e a energia e ESTIMADA por um modelo
simples de TDP: potencia(t) = TDP_W * (uso_processo(t) / 100 / nucleos_logicos);
energia = integral da potencia no tempo. Isso NAO e uma leitura real de sensor
de hardware (Windows/AMD nao expoe um contador de joules por processo sem
drivers extras) - e apenas uma estimativa proporcional ao uso de CPU, util para
comparar configuracoes entre si, nao para obter watts/joules absolutos exatos.

Uso: python run_experimentos.py
Saida: resultados/*.csv, resultados/ambiente.txt, resultados/energia.csv
"""

import csv
import random
import subprocess
import sys
import time
import platform
from pathlib import Path

import psutil

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
EXE = ".exe" if sys.platform == "win32" else ""
SEQ = BASE / f"gemm_seq_bench{EXE}"
OMP = BASE / f"gemm_omp_bench{EXE}"
FLAGS = ["-O3", "-Wall"]

TDP_W = 65.0          # TDP nominal do AMD Ryzen 5 5600X (estimativa de potencia)
NUCLEOS_LOGICOS = 12
AMOSTRA_S = 0.1        # intervalo de amostragem do psutil

RODADAS = 2
REPS = 5            # 2 x 5 = 10 medicoes por configuracao
N_THREADS = 1024    # tamanho da varredura de threads
THREADS = [1, 2, 3, 4, 6, 8, 12]
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


def medir_com_energia(cmd):
    """Executa cmd e monitora com psutil: utilizacao de CPU do processo (amostrada
    a cada AMOSTRA_S) e energia estimada pelo modelo de TDP (ver docstring do
    modulo). Retorna (duracao_s, cpu_pct_medio, util_sistema, potencia_w, energia_j).
    """
    proc = psutil.Popen(cmd, stdout=subprocess.DEVNULL)
    proc.cpu_percent(interval=None)  # descarta a primeira leitura (sempre 0.0)
    amostras = []
    t0 = time.perf_counter()
    while proc.poll() is None:
        try:
            amostras.append(proc.cpu_percent(interval=AMOSTRA_S))
        except psutil.NoSuchProcess:
            break
    duracao = time.perf_counter() - t0
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    cpu_pct_medio = sum(amostras) / len(amostras) if amostras else 0.0
    util_sistema = cpu_pct_medio / (100.0 * NUCLEOS_LOGICOS)
    potencia_w = TDP_W * util_sistema
    energia_j = potencia_w * duracao
    return duracao, cpu_pct_medio, util_sistema, potencia_w, energia_j


def executar(grupo, configs):
    """configs: lista de (rotulo, comando, csv). Embaralha a cada rodada."""
    energia_csv = RES / "energia.csv"
    novo = not energia_csv.exists()
    with open(energia_csv, "a", encoding="utf-8", newline="") as fe:
        w = csv.writer(fe)
        if novo:
            w.writerow(["grupo", "rotulo", "rodada", "duracao_s", "cpu_pct_medio", "util_sistema",
                        "potencia_w_est", "energia_j_est"])
        for rodada in range(RODADAS):
            ordem = configs[:]
            random.shuffle(ordem)
            for i, (rotulo, cmd) in enumerate(ordem, 1):
                print(f"[{grupo}] rodada {rodada + 1}/{RODADAS} ({i}/{len(ordem)}) {rotulo}", flush=True)
                dur, cpu_pct, util, pot, energia = medir_com_energia(cmd)
                w.writerow([grupo, rotulo, rodada + 1, f"{dur:.6f}", f"{cpu_pct:.2f}", f"{util:.4f}",
                            f"{pot:.3f}", f"{energia:.3f}"])
                fe.flush()


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
