# -*- coding: utf-8 -*-
"""Protocolo experimental (T1 - PCD, secao 9) - versao Linux com energia medida.

Compila as versoes sequencial e OpenMP instrumentadas e executa:
  A) varredura de threads   (N fixo, escalonamento static)
  B) comparacao de escalonamentos (static/dynamic/guided, varios chunks)
  C) varredura do tamanho da matriz (seq, 1 thread e todos os threads)

Cada configuracao gera RODADAS x REPS medicoes (apos 1 execucao de aquecimento
por processo). A ordem das configuracoes e embaralhada a cada rodada para diluir
efeitos de temperatura/turbo.

ENERGIA: medida DIRETAMENTE no codigo C, com o contador RAPL do pacote
(/sys/class/powercap/intel-rapl:0/energy_uj), lido imediatamente antes e depois
do MESMO trecho cronometrado (so a multiplicacao). Nao ha estimativa por TDP.
Antes de cada grupo o script mede a potencia do sistema ocioso (OCIOSO_S segundos),
usada na analise para descontar a energia basal.

Permissao: desde o kernel 5.10 o energy_uj so e legivel por root. Libere a leitura:
    sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
(vale ate reiniciar). Para usar outro dominio/caminho: RAPL_PATH=/caminho python3 ...
Sem RAPL, rode com --sem-energia (so desempenho/localidade; energia = nan).

Uso:
    python3 run_experimentos.py                  # protocolo completo
    python3 run_experimentos.py --teste-rapido   # roda em ~1 min para validar o ambiente
    python3 run_experimentos.py --sem-energia
Saida: resultados/*.csv, resultados/ambiente.txt, resultados/ambiente.json, resultados/ocioso.csv
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
SEQ = BASE / "gemm_seq_bench"
OMP = BASE / "gemm_omp_bench"
FLAGS = ["-O3", "-Wall"]
RAPL = Path(os.environ.get("RAPL_PATH", "/sys/class/powercap/intel-rapl:0"))

RODADAS = 3
REPS = 5              # RODADAS x REPS medicoes por configuracao
N_THREADS = 1024      # tamanho da varredura de threads
TAM_TAMANHOS = [128, 256, 512, 1024]
SCHEDS = [("static", 0), ("static", 1), ("static", 16), ("dynamic", 1),
          ("dynamic", 16), ("dynamic", 64), ("guided", 0), ("guided", 16)]
OCIOSO_S = 10         # duracao da medicao de potencia ociosa antes de cada grupo
PAUSA_S = 2           # pausa entre execucoes (deixa a temperatura assentar)


# ------------------------------------------------------------------ hardware
def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def topologia():
    """(nucleos_fisicos, nucleos_logicos) detectados na maquina."""
    logicos = os.cpu_count() or 1
    pares = set()
    for linha in sh(["lscpu", "-p=CORE,SOCKET"]).splitlines():
        if linha and not linha.startswith("#"):
            pares.add(tuple(linha.split(",")[:2]))
    return (len(pares) or logicos), logicos


FISICOS, LOGICOS = topologia()
THREADS = sorted({t for t in (1, 2, 3, 4, 6, 8, 12, 16, FISICOS, LOGICOS) if t <= LOGICOS})
NT_FIXO = LOGICOS
NT_SCHED = sorted({FISICOS, LOGICOS})


# --------------------------------------------------------------------- RAPL
def rapl_ler_uj():
    try:
        return int((RAPL / "energy_uj").read_text())
    except (OSError, ValueError):
        return None


def rapl_max_uj():
    try:
        return int((RAPL / "max_energy_range_uj").read_text())
    except (OSError, ValueError):
        return 0


def medir_ocioso(segundos):
    """Potencia media (W) do pacote com o sistema em repouso."""
    e0 = rapl_ler_uj()
    t0 = time.perf_counter()
    time.sleep(segundos)
    e1 = rapl_ler_uj()
    dt = time.perf_counter() - t0
    if e0 is None or e1 is None:
        return float("nan")
    d = e1 - e0 if e1 >= e0 else e1 + rapl_max_uj() - e0
    return d / 1e6 / dt


def verificar_rapl():
    if rapl_ler_uj() is not None:
        return
    print(f"ERRO: nao foi possivel ler {RAPL / 'energy_uj'}.\n"
          "  1) Veja se existe: ls /sys/class/powercap/\n"
          "  2) Libere a leitura: sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj\n"
          "  3) Se o caminho for outro: RAPL_PATH=/sys/class/powercap/<dominio> python3 run_experimentos.py\n"
          "  4) Sem RAPL no equipamento: python3 run_experimentos.py --sem-energia")
    sys.exit(1)


# ----------------------------------------------------------------- ambiente
def ler_texto(caminho):
    try:
        return Path(caminho).read_text().strip()
    except OSError:
        return "n/d"


def registrar_ambiente(com_energia):
    modelo = next((l.split(":", 1)[1].strip() for l in sh(["lscpu"]).splitlines()
                   if l.startswith("Model name")), platform.processor())
    mem_kb = next((int(l.split()[1]) for l in ler_texto("/proc/meminfo").splitlines()
                   if l.startswith("MemTotal")), 0)
    gcc = (sh(["gcc", "--version"]).splitlines() or ["n/d"])[0]
    ac = [ler_texto(p) for p in sorted(Path("/sys/class/power_supply").glob("A*/online"))]
    linhas = [
        f"SO: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"Compilador: {gcc}",
        f"Flags: {' '.join(FLAGS)} -fopenmp",
        f"CPU: {modelo}",
        f"Nucleos fisicos / logicos: {FISICOS} / {LOGICOS}",
        f"Memoria: {mem_kb / 1048576:.1f} GiB",
        f"Governor de frequencia: {ler_texto('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')}",
        f"Fonte AC conectada (1=sim): {','.join(ac) or 'n/d'}",
        f"Energia: {'RAPL pacote em ' + str(RAPL) if com_energia else 'nao medida'}",
        f"Protocolo: {RODADAS} rodadas x {REPS} reps; ocioso {OCIOSO_S}s por grupo; pausa {PAUSA_S}s",
    ]
    (RES / "ambiente.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    (RES / "ambiente.json").write_text(json.dumps(
        {"nucleos_fisicos": FISICOS, "nucleos_logicos": LOGICOS, "energia_rapl": com_energia,
         "cpu": modelo}, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- execucao
def compilar():
    subprocess.run(["gcc", *FLAGS, str(BASE / "gemm_seq_bench.c"), "-o", str(SEQ), "-lm"], check=True)
    subprocess.run(["gcc", *FLAGS, "-fopenmp", str(BASE / "gemm_omp_bench.c"), "-o", str(OMP), "-lm"], check=True)


def executar(grupo, configs, env, com_energia):
    """configs: lista de (rotulo, comando). Embaralha a cada rodada."""
    if com_energia:
        p = medir_ocioso(OCIOSO_S)
        print(f"[{grupo}] potencia ociosa: {p:.2f} W", flush=True)
        novo = not (RES / "ocioso.csv").exists()
        with open(RES / "ocioso.csv", "a", encoding="utf-8") as f:
            if novo:
                f.write("grupo,p_ocioso_w\n")
            f.write(f"{grupo},{p:.4f}\n")
    for rodada in range(RODADAS):
        ordem = configs[:]
        random.shuffle(ordem)
        for i, (rotulo, cmd) in enumerate(ordem, 1):
            print(f"[{grupo}] rodada {rodada + 1}/{RODADAS} ({i}/{len(ordem)}) {rotulo}", flush=True)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=env)
            time.sleep(PAUSA_S)


def main():
    global RODADAS, REPS, N_THREADS, TAM_TAMANHOS, SCHEDS, OCIOSO_S, PAUSA_S, THREADS, NT_SCHED
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-energia", action="store_true", help="nao le o RAPL (energia = nan)")
    ap.add_argument("--teste-rapido", action="store_true", help="protocolo reduzido para validar o ambiente")
    args = ap.parse_args()
    com_energia = not args.sem_energia
    if args.teste_rapido:
        RODADAS, REPS, N_THREADS, TAM_TAMANHOS = 1, 2, 256, [128, 256]
        SCHEDS = [("static", 0), ("dynamic", 16)]
        THREADS = sorted({1, FISICOS, LOGICOS})
        NT_SCHED = [LOGICOS]
        OCIOSO_S, PAUSA_S = 3, 0
    if com_energia:
        verificar_rapl()

    env = os.environ.copy()
    if not com_energia:
        env["RAPL_PATH"] = ""          # o C grava nan, sem avisos

    RES.mkdir(exist_ok=True)
    for f in RES.glob("*.csv"):
        f.unlink()
    compilar()
    registrar_ambiente(com_energia)
    random.seed(42)

    # A) threads
    arq = str(RES / "threads.csv")
    cfg = [("seq", [str(SEQ), str(N_THREADS), str(REPS), arq])]
    for t in THREADS:
        cfg.append((f"omp t={t}", [str(OMP), str(N_THREADS), str(t), str(REPS), "static", "0", arq]))
    executar("A-threads", cfg, env, com_energia)

    # B) escalonamento
    arq = str(RES / "escalonamento.csv")
    cfg = []
    for t in NT_SCHED:
        for s, c in SCHEDS:
            cfg.append((f"t={t} {s},{c}", [str(OMP), str(N_THREADS), str(t), str(REPS), s, str(c), arq]))
    executar("B-sched", cfg, env, com_energia)

    # C) tamanho
    arq = str(RES / "tamanhos.csv")
    cfg = []
    for n in TAM_TAMANHOS:
        reps = REPS * 4 if n <= 256 else REPS   # entradas pequenas: mais repeticoes
        cfg.append((f"seq n={n}", [str(SEQ), str(n), str(reps), arq]))
        for t in sorted({1, NT_FIXO}):
            cfg.append((f"omp n={n} t={t}", [str(OMP), str(n), str(t), str(reps), "static", "0", arq]))
    executar("C-tamanhos", cfg, env, com_energia)
    print("Concluido.")


if __name__ == "__main__":
    main()
