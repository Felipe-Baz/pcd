# -*- coding: utf-8 -*-
"""Analise dos resultados: tabelas (CSV/Markdown) e graficos em resultados/graficos/.

Metricas (T1 - PCD, etapas 4 e 5):
  speedup    S(p) = T_seq / T_p           (baseline: versao sequencial da MESMA variante)
  eficiencia E(p) = S(p) / p
  Karp-Flatt e(p) = (1/S - 1/p) / (1 - 1/p)   fracao serial experimental
  Amdahl     S(p) = 1 / (f + (1-f)/p)     ajustada por minimos quadrados
  MFLOPS, MFLOPS por thread, utilizacao de CPU (cpu/(parede*p)),
  custo em nucleo-segundos (tempo de CPU) como indicador indireto de energia.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
GRAF = RES / "graficos"
COLS = ["versao", "n", "threads", "sched", "chunk", "rep", "tempo_s", "cpu_s", "mflops", "util"]
COR = {"ijk": "#d1495b", "ikj": "#2a9d8f"}
NOME = {"ijk": "ijk (acesso a B por coluna)", "ikj": "ikj (acesso sequencial)"}
NUCLEOS_FISICOS, NUCLEOS_LOGICOS = 6, 12

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "legend.frameon": False})


def ler(nome):
    return pd.read_csv(RES / nome)


def agrega(df, chaves):
    g = df.groupby(chaves)
    out = g.agg(tempo_med=("tempo_s", "mean"), tempo_mediana=("tempo_s", "median"),
                tempo_dp=("tempo_s", "std"), tempo_min=("tempo_s", "min"),
                cpu_med=("cpu_s", "mean"), mflops=("mflops", "median"),
                util=("util", "median"), amostras=("tempo_s", "count")).reset_index()
    out["cv_pct"] = 100 * out["tempo_dp"] / out["tempo_med"]
    return out


def amdahl(p, f):
    return 1.0 / (f + (1.0 - f) / p)


def ajusta_amdahl(p, s):
    fs = np.linspace(0, 1, 2001)
    erro = [np.sum((amdahl(p, f) - s) ** 2) for f in fs]
    return fs[int(np.argmin(erro))]


def salvar(fig, nome):
    GRAF.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAF / nome, bbox_inches="tight")
    plt.close(fig)


def marca_ht(ax):
    ax.axvline(NUCLEOS_FISICOS, color="gray", ls=":", lw=1)
    ax.text(NUCLEOS_FISICOS, ax.get_ylim()[1], " 6 núcleos\n físicos", va="top", fontsize=8, color="gray")


# ---------------------------------------------------------------- A) threads
def analise_threads():
    df = ler("threads.csv")
    seq = agrega(df[df.sched == "seq"], ["versao"]).set_index("versao")
    par = agrega(df[df.sched != "seq"], ["versao", "threads"])
    for v in ("ijk", "ikj"):
        m = par.versao == v
        par.loc[m, "speedup"] = seq.loc[v, "tempo_mediana"] / par.loc[m, "tempo_mediana"]
        t1 = par[m & (par.threads == 1)].tempo_mediana.iloc[0]
        par.loc[m, "speedup_vs_1t"] = t1 / par.loc[m, "tempo_mediana"]
    par["eficiencia"] = par.speedup / par.threads
    par["karp_flatt"] = np.where(par.threads > 1,
                                 (1 / par.speedup - 1 / par.threads) / (1 - 1 / par.threads), np.nan)
    par["mflops_por_thread"] = par.mflops / par.threads
    par["nucleo_segundos"] = par.cpu_med
    par.to_csv(RES / "resumo_threads.csv", index=False)
    seq.reset_index().to_csv(RES / "resumo_sequencial.csv", index=False)

    # 1) tempo
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in ("ijk", "ikj"):
        d = par[par.versao == v]
        ax.errorbar(d.threads, d.tempo_med, yerr=d.tempo_dp, marker="o", capsize=3, color=COR[v], label=NOME[v])
        ax.axhline(seq.loc[v, "tempo_med"], color=COR[v], ls="--", lw=1, alpha=0.6)
    ax.set(yscale="log", xlabel="Threads OpenMP", ylabel="Tempo (s) — média ± desvio padrão",
           title="Tempo de execução (N=1024). Tracejado = sequencial")
    ax.set_xticks(par.threads.unique()); ax.legend(); marca_ht(ax)
    salvar(fig, "01_tempo_vs_threads.png")

    # 2) speedup + Amdahl
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, v in zip(axs, ("ijk", "ikj")):
        d = par[par.versao == v]
        f = ajusta_amdahl(d.threads.values, d.speedup.values)
        pp = np.linspace(1, d.threads.max(), 200)
        ax.plot(pp, pp, color="gray", ls="--", label="Ideal (linear)")
        ax.plot(pp, amdahl(pp, f), color="#555", ls="-.", label=f"Amdahl ajustada (f={f:.3f})")
        ax.plot(d.threads, d.speedup, "o-", color=COR[v], label="Medido")
        ax.set(xlabel="Threads", ylabel="Speedup vs sequencial", title=f"Speedup — {NOME[v]}")
        ax.set_xticks(d.threads.unique()); ax.legend(); marca_ht(ax)
    salvar(fig, "02_speedup.png")

    # 3) eficiencia
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in ("ijk", "ikj"):
        d = par[par.versao == v]
        ax.plot(d.threads, d.eficiencia, "o-", color=COR[v], label=NOME[v])
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set(xlabel="Threads", ylabel="Eficiência E = S/p", title="Eficiência paralela")
    ax.set_xticks(par.threads.unique()); ax.legend(); marca_ht(ax)
    salvar(fig, "03_eficiencia.png")

    # 4) MFLOPS
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in ("ijk", "ikj"):
        d = par[par.versao == v]
        axs[0].plot(d.threads, d.mflops, "o-", color=COR[v], label=NOME[v])
        axs[1].plot(d.threads, d.mflops_por_thread, "o-", color=COR[v], label=NOME[v])
        axs[0].axhline(seq.loc[v, "mflops"], color=COR[v], ls="--", lw=1, alpha=0.6)
    axs[0].set(xlabel="Threads", ylabel="MFLOPS", title="Vazão (tracejado = sequencial)")
    axs[1].set(xlabel="Threads", ylabel="MFLOPS por thread", title="Trabalho por thread (proxy de MFLOPS/recurso)")
    for ax in axs:
        ax.set_xticks(par.threads.unique()); ax.legend()
    salvar(fig, "04_mflops.png")

    # 5) recursos: utilizacao e nucleo-segundos
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in ("ijk", "ikj"):
        d = par[par.versao == v]
        axs[0].plot(d.threads, 100 * d.util, "o-", color=COR[v], label=NOME[v])
        axs[1].plot(d.threads, d.nucleo_segundos, "o-", color=COR[v], label=NOME[v])
        axs[1].axhline(seq.loc[v, "cpu_med"], color=COR[v], ls="--", lw=1, alpha=0.6)
    axs[0].set(xlabel="Threads", ylabel="Utilização de CPU (%)", title="cpu_time / (tempo × threads)")
    axs[1].set(xlabel="Threads", ylabel="Tempo de CPU total (núcleo·s)", yscale="log",
               title="Custo em recursos (tracejado = sequencial)")
    for ax in axs:
        ax.set_xticks(par.threads.unique()); ax.legend()
    salvar(fig, "05_recursos_cpu.png")

    # 6) Karp-Flatt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in ("ijk", "ikj"):
        d = par[(par.versao == v) & (par.threads > 1)]
        ax.plot(d.threads, d.karp_flatt, "o-", color=COR[v], label=NOME[v])
    ax.set(xlabel="Threads", ylabel="Fração serial experimental e(p)", title="Métrica de Karp-Flatt")
    ax.set_xticks(par.threads.unique()[1:]); ax.legend()
    salvar(fig, "06_karp_flatt.png")

    # 7) ijk x ikj: localidade de cache (sequencial vs paralelo)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    base = par[par.threads == 1].set_index("versao").tempo_mediana
    razao = par.pivot(index="threads", columns="versao", values="tempo_mediana")
    razao["ijk/ikj"] = razao.ijk / razao.ikj
    ax.bar(razao.index.astype(str), razao["ijk/ikj"], color="#6c757d")
    ax.axhline(seq.loc["ijk", "tempo_mediana"] / seq.loc["ikj", "tempo_mediana"], color="k", ls="--",
               label="sequencial")
    ax.set(xlabel="Threads", ylabel="Tempo ijk / tempo ikj", title="Ganho da ordem ikj (localidade de cache)")
    ax.legend()
    salvar(fig, "07_ijk_vs_ikj.png")
    return par, seq


# -------------------------------------------------------------- B) escalonamento
def analise_sched():
    df = ler("escalonamento.csv")
    df["config"] = df.sched + "," + df.chunk.astype(str).replace("0", "padrão")
    ag = agrega(df, ["versao", "threads", "config"])
    ag.to_csv(RES / "resumo_escalonamento.csv", index=False)
    ordem = list(dict.fromkeys(df.config))
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for j, v in enumerate(("ijk", "ikj")):
        for i, t in enumerate(sorted(df.threads.unique())):
            ax = axs[i, j]
            d = ag[(ag.versao == v) & (ag.threads == t)].set_index("config").loc[ordem]
            ax.bar(range(len(d)), d.tempo_med, yerr=d.tempo_dp, capsize=3, color=COR[v])
            ax.set_xticks(range(len(d))); ax.set_xticklabels(d.index, rotation=35, ha="right")
            ax.set(ylabel="Tempo (s)", title=f"{v} — {t} threads")
    fig.suptitle("Escalonamento OpenMP (schedule, chunk) — carga regular, N=1024", y=1.0)
    fig.tight_layout()
    salvar(fig, "08_escalonamento.png")
    return ag


# ------------------------------------------------------------------ C) tamanhos
def analise_tamanhos():
    df = ler("tamanhos.csv")
    seq = agrega(df[df.sched == "seq"], ["versao", "n"]).set_index(["versao", "n"])
    par = agrega(df[df.sched != "seq"], ["versao", "n", "threads"])
    par["speedup"] = [seq.loc[(r.versao, r.n), "tempo_mediana"] / r.tempo_mediana for r in par.itertuples()]
    par["eficiencia"] = par.speedup / par.threads
    par.to_csv(RES / "resumo_tamanhos.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    tmax = par.threads.max()
    for v in ("ijk", "ikj"):
        d = par[(par.versao == v) & (par.threads == tmax)]
        axs[0].plot(d.n, d.speedup, "o-", color=COR[v], label=f"{NOME[v]}")
        d1 = par[(par.versao == v) & (par.threads == 1)]
        axs[1].plot(d1.n, d1.speedup, "o-", color=COR[v], label=NOME[v])
    axs[0].axhline(tmax, color="gray", ls="--", label=f"ideal ({tmax})")
    axs[0].set(xscale="log", xlabel="N", ylabel="Speedup", title=f"Speedup com {tmax} threads vs tamanho")
    axs[1].axhline(1, color="gray", ls="--")
    axs[1].set(xscale="log", xlabel="N", ylabel="T_seq / T_omp(1 thread)",
               title="Overhead do OpenMP com 1 thread (<1 = mais lento)")
    for ax in axs:
        ax.set_xticks(sorted(par.n.unique())); ax.set_xticklabels(sorted(par.n.unique())); ax.legend()
    salvar(fig, "09_tamanho_da_matriz.png")
    return par


def relatorio(par, seq, ag_sched, tam):
    L = ["# Resumo dos resultados (gerado por analisar.py)\n"]
    L.append("## Sequencial (N=1024)\n")
    L.append(seq.reset_index()[["versao", "tempo_med", "tempo_dp", "cv_pct", "mflops"]].round(4).to_markdown(index=False))
    L.append("\n## Varredura de threads (N=1024, static)\n")
    cols = ["versao", "threads", "tempo_med", "tempo_dp", "cv_pct", "speedup", "eficiencia",
            "karp_flatt", "mflops", "util", "nucleo_segundos"]
    L.append(par[cols].round(3).to_markdown(index=False))
    L.append("\n## Escalonamento\n")
    L.append(ag_sched[["versao", "threads", "config", "tempo_med", "tempo_dp", "mflops"]].round(4).to_markdown(index=False))
    L.append("\n## Tamanhos\n")
    L.append(tam[["versao", "n", "threads", "tempo_med", "speedup", "eficiencia"]].round(3).to_markdown(index=False))
    (RES / "resumo.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    par, seq = analise_threads()
    ag = analise_sched()
    tam = analise_tamanhos()
    try:
        relatorio(par, seq, ag, tam)
    except ImportError:
        print("instale 'tabulate' para gerar resumo.md")
    print("Graficos em", GRAF)
