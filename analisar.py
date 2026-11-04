# -*- coding: utf-8 -*-
"""Analise dos resultados: tabelas (CSV/Markdown) e graficos em resultados/graficos/.

Metricas (T1 - PCD, etapas 4 e 5):
  speedup relativo  S(p)     = T_seq / T_p   (baseline: sequencial da MESMA ordem)
  speedup absoluto  S_abs(p) = T_melhor_seq / T_p  (baseline unico: melhor ordem sequencial)
  eficiencia        E(p)     = S(p) / p
  Karp-Flatt        e(p)     = (1/S - 1/p) / (1 - 1/p)   fracao serial experimental
  Amdahl            S(p)     = 1 / (f + (1-f)/p)         ajustada por minimos quadrados
  MFLOPS, MFLOPS por thread, utilizacao de CPU (cpu/(parede*p)), custo em nucleo-segundos.
Energia (so se o RAPL foi lido; senao a secao e omitida):
  energia E (J) do pacote no MESMO trecho cronometrado, potencia media P = E/T,
  energia liquida = E - P_ocioso*T, EDP = E*T, J/GFLOP. Media e desvio padrao por configuracao.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
GRAF = RES / "graficos"
COR = {"ijk": "#d1495b", "ikj": "#2a9d8f"}
NOME = {"ijk": "ijk (acesso a B por coluna)", "ikj": "ikj (acesso sequencial)"}
LIMIAR_CURTO_S = 0.05   # execucoes mais curtas que isso: energia/CPU pouco confiaveis (RAPL ~1 ms)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "legend.frameon": False})


def carregar_ambiente():
    f = RES / "ambiente.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return d.get("nucleos_fisicos"), d.get("nucleos_logicos")
    return None, None


def carregar_ocioso():
    f = RES / "ocioso.csv"
    if f.exists():
        return float(pd.read_csv(f).p_ocioso_w.mean())
    return float("nan")


NUCLEOS_FISICOS, NUCLEOS_LOGICOS = carregar_ambiente()
P_OCIOSO = carregar_ocioso()


def ler(nome):
    df = pd.read_csv(RES / nome)
    if "energia_j" not in df:
        df["energia_j"] = np.nan
    gflop = 2.0 * df.n.astype(float) ** 3 / 1e9
    df["potencia_w"] = df.energia_j / df.tempo_s
    df["energia_liq_j"] = df.energia_j - P_OCIOSO * df.tempo_s
    df["edp"] = df.energia_j * df.tempo_s
    df["j_por_gflop"] = df.energia_j / gflop
    return df


def agrega(df, chaves):
    g = df.groupby(chaves)
    out = g.agg(tempo_med=("tempo_s", "mean"), tempo_mediana=("tempo_s", "median"),
                tempo_dp=("tempo_s", "std"), tempo_min=("tempo_s", "min"),
                cpu_med=("cpu_s", "mean"), mflops=("mflops", "median"),
                util=("util", "median"), amostras=("tempo_s", "count"),
                energia_med=("energia_j", "mean"), energia_dp=("energia_j", "std"),
                energia_liq_med=("energia_liq_j", "mean"), potencia_med=("potencia_w", "mean"),
                edp_med=("edp", "mean"), j_gflop_med=("j_por_gflop", "mean")).reset_index()
    out["cv_pct"] = 100 * out["tempo_dp"] / out["tempo_med"]
    out["energia_cv_pct"] = 100 * out["energia_dp"] / out["energia_med"]
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
    if not NUCLEOS_FISICOS or NUCLEOS_FISICOS == NUCLEOS_LOGICOS:
        return
    ax.axvline(NUCLEOS_FISICOS, color="gray", ls=":", lw=1)
    ax.text(NUCLEOS_FISICOS, ax.get_ylim()[1], f" {NUCLEOS_FISICOS} núcleos\n físicos",
            va="top", fontsize=8, color="gray")


def tem_energia(df):
    return "energia_med" in df and df.energia_med.notna().any()


# ---------------------------------------------------------------- A) threads
def analise_threads():
    df = ler("threads.csv")
    seq = agrega(df[df.sched == "seq"], ["versao"]).set_index("versao")
    par = agrega(df[df.sched != "seq"], ["versao", "threads"])
    ref = seq.tempo_mediana.min()  # melhor tempo sequencial (baseline absoluto)
    for v in ("ijk", "ikj"):
        m = par.versao == v
        par.loc[m, "speedup"] = seq.loc[v, "tempo_mediana"] / par.loc[m, "tempo_mediana"]
        t1 = par[m & (par.threads == 1)].tempo_mediana.iloc[0]
        par.loc[m, "speedup_vs_1t"] = t1 / par.loc[m, "tempo_mediana"]
        par.loc[m, "energia_rel_seq"] = par.loc[m, "energia_med"] / seq.loc[v, "energia_med"]
    par["speedup_abs"] = ref / par.tempo_mediana
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
        ax.plot(d.threads, d.speedup, "o-", color=COR[v], label="Medido (relativo)")
        ax.plot(d.threads, d.speedup_abs, "s:", color=COR[v], alpha=0.6, label="Medido (absoluto)")
        ax.set(xlabel="Threads", ylabel="Speedup", title=f"Speedup — {NOME[v]}")
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
    axs[1].set(xlabel="Threads", ylabel="MFLOPS por thread", title="Trabalho por thread")
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
    razao = par.pivot(index="threads", columns="versao", values="tempo_mediana")
    razao["ijk/ikj"] = razao.ijk / razao.ikj
    ax.bar(razao.index.astype(str), razao["ijk/ikj"], color="#6c757d")
    ax.axhline(seq.loc["ijk", "tempo_mediana"] / seq.loc["ikj", "tempo_mediana"], color="k", ls="--",
               label="sequencial")
    ax.set(xlabel="Threads", ylabel="Tempo ijk / tempo ikj", title="Ganho da ordem ikj (localidade de cache)")
    ax.legend()
    salvar(fig, "07_ijk_vs_ikj.png")

    # 8) energia (somente com leituras RAPL)
    if tem_energia(par):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for v in ("ijk", "ikj"):
            d = par[par.versao == v]
            axs[0].errorbar(d.threads, d.energia_med, yerr=d.energia_dp, marker="o", capsize=3,
                            color=COR[v], label=NOME[v])
            axs[0].axhline(seq.loc[v, "energia_med"], color=COR[v], ls="--", lw=1, alpha=0.6)
            axs[1].plot(d.threads, d.potencia_med, "o-", color=COR[v], label=NOME[v])
            axs[2].plot(d.threads, d.edp_med, "o-", color=COR[v], label=NOME[v])
        axs[0].set(xlabel="Threads", ylabel="Energia do pacote (J) — média ± dp", yscale="log",
                   title="Energia por execução (RAPL). Tracejado = seq.")
        axs[1].set(xlabel="Threads", ylabel="Potência média (W)", title="Potência média = E/T")
        axs[2].set(xlabel="Threads", ylabel="EDP = E × T (J·s)", yscale="log", title="Energy-Delay Product")
        for ax in axs:
            ax.set_xticks(par.threads.unique()); ax.legend(); marca_ht(ax)
        salvar(fig, "10_energia.png")
    return par, seq


# -------------------------------------------------------------- B) escalonamento
def analise_sched():
    df = ler("escalonamento.csv")
    df["config"] = df.sched + "," + df.chunk.astype(str).replace("0", "padrão")
    ag = agrega(df, ["versao", "threads", "config"])
    ag.to_csv(RES / "resumo_escalonamento.csv", index=False)
    ordem = list(dict.fromkeys(df.config))
    ts = sorted(df.threads.unique())
    fig, axs = plt.subplots(len(ts), 2, figsize=(12, 3.5 * len(ts)), squeeze=False)
    for j, v in enumerate(("ijk", "ikj")):
        for i, t in enumerate(ts):
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
    par["curto"] = par.tempo_med < LIMIAR_CURTO_S
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


# ------------------------------------------------------- D) tabela principal
def tabela_principal(par, seq):
    """Tabela unica para o artigo (N=1024, schedule static)."""
    ref = seq.tempo_mediana.min()
    s = seq.reset_index()
    s["threads"] = "seq"
    s["speedup"] = 1.0
    s["speedup_abs"] = ref / s.tempo_mediana
    s["eficiencia"] = np.nan
    t = par.copy()
    t["threads"] = t.threads.astype(str)
    cols = ["versao", "threads", "tempo_med", "tempo_dp", "cv_pct", "speedup", "speedup_abs",
            "eficiencia", "mflops"]
    nomes = {"versao": "Ordem", "threads": "Threads", "tempo_med": "Tempo médio (s)",
             "tempo_dp": "Desvio padrão (s)", "cv_pct": "CV (%)", "speedup": "Speedup relativo",
             "speedup_abs": "Speedup absoluto", "eficiencia": "Eficiência", "mflops": "MFLOPS (mediana)",
             "energia_med": "Energia média (J)", "energia_dp": "Desvio energia (J)",
             "potencia_med": "Potência média (W)"}
    if tem_energia(par):
        cols += ["energia_med", "energia_dp", "potencia_med"]
    out = pd.concat([s[cols], t[cols]], ignore_index=True)
    out["ordem"] = out.threads.map(lambda x: 0 if x == "seq" else int(x))
    out = out.sort_values(["versao", "ordem"]).drop(columns="ordem").reset_index(drop=True)
    out = out.rename(columns=nomes)
    casas = {"Tempo médio (s)": 4, "Desvio padrão (s)": 4, "CV (%)": 1, "Speedup relativo": 2,
             "Speedup absoluto": 2, "Eficiência": 2, "MFLOPS (mediana)": 0,
             "Energia média (J)": 2, "Desvio energia (J)": 2, "Potência média (W)": 1}
    out = out.round({k: v for k, v in casas.items() if k in out})
    out.to_csv(RES / "tabela_principal.csv", index=False)
    return out


def relatorio(par, seq, ag_sched, tam, tab):
    L = ["# Resumo dos resultados (gerado por analisar.py)\n"]
    amb = RES / "ambiente.txt"
    if amb.exists():
        L.append("## Ambiente\n")
        L.append("\n".join("- " + l for l in amb.read_text(encoding="utf-8").splitlines()))
    L.append("\n## Tabela principal (N=1024, schedule static)\n")
    L.append("Speedup relativo = T_seq da mesma ordem / T_p. Speedup absoluto = T da melhor ordem "
             "sequencial / T_p. Eficiência = speedup relativo / p.\n")
    L.append(tab.to_markdown(index=False))
    if tem_energia(par):
        L.append(f"\n## Energia (RAPL, pacote; potência ociosa de referência = {P_OCIOSO:.2f} W)\n")
        L.append("Energia e tempo medidos sobre o mesmo trecho (apenas a multiplicação). "
                 "Média e desvio padrão sobre as repetições de cada configuração.\n")
        cols = ["versao", "threads", "energia_med", "energia_dp", "energia_cv_pct", "energia_liq_med",
                "potencia_med", "edp_med", "j_gflop_med", "energia_rel_seq"]
        L.append(par[cols].round(3).to_markdown(index=False))
    L.append("\n## Varredura de threads (detalhe, N=1024, static)\n")
    cols = ["versao", "threads", "tempo_med", "tempo_dp", "cv_pct", "speedup", "speedup_abs", "eficiencia",
            "karp_flatt", "mflops", "util", "nucleo_segundos"]
    L.append(par[cols].round(3).to_markdown(index=False))
    L.append("\n## Escalonamento\n")
    cols = ["versao", "threads", "config", "tempo_med", "tempo_dp", "mflops"]
    if tem_energia(ag_sched):
        cols += ["energia_med", "energia_dp"]
    L.append(ag_sched[cols].round(4).to_markdown(index=False))
    L.append("\n## Tamanhos\n")
    cols = ["versao", "n", "threads", "tempo_med", "speedup", "eficiencia", "curto"]
    if tem_energia(tam):
        cols += ["energia_med", "energia_dp"]
    L.append(tam[cols].round(3).to_markdown(index=False))
    L.append(f"\n> `curto` = execução média < {LIMIAR_CURTO_S * 1000:.0f} ms: energia e tempo de CPU "
             "pouco confiáveis (resolução do RAPL e do relógio de CPU).")
    L.append("\n> Escopo: resultados válidos apenas para os tamanhos e o ambiente acima.")
    (RES / "resumo.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    par, seq = analise_threads()
    ag = analise_sched()
    tam = analise_tamanhos()
    tab = tabela_principal(par, seq)
    try:
        relatorio(par, seq, ag, tam, tab)
    except ImportError:
        print("instale 'tabulate' para gerar resumo.md")
    print("Graficos em", GRAF)
