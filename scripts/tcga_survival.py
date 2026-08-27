#!/usr/bin/env python3
"""
Download TCGA STAR-count RNA-seq data from the NCI GDC and produce:

1. Kaplan-Meier overall-survival plot
   - Primary Tumor samples only
   - Expression = GDC tpm_unstranded (NOT DESeq2)
   - Low  = <= 25th percentile
   - High = >= 75th percentile
   - Middle 50% excluded

2. Tumor-vs-normal expression plot
   - Primary Tumor vs Solid Tissue Normal from the same TCGA project
   - Raw GDC unstranded counts are normalized/analyzed with PyDESeq2
   - Plot uses log2(DESeq2 normalized count + 1)

Example:
    python tcga_gene_km.py --project TCGA-LUAD --gene RAB15 --outdir RAB15_LUAD

Dependencies:
    pip install requests pandas numpy matplotlib lifelines pydeseq2
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


GDC_API = "https://api.gdc.cancer.gov"
FILES_ENDPOINT = f"{GDC_API}/files"
CASES_ENDPOINT = f"{GDC_API}/cases"
DATA_ENDPOINT = f"{GDC_API}/data"

SAMPLE_TYPES = {"Primary Tumor", "Solid Tissue Normal"}


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def query_expression_files(project: str) -> pd.DataFrame:
    """Get open-access STAR-count files for tumor + TCGA normal samples."""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project,
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_category",
                    "value": "Transcriptome Profiling",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_type",
                    "value": "Gene Expression Quantification",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": "STAR - Counts",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.access",
                    "value": "open",
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.samples.sample_type",
                    "value": sorted(SAMPLE_TYPES),
                },
            },
        ],
    }

    fields = ",".join(
        [
            "file_id",
            "file_name",
            "cases.case_id",
            "cases.submitter_id",
            "cases.samples.sample_id",
            "cases.samples.submitter_id",
            "cases.samples.sample_type",
        ]
    )

    rows = []
    page_size = 500
    offset = 0

    while True:
        payload = {
            "filters": filters,
            "format": "JSON",
            "fields": fields,
            "size": page_size,
            "from": offset,
        }
        data = post_json(FILES_ENDPOINT, payload)
        hits = data["data"]["hits"]
        if not hits:
            break

        for hit in hits:
            cases = hit.get("cases", [])
            if not cases:
                continue

            case = cases[0]
            samples = [s for s in case.get("samples", []) if s.get("sample_type") in SAMPLE_TYPES]

            # Gene-expression files should map to one RNA sample. If GDC returns
            # more than one nested sample, retain each candidate and later
            # de-duplicate at the sample level.
            for sample in samples:
                rows.append(
                    {
                        "file_id": hit["file_id"],
                        "file_name": hit["file_name"],
                        "case_id": case.get("case_id"),
                        "case_submitter_id": case.get("submitter_id"),
                        "sample_id": sample.get("sample_id"),
                        "sample_submitter_id": sample.get("submitter_id"),
                        "sample_type": sample.get("sample_type"),
                    }
                )

        offset += len(hits)
        total = data["data"]["pagination"]["total"]
        if offset >= total:
            break

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError(f"No STAR-count files found for {project}.")

    meta = meta.drop_duplicates(subset=["file_id", "sample_id"]).copy()
    meta = meta.sort_values(["case_submitter_id", "sample_submitter_id", "file_id"])

    print("GDC files found:")
    print(meta["sample_type"].value_counts().to_string())
    return meta


def download_file(file_id: str, file_name: str, cache_dir: Path) -> Path:
    """Download one open-access GDC file, with local caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / file_name
    if path.exists() and path.stat().st_size > 0:
        return path

    url = f"{DATA_ENDPOINT}/{file_id}"
    tmp = path.with_suffix(path.suffix + ".part")

    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(path)
            return path
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise
            time.sleep(2 * attempt)

    raise RuntimeError("unreachable")


def read_star_counts(path: Path) -> pd.DataFrame:
    """
    Read a current GDC STAR-count TSV.

    The file contains annotation columns plus raw unstranded counts and TPM.
    Comment lines are ignored. STAR summary rows such as N_unmapped are removed.
    """
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)

    required = {"gene_id", "gene_name", "unstranded", "tpm_unstranded"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected STAR-count columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.loc[
        df["gene_id"].astype(str).str.startswith("ENSG"),
        ["gene_id", "gene_name", "unstranded", "tpm_unstranded"],
    ].copy()

    df["unstranded"] = pd.to_numeric(df["unstranded"], errors="coerce")
    df["tpm_unstranded"] = pd.to_numeric(df["tpm_unstranded"], errors="coerce")
    df = df.dropna(subset=["gene_id", "unstranded", "tpm_unstranded"])
    df["unstranded"] = df["unstranded"].astype(np.int64)
    return df


def build_expression_matrices(meta: pd.DataFrame, cache_dir: Path):
    """Build samples x genes raw-count matrix and sample x gene TPM matrix."""
    counts = {}
    tpms = {}
    gene_names = None

    unique_files = meta.drop_duplicates("file_id")
    n = len(unique_files)

    for i, row in enumerate(unique_files.itertuples(index=False), start=1):
        print(f"[{i:4d}/{n}] {row.file_name}", end="\r", flush=True)
        path = download_file(row.file_id, row.file_name, cache_dir)
        d = read_star_counts(path)
        d = d.drop_duplicates("gene_id", keep="first").set_index("gene_id")

        # Use file UUID as the matrix row key; metadata is joined afterward.
        counts[row.file_id] = d["unstranded"]
        tpms[row.file_id] = d["tpm_unstranded"]

        if gene_names is None:
            gene_names = d["gene_name"].copy()

    print()
    count_df = pd.DataFrame(counts).T
    tpm_df = pd.DataFrame(tpms).T

    # Require genes to exist in all selected files.
    common = count_df.columns.intersection(tpm_df.columns)
    count_df = count_df.loc[:, common]
    tpm_df = tpm_df.loc[:, common]
    gene_names = gene_names.reindex(common)

    return count_df, tpm_df, gene_names


def resolve_gene(
    query: str,
    gene_names: pd.Series,
    counts: pd.DataFrame,
) -> tuple[str, str]:
    """Resolve a gene symbol or Ensembl ID to the GDC gene_id used in matrices."""
    query = query.strip()

    if query.upper().startswith("ENSG"):
        bare = query.split(".")[0].upper()
        candidates = [g for g in counts.columns if g.split(".")[0].upper() == bare]
    else:
        candidates = gene_names.index[gene_names.astype(str).str.upper() == query.upper()].tolist()

    if not candidates:
        raise ValueError(f"Gene '{query}' was not found in the GDC STAR-count files.")

    if len(candidates) > 1:
        totals = counts[candidates].sum(axis=0)
        gene_id = totals.idxmax()
        print(
            f"Warning: {query} matched {len(candidates)} gene IDs; "
            f"using {gene_id} (largest total count)."
        )
    else:
        gene_id = candidates[0]

    gene_symbol = str(gene_names.get(gene_id, query))
    return gene_id, gene_symbol


def make_sample_table(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Choose one expression file per biological sample, then one sample per
    case + sample type. This prevents technical/repeated files from being
    counted as independent observations.
    """
    m = meta.sort_values(["case_submitter_id", "sample_submitter_id", "file_id"]).copy()
    m = m.drop_duplicates(subset=["sample_id"], keep="first")
    m = m.drop_duplicates(subset=["case_id", "sample_type"], keep="first")
    return m


def get_num(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def query_survival(project: str) -> pd.DataFrame:
    """
    Retrieve overall-survival fields from GDC.

    Dead:  event=1, time=demographic.days_to_death
    Alive: event=0, time=max(diagnoses.days_to_last_follow_up,
                             follow_ups.days_to_follow_up)
    """
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": project},
    }
    fields = ",".join(
        [
            "case_id",
            "submitter_id",
            "demographic.vital_status",
            "demographic.days_to_death",
            "diagnoses.days_to_last_follow_up",
            "follow_ups.days_to_follow_up",
        ]
    )

    rows = []
    page_size = 500
    offset = 0

    while True:
        payload = {
            "filters": filters,
            "format": "JSON",
            "fields": fields,
            "expand": "demographic,diagnoses,follow_ups",
            "size": page_size,
            "from": offset,
        }
        data = post_json(CASES_ENDPOINT, payload)
        hits = data["data"]["hits"]
        if not hits:
            break

        for case in hits:
            dem = case.get("demographic") or {}
            vital = str(dem.get("vital_status", "")).strip().lower()

            if vital not in {"alive", "dead"}:
                continue

            death = get_num(dem.get("days_to_death"))
            follow = []

            for dx in case.get("diagnoses", []) or []:
                x = get_num(dx.get("days_to_last_follow_up"))
                if np.isfinite(x):
                    follow.append(x)

            for fu in case.get("follow_ups", []) or []:
                x = get_num(fu.get("days_to_follow_up"))
                if np.isfinite(x):
                    follow.append(x)

            if vital == "dead":
                if not np.isfinite(death) or death < 0:
                    continue
                os_days = death
                event = 1
            else:
                follow = [x for x in follow if x >= 0]
                if not follow:
                    continue
                os_days = max(follow)
                event = 0

            rows.append(
                {
                    "case_id": case.get("case_id"),
                    "case_submitter_id": case.get("submitter_id"),
                    "os_days": os_days,
                    "event": event,
                }
            )

        offset += len(hits)
        total = data["data"]["pagination"]["total"]
        if offset >= total:
            break

    surv = pd.DataFrame(rows).drop_duplicates("case_id")
    if surv.empty:
        raise RuntimeError(f"No usable survival data found for {project}.")
    return surv


def run_km(
    sample_table: pd.DataFrame,
    tpm_df: pd.DataFrame,
    survival: pd.DataFrame,
    gene_id: str,
    gene_symbol: str,
    project: str,
    outdir: Path,
):
    """TPM-based Q25/Q75 KM analysis. No DESeq2 values are used here."""
    tumor = sample_table[sample_table["sample_type"] == "Primary Tumor"].copy()
    tumor["tpm"] = tumor["file_id"].map(tpm_df[gene_id])
    tumor = tumor.merge(survival, on=["case_id", "case_submitter_id"], how="inner")
    tumor = tumor.dropna(subset=["tpm", "os_days", "event"]).copy()
    tumor = tumor[tumor["os_days"] >= 0].copy()

    if len(tumor) < 20:
        raise RuntimeError(f"Only {len(tumor)} tumor cases have usable TPM + survival data.")

    q25 = tumor["tpm"].quantile(0.25)
    q75 = tumor["tpm"].quantile(0.75)

    if not np.isfinite(q25) or not np.isfinite(q75) or q25 >= q75:
        raise RuntimeError(
            f"Cannot make a non-overlapping quartile split for {gene_symbol}: "
            f"Q25={q25}, Q75={q75}. The expression distribution has too many ties."
        )

    low = tumor[tumor["tpm"] <= q25].copy()
    high = tumor[tumor["tpm"] >= q75].copy()

    low["group"] = "Low"
    high["group"] = "High"
    km = pd.concat([low, high], ignore_index=True)
    km["years"] = km["os_days"] / 365.25

    lr = logrank_test(
        high["os_days"],
        low["os_days"],
        event_observed_A=high["event"],
        event_observed_B=low["event"],
    )

    # Optional univariate HR: High vs Low.
    hr_text = "HR unavailable"
    hr = ci_low = ci_high = np.nan
    try:
        cox_df = km[["os_days", "event", "group"]].copy()
        cox_df["high"] = (cox_df["group"] == "High").astype(int)
        cph = CoxPHFitter()
        cph.fit(cox_df[["os_days", "event", "high"]], duration_col="os_days", event_col="event")
        hr = float(np.exp(cph.params_["high"]))
        ci = np.exp(cph.confidence_intervals_.loc["high"].to_numpy(dtype=float))
        ci_low, ci_high = float(ci[0]), float(ci[1])
        hr_text = f"HR={hr:.2f} (95% CI {ci_low:.2f}-{ci_high:.2f})"
    except Exception as e:
        print(f"Warning: Cox HR could not be estimated: {e}", file=sys.stderr)

    km_low = KaplanMeierFitter(label=f"Low (<=Q25), n={len(low)}")
    km_high = KaplanMeierFitter(label=f"High (>=Q75), n={len(high)}")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    km_low.fit(low["years"], event_observed=low["event"])
    km_high.fit(high["years"], event_observed=high["event"])
    km_low.plot_survival_function(ax=ax, ci_show=True)
    km_high.plot_survival_function(ax=ax, ci_show=True)

    ax.set_title(f"{project}: {gene_symbol} overall survival")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.03)
    ax.text(
        0.98,
        0.03,
        f"Log-rank p={lr.p_value:.3g}\n{hr_text}\nQ25={q25:.3g}, Q75={q75:.3g} TPM",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    add_at_risk_counts(km_low, km_high, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / f"{gene_symbol}_KM_TPM_Q25_Q75.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{gene_symbol}_KM_TPM_Q25_Q75.pdf", bbox_inches="tight")
    plt.close(fig)

    km.to_csv(outdir / f"{gene_symbol}_KM_cases.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "gene": gene_symbol,
                "gene_id": gene_id,
                "project": project,
                "q25_tpm": q25,
                "q75_tpm": q75,
                "n_low": len(low),
                "n_high": len(high),
                "events_low": int(low["event"].sum()),
                "events_high": int(high["event"].sum()),
                "logrank_p": lr.p_value,
                "hr_high_vs_low": hr,
                "hr_ci95_low": ci_low,
                "hr_ci95_high": ci_high,
            }
        ]
    ).to_csv(outdir / f"{gene_symbol}_KM_stats.tsv", sep="\t", index=False)

    print("\nKM analysis")
    print(f"  usable tumor cases: {len(tumor)}")
    print(f"  Low <= Q25 ({q25:.4g} TPM):  n={len(low)}")
    print(f"  High >= Q75 ({q75:.4g} TPM): n={len(high)}")
    print(f"  log-rank p = {lr.p_value:.4g}")
    print(f"  {hr_text}")


def run_deseq_expression(
    sample_table: pd.DataFrame,
    count_df: pd.DataFrame,
    gene_id: str,
    gene_symbol: str,
    project: str,
    outdir: Path,
    n_cpus: int,
):
    """DESeq2 tumor-vs-TCGA-normal analysis using raw unstranded counts."""
    m = sample_table[sample_table["sample_type"].isin(SAMPLE_TYPES)].copy()
    m["condition"] = m["sample_type"].map(
        {"Primary Tumor": "Tumor", "Solid Tissue Normal": "Normal"}
    )

    n_tumor = int((m["condition"] == "Tumor").sum())
    n_normal = int((m["condition"] == "Normal").sum())
    if n_tumor < 2 or n_normal < 2:
        raise RuntimeError(
            f"DESeq2 requires replicates in both groups; found Tumor={n_tumor}, Normal={n_normal}."
        )

    counts = count_df.loc[m["file_id"]].copy()
    counts.index = m["file_id"].to_numpy()

    # Remove near-empty genes to reduce memory/runtime, but always keep target gene.
    keep = counts.sum(axis=0) >= 10
    if counts[gene_id].sum() > 0:
        keep.loc[gene_id] = True
    counts = counts.loc[:, keep].astype(np.int64)

    metadata = m.set_index("file_id")[["condition"]].loc[counts.index].copy()
    metadata["condition"] = pd.Categorical(metadata["condition"], categories=["Normal", "Tumor"])

    print("\nRunning PyDESeq2...")
    print(f"  Tumor samples:  {n_tumor}")
    print(f"  Normal samples: {n_normal}")
    print(f"  Genes retained: {counts.shape[1]}")

    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~condition",
        refit_cooks=True,
        n_cpus=n_cpus,
    )
    dds.deseq2()

    ds = DeseqStats(
        dds,
        contrast=["condition", "Tumor", "Normal"],
        n_cpus=n_cpus,
    )
    ds.summary()
    results = ds.results_df.copy()
    results.to_csv(outdir / "DESeq2_Tumor_vs_Normal_all_genes.tsv", sep="\t")

    if gene_id not in results.index:
        raise RuntimeError(f"{gene_symbol} ({gene_id}) is absent from the DESeq2 result table.")

    gene_res = results.loc[gene_id]
    gene_res.to_frame().T.to_csv(outdir / f"{gene_symbol}_DESeq2_result.tsv", sep="\t", index=True)

    norm_counts = pd.DataFrame(
        dds.layers["normed_counts"],
        index=dds.obs_names,
        columns=dds.var_names,
    )

    plot_df = m.set_index("file_id").loc[norm_counts.index].copy()
    plot_df["norm_count"] = norm_counts[gene_id]
    plot_df["log2_norm_count"] = np.log2(plot_df["norm_count"] + 1.0)
    plot_df.reset_index().to_csv(
        outdir / f"{gene_symbol}_DESeq2_normalized_expression.tsv",
        sep="\t",
        index=False,
    )

    groups = ["Normal", "Tumor"]
    vals = [plot_df.loc[plot_df["condition"] == g, "log2_norm_count"].to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.boxplot(vals, labels=groups, showfliers=False)

    rng = np.random.default_rng(1)
    for x, y in enumerate(vals, start=1):
        jitter = rng.normal(x, 0.055, size=len(y))
        ax.scatter(jitter, y, s=15, alpha=0.55)

    lfc = gene_res.get("log2FoldChange", np.nan)
    padj = gene_res.get("padj", np.nan)
    pval = gene_res.get("pvalue", np.nan)

    p_label = f"padj={padj:.3g}" if np.isfinite(padj) else f"p={pval:.3g}"
    ax.set_title(f"{project}: {gene_symbol}\nDESeq2 log2FC={lfc:.2f}, {p_label}")
    ax.set_ylabel("log2(DESeq2 normalized count + 1)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(outdir / f"{gene_symbol}_Tumor_vs_Normal_DESeq2.png", dpi=300)
    fig.savefig(outdir / f"{gene_symbol}_Tumor_vs_Normal_DESeq2.pdf")
    plt.close(fig)

    print("\nDESeq2 target-gene result")
    print(gene_res.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="TCGA GDC TPM-quartile KM + DESeq2 tumor/normal expression"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="GDC project ID, e.g. TCGA-LUAD",
    )
    parser.add_argument(
        "--gene",
        required=True,
        help="Gene symbol or Ensembl gene ID, e.g. RAB15 or ENSG00000139998",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <PROJECT>_<GENE>)",
    )
    parser.add_argument(
        "--cache",
        default="gdc_star_counts",
        help="Directory used to cache downloaded GDC STAR-count files",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="CPUs for PyDESeq2 (default: 4)",
    )
    args = parser.parse_args()

    project = args.project.upper()
    outdir = Path(args.outdir or f"{project}_{args.gene}")
    cache_dir = Path(args.cache) / project
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Project: {project}")
    print(f"Gene:    {args.gene}")

    meta = query_expression_files(project)
    sample_table = make_sample_table(meta)

    sample_table.to_csv(outdir / "GDC_expression_samples.tsv", sep="\t", index=False)

    count_df, tpm_df, gene_names = build_expression_matrices(sample_table, cache_dir)
    gene_id, gene_symbol = resolve_gene(args.gene, gene_names, count_df)
    print(f"Resolved gene: {gene_symbol} ({gene_id})")

    # 1) KM: TPM only. No DESeq2 values are used.
    survival = query_survival(project)
    survival.to_csv(outdir / "GDC_survival.tsv", sep="\t", index=False)
    run_km(
        sample_table,
        tpm_df,
        survival,
        gene_id,
        gene_symbol,
        project,
        outdir,
    )

    # 2) Tumor vs TCGA normal: raw counts -> PyDESeq2.
    run_deseq_expression(
        sample_table,
        count_df,
        gene_id,
        gene_symbol,
        project,
        outdir,
        args.cpus,
    )

    print(f"\nDone. Results written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
