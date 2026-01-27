#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "report" / "sim_conditions_inputs.txt"
DEFAULT_OUTPUT = ROOT / "report" / "sim_conditions_report.tex"


def _read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _parse_kv_tokens(stem: str) -> Dict[str, str]:
    parts = stem.split("_")
    out: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k] = v
    return out


def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _fmt_bool(x: Any) -> str:
    return "True" if bool(x) else "False"


def _tex_escape_basic(s: str) -> str:
    # Keep this conservative; paths/filenames are handled via \detokenize{...}.
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _tex_num(x: Any) -> str:
    if x is None:
        return "---"
    if isinstance(x, bool):
        return _fmt_bool(x)
    if isinstance(x, (int, float)):
        return f"\\num{{{x}}}"
    return _tex_escape_basic(str(x))


def _detok(s: str) -> str:
    return f"\\detokenize{{{s}}}"


@dataclass(frozen=True)
class RunGroup:
    images_dir: Path
    input_paths: List[Path]
    meta_path: Path
    meta: Dict[str, Any]

    @property
    def scenario(self) -> str:
        sc = self.meta.get("scenario")
        if isinstance(sc, str) and sc.strip():
            return sc.strip()
        # Fallback: infer from filenames (piecewise/linear)
        for p in self.input_paths:
            stem = p.stem
            if stem.startswith("piecewise"):
                return "piecewise"
            if stem.startswith("linear"):
                return "linear"
        return "unknown"


def _choose_best_meta(images_dir: Path, targets: List[Path]) -> Path:
    metas = sorted(images_dir.glob("*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No *_meta.json found under {images_dir}")
    if len(metas) == 1:
        return metas[0]

    # Score by token overlap with provided targets.
    target_tokens: List[set[str]] = []
    for t in targets:
        target_tokens.append(set(f"{k}={v}" for k, v in _parse_kv_tokens(t.stem).items()))

    def score(meta: Path) -> Tuple[int, int]:
        stem = meta.name.removesuffix("_meta.json")
        mtoks = set(f"{k}={v}" for k, v in _parse_kv_tokens(stem).items())
        best = 0
        for tt in target_tokens:
            best = max(best, len(mtoks & tt))
        # Secondary: prefer longer match (more tokens in meta filename)
        return best, len(mtoks)

    metas.sort(key=score, reverse=True)
    return metas[0]


def _load_groups(input_items: List[str]) -> List[RunGroup]:
    paths = [ROOT / p for p in input_items]
    # Only keep files that exist; fail fast with a useful error.
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing paths:\n" + "\n".join(str(p) for p in missing))

    groups: Dict[Path, List[Path]] = {}
    for p in paths:
        if p.suffix.lower() != ".png":
            # The user-provided list may include non-figures; ignore for figure embedding,
            # but still include as a “target” for selecting the corresponding meta.json.
            pass
        images_dir = p.parent
        groups.setdefault(images_dir, []).append(p)

    run_groups: List[RunGroup] = []
    for images_dir, target_paths in sorted(groups.items()):
        meta_path = _choose_best_meta(images_dir, target_paths)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_groups.append(
            RunGroup(
                images_dir=images_dir,
                input_paths=sorted(target_paths),
                meta_path=meta_path,
                meta=meta,
            )
        )
    return run_groups


def _method_label(method_key: str) -> str:
    return {
        "pp": "PP (Proposed)",
        "pp_sgd": "PP-SGD",
        "pc": "PC (Prediction-Correction)",
        "co": "CO (Correction-Only)",
        "sgd": "SGD",
        "pg": "PG (Prox-Grad)",
        "offline": "Offline",
    }.get(method_key, method_key)


def _hyperparam_mapping_rows() -> List[Tuple[str, str]]:
    # (hyperparam key in code/json, which method uses it)
    return [
        ("$r, q, \\rho, \\mu_\\lambda, \\lambda_S$", "PP / PP-SGD"),
        ("$\\lambda_{\\mathrm{reg}}, \\alpha, \\beta, \\gamma, P, C$", "PC"),
        ("$\\lambda_{\\mathrm{reg}}, \\alpha, \\beta_{\\mathrm{co}}, \\gamma, C$", "CO"),
        ("$\\lambda_{\\mathrm{reg}}, \\alpha, \\beta_{\\mathrm{sgd}}, C$", "SGD"),
        ("$\\lambda_{\\mathrm{reg}}$, \\texttt{step\\_scale}, \\texttt{step\\_size}, \\texttt{use\\_fista}, \\texttt{use\\_backtracking}, \\texttt{max\\_iter}, \\texttt{tol}", "PG"),
        ("$\\lambda_{\\mathrm{offline}}$", "Offline solution (reference)"),
    ]


def _extract_common(meta: Dict[str, Any]) -> Dict[str, Any]:
    cfg = meta.get("config") or {}
    return {
        "num_trials": cfg.get("num_trials"),
        "seed_base": cfg.get("seed_base"),
        "N": cfg.get("N"),
        "T": cfg.get("T"),
        "sparsity": cfg.get("sparsity"),
        "max_weight": cfg.get("max_weight"),
        "std_e": cfg.get("std_e"),
        # Piecewise only (may be absent for linear)
        "K": cfg.get("K"),
    }


def _extract_generator(meta: Dict[str, Any]) -> Dict[str, Any]:
    g = meta.get("generator") or {}
    kwargs = g.get("kwargs") or {}
    return {
        "function": g.get("function"),
        "s_type": kwargs.get("s_type"),
        "t_min": kwargs.get("t_min"),
        "t_max": kwargs.get("t_max"),
        "z_dist": kwargs.get("z_dist"),
    }


def _extract_metric(meta: Dict[str, Any]) -> Dict[str, Any]:
    m = meta.get("metric") or {}
    return {
        "error_normalization": m.get("error_normalization"),
        "divide_by_n2": m.get("divide_by_n2"),
        "burn_in": m.get("burn_in"),
        "burn_in_effective": m.get("burn_in_effective"),
        "offline_lambda_l1": m.get("offline_lambda_l1"),
        "plot_variants": m.get("plot_variants") or [],
    }


def _extract_comparison(meta: Dict[str, Any]) -> Dict[str, Any]:
    c = meta.get("comparison") or {}
    if not c:
        return {}
    return {
        "pc_model": c.get("pc_model"),
        "pc_use_true_T_init": c.get("pc_use_true_T_init"),
        "pc_T_init_identity_scale": c.get("pc_T_init_identity_scale"),
        "pp_init_b0": c.get("pp_init_b0"),
        "pp_lookahead": c.get("pp_lookahead"),
        "pp_lookahead_effective": c.get("pp_lookahead_effective"),
    }


def _extract_methods(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer meta["methods"] which is already “enabled + hyperparams”.
    methods = meta.get("methods")
    if isinstance(methods, dict) and methods:
        return methods
    # Fallback: some formats keep raw hyperparams under repro.hyperparam_json_content.
    hp = _safe_get(meta, "repro.hyperparam_json_content", {}) or {}
    out: Dict[str, Any] = {}
    for k, v in hp.items():
        out[k] = {"enabled": None, "hyperparams": v}
    return out


def _render_tex(groups: List[RunGroup]) -> str:
    lines: List[str] = []
    lines += [
        "% Auto-generated by generate_sim_conditions_report.py",
        "% Compile (example): latexmk -pdf -interaction=nonstopmode report/sim_conditions_report.tex",
        "",
        "\\PassOptionsToPackage{margin=25mm}{geometry}",
        "\\documentclass[a4paper,11pt]{bxjsarticle}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{graphicx}",
        "\\usepackage{siunitx}",
        "\\sisetup{detect-all=true, scientific-notation=true, round-mode=places, round-precision=6}",
        "",
        "\\newcommand{\\path}[1]{\\texttt{\\detokenize{#1}}}",
        "\\newcommand{\\img}[2][]{\\includegraphics[#1]{\\detokenize{#2}}}",
        "",
        "\\title{シミュレーション条件レポート（出力結果対応表）}",
        "\\author{}",
        "\\date{\\today}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\section{目的}",
        "指定された出力図（\\path{result/.../*.png}）に対して、対応するシミュレーション条件（データ生成パラメータ、評価設定、手法別ハイパーパラメータ）を整理する。",
        "",
        "\\section{ハイパーパラメータと手法の対応}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "ハイパーパラメータ & 対応する手法 \\\\",
        "\\midrule",
    ]
    for hp, method in _hyperparam_mapping_rows():
        lines.append(f"{hp} & {method} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\section{各実験（ディレクトリ単位）の設定}",
        "",
    ]

    for idx, g in enumerate(groups, start=1):
        meta = g.meta
        common = _extract_common(meta)
        gen = _extract_generator(meta)
        metric = _extract_metric(meta)
        comp = _extract_comparison(meta)
        methods = _extract_methods(meta)
        created_at = meta.get("created_at")

        lines += [
            f"\\subsection{{Run {idx}: {g.scenario}}}",
            "\\begin{itemize}",
            f"\\item 出力ディレクトリ: \\path{{{str(g.images_dir.relative_to(ROOT))}}}",
            f"\\item メタデータ: \\path{{{str(g.meta_path.relative_to(ROOT))}}}",
            f"\\item 作成日時 (created\\_at): \\path{{{created_at}}}" if created_at else "\\item 作成日時 (created\\_at): ---",
            "\\end{itemize}",
            "",
        ]

        # Common/scenario params
        lines += [
            "\\begin{table}[h]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "項目 & 値 \\\\",
            "\\midrule",
        ]
        for k in ["N", "T", "K", "sparsity", "max_weight", "std_e", "seed_base", "num_trials"]:
            v = common.get(k)
            if v is None:
                continue
            label = {
                "N": "$N$ (ノード数)",
                "T": "$T$ (時系列長)",
                "K": "$K$ (変化点数 / 区間数)",
                "sparsity": "sparsity（0要素割合）",
                "max_weight": "max\\_weight（生成時上限）",
                "std_e": "$\\sigma_e$ (ノイズ標準偏差)",
                "seed_base": "seed（基点）",
                "num_trials": "num\\_trials（試行回数）",
            }[k]
            lines.append(f"{label} & {_tex_num(v)} \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]

        # Generator
        lines += [
            "\\begin{table}[h]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "データ生成 & 値 \\\\",
            "\\midrule",
            f"関数 & \\path{{{gen.get('function')}}} \\\\" if gen.get("function") else "関数 & --- \\\\",
            f"$S$ 生成タイプ (s\\_type) & \\path{{{gen.get('s_type')}}} \\\\",
            f"$T$ 対角の範囲 (t\\_min, t\\_max) & ({_tex_num(gen.get('t_min'))}, {_tex_num(gen.get('t_max'))}) \\\\",
            f"外生入力分布 (z\\_dist) & \\path{{{gen.get('z_dist')}}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]

        # Metric + variants
        lines += [
            "\\begin{table}[h]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "評価設定 & 値 \\\\",
            "\\midrule",
            f"error\\_normalization & \\path{{{metric.get('error_normalization')}}} \\\\",
            f"divide\\_by\\_n2 & {_tex_num(metric.get('divide_by_n2'))} \\\\",
            f"burn\\_in & {_tex_num(metric.get('burn_in'))} \\\\",
            f"burn\\_in\\_effective & {_tex_num(metric.get('burn_in_effective'))} \\\\",
        ]
        if metric.get("offline_lambda_l1") is not None:
            lines.append(f"offline\\_lambda\\_l1 & {_tex_num(metric.get('offline_lambda_l1'))} \\\\")
        else:
            lines.append("offline\\_lambda\\_l1 & --- \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]

        if comp:
            lines += [
                "\\begin{table}[h]",
                "\\centering",
                "\\small",
                "\\begin{tabular}{ll}",
                "\\toprule",
                "比較条件 (comparison) & 値 \\\\",
                "\\midrule",
            ]
            for k, label in [
                ("pc_model", "pc\\_model"),
                ("pc_use_true_T_init", "pc\\_use\\_true\\_T\\_init"),
                ("pc_T_init_identity_scale", "pc\\_T\\_init\\_identity\\_scale"),
                ("pp_init_b0", "pp\\_init\\_b0"),
                ("pp_lookahead", "pp\\_lookahead"),
                ("pp_lookahead_effective", "pp\\_lookahead\\_effective"),
            ]:
                if k in comp and comp[k] is not None:
                    v = comp[k]
                    v_tex = _tex_num(v) if not isinstance(v, str) else f"\\path{{{v}}}"
                    lines.append(f"{label} & {v_tex} \\\\")
            lines += [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]

        # Method hyperparams table
        lines += [
            "\\begin{longtable}{lll}",
            "\\caption{手法別ハイパーパラメータ（Run "
            + str(idx)
            + "）}\\\\",
            "\\toprule",
            "手法 & 有効化 & ハイパーパラメータ \\\\",
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            "手法 & 有効化 & ハイパーパラメータ \\\\",
            "\\midrule",
            "\\endhead",
        ]
        for method_key in ["pp", "pp_sgd", "pc", "co", "sgd", "pg"]:
            m = methods.get(method_key)
            if not isinstance(m, dict):
                continue
            enabled = m.get("enabled")
            enabled_tex = "---" if enabled is None else ("ON" if enabled else "OFF")
            hp = m.get("hyperparams") or {}
            if isinstance(hp, dict) and hp:
                hp_parts = []
                for hk in sorted(hp.keys()):
                    hv = hp[hk]
                    if isinstance(hv, str):
                        hp_parts.append(f"\\path{{{hk}}}={_detok(hv)}")
                    else:
                        hp_parts.append(f"\\path{{{hk}}}={{{_tex_num(hv)}}}")
                hp_tex = "\\,;\\, ".join(hp_parts)
            else:
                hp_tex = "---"
            lines.append(f"{_method_label(method_key)} & {enabled_tex} & {hp_tex} \\\\")
        lines += [
            "\\bottomrule",
            "\\end{longtable}",
            "",
        ]

        # Figures
        fig_paths = [p for p in g.input_paths if p.suffix.lower() == ".png"]
        if fig_paths:
            lines += [
                "\\paragraph{対応する図（入力指定分）}",
                "",
            ]
            for fp in fig_paths:
                rel = fp.relative_to(ROOT)
                stem = fp.stem
                toks = _parse_kv_tokens(stem)
                norm = toks.get("norm")
                n2 = toks.get("n2")
                variant = []
                if norm is not None:
                    variant.append(f"norm={norm}")
                if n2 is not None:
                    variant.append(f"n2={n2}")
                vtxt = f"（{', '.join(variant)}）" if variant else ""
                lines += [
                    "\\begin{figure}[h]",
                    "\\centering",
                    f"\\img[width=0.95\\linewidth]{{{str(rel)}}}",
                    f"\\caption{{\\path{{{fp.name}}}{vtxt}}}",
                    "\\end{figure}",
                    "",
                ]
        lines.append("\\clearpage\n")

    lines += ["\\end{document}", ""]
    return "\n".join(lines)


def main() -> None:
    groups = _load_groups(_read_lines(DEFAULT_INPUT))
    tex = _render_tex(groups)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(tex, encoding="utf-8")
    print(f"Wrote: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
