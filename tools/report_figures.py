"""Figures and tables of the report, from the test predictions saved by our runs (no training, no selection).

Recreates the paper's results figures with our retrained models:
  fig8_confusion.pdf   Fig. 8: confusion matrices (AEID RF, AEID CNN, TOW-IDS RF, TOW-IDS CNN), mean of the
                       5 fold models; the TOW-IDS CNN is split into folds 0-3 and fold 4 (the fold that did not
                       learn CAN Replay)
  fig9_roc.pdf         Fig. 9: ROC curves of both stages on both datasets (one thin line per fold)
and adds the figures of our modifications:
  fig_window_search.pdf   SOME/IP CSV: validation AUC per window size, arrival order x per conversation
  fig_feedback_attack.pdf TOW-IDS RF: test recall per attack and false positive rate, paper RF x feedback x control
  fig_message_level.pdf   SOME/IP CSV: message-level AUC with 95% bootstrap intervals (numbers from
                          tools/message_level_eval.py, job of 14/11)
  fig_time.pdf            detection time per window on one CPU core (numbers from jobs/time_cpu.sbatch)
It also prints the rows of Tables 7 and 8 (mean ± std over the 5 folds) for the report.

Test runs of the paper's models are found automatically: for each dataset and model, the EARLIEST test run
whose number of test windows matches (AEID 1,870,489; TOW-IDS 791,566). The later TOW-IDS RF tests are the
feedback variants; pass them with --feedback-test and --control-test.

Usage (on the cluster, see jobs/report_figures.sbatch): python tools/report_figures.py --out ~/outputs/report_figures
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,  # noqa: E402
                             roc_auc_score, roc_curve)

# Validated reference palette (categorical slots 1-4, light surface); sequential blue for the matrices
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})
TOW_CLASSES = ["C_D", "C_R", "F_I", "M_F", "Normal", "P_I"]
N_TEST = {"aeid": 1_870_489, "tow": 791_566}
MULTI = "data/tow/processed/y_test_TOW_IDS_dataset_multi_class_Wsize_44_Cols_116_Wslide_1_MC_True_sumX_False.csv"


def save(fig, out, name):
    """PDF for the report (vector), PNG for the slides"""
    fig.savefig(f"{out}/{name}.pdf")
    fig.savefig(f"{out}/{name}.png", dpi=200)


def fold_of(path):
    return int(re.search(r"fold_(\d+)", os.path.basename(path)).group(1))


def load_run(metrics_dir):
    """[(proba, y_true)] per fold; binary RF -> attack column, binary CNN -> single column, multiclass -> 6 columns"""
    paths = sorted(glob.glob(f"{metrics_dir}/*test_proba*.npz"), key=fold_of)
    out = []
    for p in paths:
        d = np.load(p)
        proba, y = d["proba"], np.asarray(d["y_true"])
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] == 1:
            proba = proba.ravel()
        if y.ndim == 2 and y.shape[1] > 1:
            y = y.argmax(axis=1)
        out.append((proba, y.ravel().astype(int)))
    return out


def find_earliest(outputs, kind, n, n_outputs):
    for run in sorted(glob.glob(f"{outputs}/*_{kind}")):
        files = glob.glob(f"{run}/metrics/*test_proba*fold_0*.npz")
        if not files:
            continue
        d = np.load(files[0], mmap_mode="r")
        cols = 1 if d["proba"].ndim == 1 else d["proba"].shape[1]
        if len(d["y_true"]) == n and (cols == n_outputs or (n_outputs == 1 and cols == 2)):
            return f"{run}/metrics"
    raise SystemExit(f"no {kind} run with {n} test windows and {n_outputs} output(s) in {outputs}")


def binary_scores(proba):
    """Attack score: the probability for binary models; 1 - P(Normal) for the 6-output CNN"""
    return 1 - proba[:, 4] if proba.ndim == 2 else proba


def binary_metrics(folds, multiclass=False):
    rows = []
    for proba, y in folds:
        if multiclass:
            pred = proba.argmax(axis=1)
            rows.append([accuracy_score(y, pred), precision_score(y, pred, average="macro", zero_division=0),
                         recall_score(y, pred, average="macro"), f1_score(y, pred, average="macro"),
                         roc_auc_score((y != 4).astype(int), binary_scores(proba))])
        else:
            pred = (proba >= 0.5).astype(int)
            rows.append([accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred), f1_score(y, pred),
                         roc_auc_score(y, proba)])
    return np.array(rows)


def table_rows(runs):
    print("\nTables 7 and 8 (test, mean ± std over the 5 fold models): accuracy, precision, recall, F1, ROC-AUC")
    print("TOW-IDS CNN: accuracy from the predicted class; precision, recall and F1 macro over the 6 classes;"
          " ROC-AUC of attack x normal with score 1 - P(Normal)")
    for name, (folds, multi) in runs.items():
        r = binary_metrics(folds, multi)
        cells = " & ".join(f"{m:.4f} $\\pm$ {s:.4f}" for m, s in zip(r.mean(0), r.std(0)))
        print(f"{name}: {cells} \\\\")
        if multi:
            print("   per fold accuracy: " + ", ".join(f"{v:.4f}" for v in r[:, 0]))


def plot_matrix(ax, cm, labels, title):
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#184f95"]),
              vmin=0, vmax=1)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,.0f}".replace(",", "."), ha="center", va="center", fontsize=6 if len(labels) > 2 else 7,
                    color="white" if share[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)


def fig8(runs, out):
    def mean_cm(folds, labels, multi):
        cms = []
        for proba, y in folds:
            pred = proba.argmax(axis=1) if multi else (proba >= 0.5).astype(int)
            cms.append(confusion_matrix(y, pred, labels=range(len(labels))))
        return np.mean(cms, axis=0)

    fig = plt.figure(figsize=(7.16, 5.6))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.35])
    panels = [("aeid_rf", ["Normal", "Ataque"], False, "(a) AEID, estágio 1 (RF)"),
              ("aeid_cnn", ["Normal", "Ataque"], False, "(b) AEID, estágio 2 (CNN)"),
              ("tow_rf", ["Normal", "Ataque"], False, "(c) TOW-IDS, estágio 1 (RF)")]
    for k, (key, labels, multi, title) in enumerate(panels):
        plot_matrix(fig.add_subplot(grid[0, 2 * k:2 * k + 2]), mean_cm(runs[key][0], labels, multi), labels, title)
    folds = runs["tow_cnn"][0]
    plot_matrix(fig.add_subplot(grid[1, 0:3]), mean_cm(folds[:4], TOW_CLASSES, True), TOW_CLASSES,
                "(d) TOW-IDS, estágio 2 (CNN), folds 0–3")
    plot_matrix(fig.add_subplot(grid[1, 3:6]), mean_cm(folds[4:], TOW_CLASSES, True), TOW_CLASSES,
                "(e) TOW-IDS, estágio 2 (CNN), fold 4")
    fig.tight_layout()
    save(fig, out, "fig8_confusion")
    plt.close(fig)


def fig9(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, ds, title in zip(axes, ("aeid", "tow"), ("(a) AEID", "(b) TOW-IDS")):
        for key, color, name in ((f"{ds}_rf", BLUE, "Estágio 1 (RF)"), (f"{ds}_cnn", ORANGE, "Estágio 2 (CNN)")):
            aucs = []
            for k, (proba, y) in enumerate(runs[key][0]):
                score, yb = binary_scores(proba), (y != 4).astype(int) if proba.ndim == 2 else y
                fpr, tpr, _ = roc_curve(yb, score, drop_intermediate=True)
                aucs.append(roc_auc_score(yb, score))
                ax.plot(fpr, tpr, color=color, lw=1.0, alpha=0.8, label=None)
            ax.plot([], [], color=color, lw=2, label=f"{name}, AUC {np.mean(aucs):.4f}")
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Taxa de falsos positivos")
        ax.set_ylabel("Taxa de verdadeiros positivos")
        ax.set_title(title, color=INK)
        ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "fig9_roc")
    plt.close(fig)


def fig_window_search(outputs, out):
    rows = []
    for grouping in ("arrival", "conversation"):
        for w in (10, 20, 30, 44):
            rf = f"{outputs}/window_search/rf_{grouping}_w{w}.csv"
            cnn = sorted(glob.glob(f"{outputs}/*_pytorch_train*/metrics/val_metrics_PrunedCNNIDS_{grouping}_w{w}_*.csv"))
            if os.path.exists(rf) and cnn:
                a, b = pd.read_csv(rf).val_auc, pd.read_csv(cnn[-1]).roc_auc
                rows.append((grouping, w, a.mean(), a.std(ddof=0), b.mean(), b.std(ddof=0)))
    if not rows:
        print("window search results not found, skipping fig_window_search")
        return
    df = pd.DataFrame(rows, columns=["grouping", "w", "rf", "rf_sd", "cnn", "cnn_sd"])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
    for ax, model, title in zip(axes, ("rf", "cnn"), ("(a) Estágio 1 (RF)", "(b) Estágio 2 (CNN)")):
        for grouping, color, name, marker in (("arrival", BLUE, "Ordem de chegada (artigo)", "o"),
                                              ("conversation", ORANGE, "Por conversa (proposta)", "s")):
            d = df[df.grouping == grouping]
            ax.errorbar(d.w, d[model], yerr=d[f"{model}_sd"], color=color, marker=marker, ms=5, capsize=2,
                        lw=1.5, elinewidth=0.8, label=name)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_xticks([10, 20, 30, 44])
        ax.set_xlabel("Tamanho da janela (diferenças entre mensagens)")
        ax.set_title(title, color=INK)
    axes[0].set_ylabel("AUC de validação (média ± desvio, 5 folds)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, out, "fig_window_search")
    plt.close(fig)


def fig_feedback(base, feedback, control, out):
    classes = pd.read_csv(MULTI)["Class"].astype(str).values
    attacks = ["C_D", "C_R", "F_I", "M_F", "P_I"]
    series = [("RF do artigo", base, INK2), ("Controle (peso 5)", control, BLUE), ("Retreino (peso 5)", feedback, ORANGE)]
    values = {}
    print("\nTOW-IDS RF on the test set, mean of 5 folds: recall per attack, false positive rate, F1")
    for name, folds, _ in series:
        rec, fpr, f1 = [], [], []
        for p, y in folds:
            if len(p) != len(classes) or not np.array_equal(y == 0, classes == "Normal"):
                raise SystemExit("RF test windows do not match the multiclass labels")
            pred = p >= 0.5
            rec.append([pred[classes == a].mean() for a in attacks])
            fpr.append(pred[classes == "Normal"].mean())
            f1.append(f1_score(y, pred))
        values[name] = (np.mean(rec, 0), np.mean(fpr), np.mean(f1), np.std(f1))
        r, fp, m, s = values[name]
        print(f"  {name:>18}: " + "  ".join(f"{a} {v:.4f}" for a, v in zip(attacks, r))
              + f"  FPR {fp:.4f}  F1 {m:.4f} ± {s:.4f}")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(attacks) + 1)
    width = 0.26
    for k, (name, _, color) in enumerate(series):
        r, fp, _, _ = values[name]
        ax.bar(x + (k - 1) * width, list(r) + [fp], width - 0.03, color=color, label=name)
    ax.set_xticks(x, attacks + ["FPR\n(normais)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Taxa no teste (média de 5 folds)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig_feedback_attack")
    plt.close(fig)


def fig_message_level(out):
    # tools/message_level_eval.py, 14/11: message-level AUC (max of the windows) and 95% bootstrap intervals
    data = [("RF", 0.5205, 0.4925, 0.5501, 0.7003, 0.6655, 0.7393), ("CNN", 0.5467, 0.5099, 0.5862, 0.7185, 0.6815, 0.7576)]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for k, (model, a, alo, ahi, c, clo, chi) in enumerate(data):
        for dx, v, lo, hi, color, name in ((-0.12, a, alo, ahi, BLUE, "Ordem de chegada, janela 44 (artigo)"),
                                           (0.12, c, clo, chi, ORANGE, "Por conversa, janela 10 (proposta)")):
            ax.errorbar(k + dx, v, yerr=[[v - lo], [hi - v]], fmt="o", color=color, ms=6, capsize=3,
                        label=name if k == 0 else None)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xticks([0, 1], ["Estágio 1 (RF)", "Estágio 2 (CNN)"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.45, 0.8)
    ax.set_ylabel("AUC por mensagem (IC 95%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig_message_level")
    plt.close(fig)


def fig_time(out):
    # jobs/time_cpu.sbatch with 1 core (job 30418) and the paper's Table 10, microseconds per window
    rows = [("AEID RF", 75.9, 107), ("AEID CNN", 3796, 4510), ("TOW RF", 154.3, 221), ("TOW CNN", 3742, 4757),
            ("SOME/IP RF\n(janela 44)", 40.9, None), ("SOME/IP CNN\n(janela 44)", 3883, None),
            ("SOME/IP RF\n(conversa 10)", 39.9, None), ("SOME/IP CNN\n(conversa 10)", 593.6, None)]
    fig, ax = plt.subplots(figsize=(7.16, 2.4))
    x = np.arange(len(rows))
    ours = ax.bar(x - 0.18, [r[1] for r in rows], 0.34, color=BLUE, label="Nossa medição (1 núcleo)")
    paper = [(i, r[2]) for i, r in enumerate(rows) if r[2]]
    theirs = ax.bar([i + 0.18 for i, _ in paper], [v for _, v in paper], 0.34, color=INK2, label="Artigo (Tabela 10)")
    for bars in (ours, theirs):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12, f"{b.get_height():,.0f}".replace(",", "."),
                    ha="center", va="bottom", fontsize=6, color=INK2)
    ax.axhline(1000, color=ORANGE, lw=1, ls="--", label="Limite de 1.000 µs citado no artigo")
    ax.set_yscale("log")
    ax.set_ylim(20, 20000)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("µs por janela (escala log)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6, ncol=3)
    fig.tight_layout()
    save(fig, out, "fig_time")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--outputs", default=os.environ.get("IDS_ARTIFACTS_DIR", os.path.expanduser("~/outputs")))
    parser.add_argument("--feedback-test", help="metrics folder of the feedback_w5 test run")
    parser.add_argument("--control-test", help="metrics folder of the control_w5 test run")
    args = parser.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    found = {"aeid_rf": find_earliest(args.outputs, "sklearn_test", N_TEST["aeid"], 1),
             "aeid_cnn": find_earliest(args.outputs, "pytorch_test", N_TEST["aeid"], 1),
             "tow_rf": find_earliest(args.outputs, "sklearn_test", N_TEST["tow"], 1),
             "tow_cnn": find_earliest(args.outputs, "pytorch_test", N_TEST["tow"], 6)}
    for k, v in found.items():
        print(f"{k}: {v}")
    runs = {k: (load_run(v), k == "tow_cnn") for k, v in found.items()}

    table_rows({"AEID RF": runs["aeid_rf"], "AEID CNN": runs["aeid_cnn"],
                "TOW-IDS RF": runs["tow_rf"], "TOW-IDS CNN": runs["tow_cnn"]})
    fig8(runs, out)
    fig9(runs, out)
    fig_window_search(args.outputs, out)
    if args.feedback_test and args.control_test:
        fig_feedback(runs["tow_rf"][0], load_run(args.feedback_test), load_run(args.control_test), out)
    fig_message_level(out)
    fig_time(out)
    print(f"\nfigures written to {out}: {sorted(os.listdir(out))}")


if __name__ == "__main__":
    main()
