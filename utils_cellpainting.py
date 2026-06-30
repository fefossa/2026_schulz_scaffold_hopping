import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ------------------------------
# Defaults / constants
# ------------------------------
COMP_COL = "Compound_Id"
COMP_ID_COL = "Compound_Id"
CONC_COL = "Conc_uM"
TOXIC_COL = "Toxic"
TIME_COL = None  # e.g., 'Time Point [h]' or None
TIME_KEEP = None  # e.g., 20 or None
FEATURE_PREFIX = "Median_"

COMPARTMENTS_ORDER = ["Median_Cells", "Median_Nuclei", "Median_Cytoplasm"]
CHANNELS_ORDER = ["Mito", "Ph_golgi", "Syto", "ER", "Hoechst"]


# ------------------------------
# Basic utilities
# ------------------------------
def filter_window(
    df: pd.DataFrame,
    conc_max: float = 10,
    drop_toxic: bool = True,
    comp_col: str = COMP_ID_COL,
    conc_col: str = CONC_COL,
    toxic_col: str = TOXIC_COL,
    time_col: Optional[str] = TIME_COL,
    time_keep: Optional[float] = TIME_KEEP,
) -> pd.DataFrame:
    """Filter to rows with conc<=conc_max and optionally non-toxic, and optional fixed time point."""
    df = df.copy()
    assert comp_col in df.columns, f"Missing {comp_col}"
    assert conc_col in df.columns, f"Missing {conc_col}"
    assert toxic_col in df.columns, f"Missing {toxic_col}"

    mask = df[conc_col] <= conc_max
    if drop_toxic:
        mask &= df[toxic_col] == False
    if time_col and time_keep is not None and time_col in df.columns:
        mask &= df[time_col] == time_keep
    return df.loc[mask].copy()


def pick_feature_cols(df: pd.DataFrame, prefix: str = FEATURE_PREFIX) -> List[str]:
    """Return feature columns that start with the given prefix."""
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    if not feat_cols:
        raise ValueError(f"No feature columns starting with '{prefix}' were found.")
    return feat_cols


def parse_compartment(feat: str) -> str:
    for comp in COMPARTMENTS_ORDER:
        if feat.startswith(comp):
            return comp
    return "Other"


def parse_channel(feat: str) -> str:
    for ch in CHANNELS_ORDER:
        if ch in feat:
            return ch
    return "Other"


# ------------------------------
# Aggregation
# ------------------------------
def aggregate_and_order(
    df: pd.DataFrame,
    aggfunc: str = "median",
    by_concentration: bool = False,
    comp_col: str = COMP_ID_COL,
    conc_col: str = CONC_COL,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Aggregate feature columns per compound.
    If by_concentration=True, aggregate per (Compound, Concentration) so conc_col is preserved.

    Returns:
        agg_df: aggregated dataframe with id columns kept (comp_col and optionally conc_col)
        comp_map, ch_map: dicts for compartment/channel parsed from feature column names
    """
    feat_cols = pick_feature_cols(df)

    group_keys = [comp_col] + (
        [conc_col] if by_concentration and conc_col in df.columns else []
    )

    if aggfunc == "median":
        agg_df = df.groupby(group_keys, as_index=False)[feat_cols].median()
    elif aggfunc == "mean":
        agg_df = df.groupby(group_keys, as_index=False)[feat_cols].mean()
    elif aggfunc == "max":
        agg_df = df.groupby(group_keys, as_index=False)[feat_cols].max()
    else:
        raise ValueError("aggfunc must be 'median', 'mean', or 'max'.")

    comp_map = {c: parse_compartment(c) for c in feat_cols}
    ch_map = {c: parse_channel(c) for c in feat_cols}
    comp_rank = {k: i for i, k in enumerate(COMPARTMENTS_ORDER)}
    ch_rank = {k: i for i, k in enumerate(CHANNELS_ORDER)}

    def sort_key(col: str):
        comp = comp_map.get(col)
        ch = ch_map.get(col)
        cr = comp_rank.get(comp, 10_000)
        hr = ch_rank.get(ch, 10_000)
        return (cr, hr, col)

    ordered_feat_cols = sorted(feat_cols, key=sort_key)
    lead_cols = [c for c in [comp_col, conc_col] if c in agg_df.columns]
    agg_df = agg_df[lead_cols + ordered_feat_cols]
    return agg_df, comp_map, ch_map


def filter_to_highest_conc_per_compound(
    df,
    conc_max=10,
    drop_toxic=True,
    agg='median',
    select_max=True
):
    """
    If select_max=True: keep, for each compound, only rows at the highest
        concentration <= conc_max.
    If select_max=False: keep all concentrations <= conc_max.

    drop_toxic: if True, drop toxic rows first.
    Returns (df_selected, chosen_conc_series or None).
    """
    df = df.copy()

    # base window
    mask = df[CONC_COL] <= conc_max
    if drop_toxic and TOXIC_COL in df.columns:
        mask &= (df[TOXIC_COL] == False)
    win = df[mask].copy()

    if win.empty:
        print("No rows found under the given filters.")
        return win, None

    if select_max:
        # highest conc <= conc_max per compound
        top_conc = (
            win.groupby(COMP_COL, as_index=True)[CONC_COL]
               .max()
        )

        # keep only rows at that specific concentration
        selected = (
            win.merge(top_conc.rename('ChosenConc'), left_on=COMP_COL, right_index=True)
               .loc[lambda x: x[CONC_COL] == x['ChosenConc']]
               .drop(columns=['ChosenConc'])
               .copy()
        )

        # summary
        summary = top_conc.sort_index()
        print(f"Chosen highest concentration per compound (<= {conc_max} µM), "
              f"after toxic={'dropped' if drop_toxic else 'kept'}; planned agg: {agg}")
        for comp, conc in summary.items():
            print(f"  - {comp}: {conc:g} µM")

        return selected, top_conc
    else:
        # just keep all concentrations
        print(f"Keeping all concentrations <= {conc_max} µM, "
              f"after toxic={'dropped' if drop_toxic else 'kept'}; planned agg: {agg}")
        return win, None


# ------------------------------
# Label helpers
# ------------------------------
def _normalize_compound_id(val) -> str:
    """Return id as string without trailing .0 (handles int/float/str)."""
    if pd.isna(val):
        return ""
    try:
        fv = float(val)
        if fv.is_integer():
            return str(int(fv))
        return str(val)
    except Exception:
        return str(val)


def _format_row_label(comp_id, conc, rename_dict, row_label_fmt: str) -> str:
    name = _normalize_compound_id(comp_id)
    if rename_dict:
        name = rename_dict.get(name, name)
    if "{conc" in row_label_fmt and conc is None:
        conc = ""
    return row_label_fmt.format(name=name, conc=conc)


# ------------------------------
# Matrix / series builders
# ------------------------------
def to_feature_matrix(
    agg_df: pd.DataFrame,
    comp_map: Dict[str, str],
    by_concentration: bool = False,
    comp_col: str = COMP_ID_COL,
    conc_col: str = CONC_COL,
    rename_dict: Optional[Dict[str, str]] = None,
    row_label_fmt: str = "{name}",
) -> pd.DataFrame:
    """Build a matrix (rows x features) from aggregated df, keeping only feature columns."""
    feat_cols = [c for c in agg_df.columns if c in comp_map]

    if by_concentration and conc_col in agg_df.columns:
        rows = []
        idx_labels = []
        key_cols = [comp_col, conc_col]
        for _, r in agg_df[key_cols].drop_duplicates().iterrows():
            sub = agg_df[
                (agg_df[comp_col] == r[comp_col]) & (agg_df[conc_col] == r[conc_col])
            ][feat_cols]
            if sub.empty:
                continue
            rows.append(sub.iloc[0].values)
            idx_labels.append(
                _format_row_label(r[comp_col], r[conc_col], rename_dict, row_label_fmt)
            )
        mat = pd.DataFrame(rows, index=idx_labels, columns=feat_cols)
    else:
        df2 = agg_df.set_index(comp_col)[feat_cols].copy()
        new_index = [
            _format_row_label(i, None, rename_dict, "{name}") for i in df2.index
        ]
        df2.index = new_index
        mat = df2

    mat = mat.sort_index()
    return mat


def induction_series(
    df_filtered: pd.DataFrame,
    agg_kind: str = "max",
    by_concentration: bool = False,
    comp_col: str = COMP_ID_COL,
    conc_col: str = CONC_COL,
    induction_col: str = "Induction [%]",
    rename_dict: Optional[Dict[str, str]] = None,
    row_label_fmt: str = "{name} ({conc:g} µM)",
) -> pd.Series:
    """Build an induction Series aligned to the matrix row labels."""
    if by_concentration and conc_col in df_filtered.columns:
        gcols = [comp_col, conc_col]
    else:
        gcols = [comp_col]

    if agg_kind == "max":
        s = df_filtered.groupby(gcols)[induction_col].max()
    elif agg_kind == "median":
        s = df_filtered.groupby(gcols)[induction_col].median()
    else:
        raise ValueError("agg_kind must be 'max' or 'median'")

    if by_concentration and conc_col in df_filtered.columns:
        s = s.reset_index()
        labels = [
            _format_row_label(row[comp_col], row[conc_col], rename_dict, row_label_fmt)
            for _, row in s.iterrows()
        ]
        s.index = labels
        s = s.iloc[:, -1]
    else:
        s.index = [_format_row_label(i, None, rename_dict, "{name}") for i in s.index]

    return s.sort_index().astype(float), s


# ------------------------------
# Plot
# ------------------------------
def plot_heatmap_with_induction_bars(
    matrix: pd.DataFrame,
    comp_map: Dict[str, str],
    title: str,
    induction_s: pd.Series,
    vmin: float = -3,
    vmax: float = 3,
    fs_heat: int = 8,
    fs_comp: int = 9,
    figsize: Tuple[float, float] = (14, 6),
    induction_xlim: Tuple[float, float] = (0, 100),
    bar_color: str = "0.55",
    bar_edgecolor: str = "0.25",
    show_value_labels: bool = True,
    save_path: Optional[str] = None,
):
    """Plot feature heatmap (columns-only features) + bottom compartment band + right-side induction bars."""
    feature_cols = [c for c in matrix.columns if c in comp_map]
    if not feature_cols:
        raise ValueError("Matrix has no feature columns matching comp_map.")
    matrix = matrix[feature_cols].copy()

    common = matrix.index.intersection(induction_s.index)
    if len(common) == 0:
        raise ValueError(
            "No common row labels between matrix and induction_s. Check rename/row_label_fmt."
        )
    matrix = matrix.loc[common].sort_index()
    induction_s = induction_s.loc[common].sort_values()
    matrix = matrix.loc[induction_s.index]

    X = matrix.copy()
    med = X.median(axis=0)
    mad = (X - med).abs().median(axis=0).replace(0, np.nan)
    Z = ((X - med) / mad).fillna(0).clip(vmin, vmax)

    comp_map_clean = {
        col: comp_map.get(col, "Other").replace("Median_", "") for col in Z.columns
    }

    cols = list(Z.columns)
    comp_blocks = []
    start = 0
    prev = comp_map_clean[cols[0]]
    for i, c in enumerate(cols[1:], 1):
        cur = comp_map_clean[c]
        if cur != prev:
            comp_blocks.append((prev, start, i - 1))
            start = i
            prev = cur
    comp_blocks.append((prev, start, len(cols) - 1))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=3,
        ncols=2,
        width_ratios=[22, 6],
        height_ratios=[20, 1.8, 1.2],
        hspace=0.08,
        wspace=0.18,
    )
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[1, 0], sharex=ax_hm)
    ax_cbar = fig.add_subplot(gs[2, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_hm)

    sns.heatmap(Z, cmap="vlag", center=0, ax=ax_hm, cbar=False)
    ax_hm.set_ylabel("Compound", fontsize=fs_heat)
    ax_hm.set_title(title, fontsize=fs_heat + 4)
    ax_hm.tick_params(axis="both", labelsize=fs_heat)
    ax_hm.set_xticklabels([])
    ax_hm.set_yticks(np.arange(len(matrix)) + 0.5)
    ax_hm.set_yticklabels(matrix.index, fontsize=fs_heat)

    for _, _, end in comp_blocks[:-1]:
        ax_hm.vlines(end + 1, -0.5, len(Z.index) - 0.5, colors="k", linewidth=0.6)

    ax_comp.set_ylim(0, 1)
    ax_comp.axis("off")
    for comp, s, e in comp_blocks:
        xc = (s + e + 1) / 2.0
        ax_comp.plot([s, e + 1], [0.88, 0.88], color="k", linewidth=1.2, clip_on=False)
        ax_comp.text(
            xc,
            0.42,
            comp,
            ha="center",
            va="center",
            fontsize=fs_comp,
            fontweight="bold",
        )

    norm_hm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm_hm = plt.cm.ScalarMappable(cmap="vlag", norm=norm_hm)
    cb_hm = fig.colorbar(sm_hm, cax=ax_cbar, orientation="horizontal")
    cb_hm.set_label("Feature z-score (robust)", fontsize=fs_heat)
    cb_hm.ax.tick_params(labelsize=fs_heat)

    y_positions = np.arange(len(induction_s))
    ax_bar.barh(
        y=y_positions,
        width=induction_s.values,
        height=1.0,
        color=bar_color,
        edgecolor=bar_edgecolor,
    )
    ax_bar.set_ylim(ax_hm.get_ylim())
    ax_bar.set_xlim(*induction_xlim)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Induction (%)", fontsize=fs_heat)
    ax_bar.tick_params(axis="both", labelsize=fs_heat)
    ax_bar.set_yticks([])
    ax_bar.yaxis.set_ticks_position('none')

    ax_hm.set_yticks(np.arange(len(matrix)) + 0.5)
    ax_hm.set_yticklabels(matrix.index, fontsize=fs_heat)

    if show_value_labels:
        span = induction_xlim[1] - induction_xlim[0]
        for i, val in enumerate(induction_s.values):
            ax_bar.text(
                val + 0.01 * span,
                i,
                f"{val:.0f}",
                va="center",
                ha="left",
                fontsize=fs_heat,
            )

    if save_path:
        plt.savefig(save_path, format=save_path.split(".")[-1], bbox_inches="tight")
    plt.show()
