# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:15:03 2026

@author: tameem
"""

import pandas as pd

#df = pd.read_csv('D:\\Nigrasteer_recon.csv')
df_SN = pd.read_csv('D:\\new code\\combinednigrasteer_SN_2026.csv')
df_SN['Z_mm'] = -df_SN['Z_mm']
df_clinscore = pd.read_csv('D:\\new code\\nigrasteer_clinscores_2026.csv')
df_connect = pd.read_csv('D:\\new code\\connectiv_HMAT_SNcombined.csv')


#%%

# Calculate score differences
df_clinscore['StimOFF_vs_STN'] = df_clinscore['StimOFF'] - df_clinscore['STN']
df_clinscore['StimOFF_vs_STN+SNr'] = df_clinscore['StimOFF'] - df_clinscore['STN+SNr']
df_clinscore['STN_vs_STN+SNr'] = df_clinscore['STN'] - df_clinscore['STN+SNr']
print(df_clinscore)

#%% raincloud plot (Figure 2) - subcohort filter (responders only) - combined plots

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Fixed random generator seed for exact bootstrap reproducibility
rng = np.random.default_rng(seed=42)
n_bootstraps = 10000

def compute_paired_bootstrap_ci(vec1, vec2, n_boots=10000, alpha=0.05):
    """Calculates non-parametric bootstrap 95% CI for the mean paired difference (vec2 - vec1)."""
    diffs = np.array(vec2) - np.array(vec1)
    n_samples = len(diffs)
    boot_means = np.zeros(n_boots)
    
    for i in range(n_boots):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_means[i] = np.mean(diffs[boot_idx])
        
    ci_low, ci_high = np.percentile(boot_means, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    mean_diff = np.mean(diffs)
    return mean_diff, ci_low, ci_high

def compute_exact_paired_permutation_p(vec1, vec2):
    """Calculates exact two-sided paired permutation p-value via 2^N sign flips."""
    diffs = np.array(vec2) - np.array(vec1)
    n = len(diffs)
    obs_mean_diff = np.mean(diffs)
    obs_abs_diff = np.abs(obs_mean_diff)
    
    # Generate all 2^N sign reassignments (+1 / -1)
    sign_flips = list(itertools.product([-1, 1], repeat=n))
    count_extreme = 0
    total_perms = len(sign_flips)
    
    for signs in sign_flips:
        perm_diff = np.mean(diffs * np.array(signs))
        if np.abs(perm_diff) >= obs_abs_diff - 1e-12:
            count_extreme += 1
            
    p_perm = count_extreme / total_perms
    return p_perm, total_perms

def compute_cohens_d_z(vec1, vec2):
    """Calculates Cohen's dz for paired samples: mean_diff / std_diff."""
    diffs = np.array(vec2) - np.array(vec1)
    std_diff = np.std(diffs, ddof=1)
    if std_diff == 0:
        return 0.0
    return np.mean(diffs) / std_diff

def run_panel_statistics(data_x):
    """Computes full suite of statistical metrics for a panel."""
    vec1 = data_x[0]
    vec2 = data_x[1]
    n_samples = len(vec1)
    df_val = n_samples - 1
    
    # 1. Paired t-test
    t_stat, p_analytic = ttest_rel(vec1, vec2)
    
    # 2. Exact Permutation Test (2^N sign flips)
    p_perm, total_perms = compute_exact_paired_permutation_p(vec1, vec2)
    
    # 3. Effect Estimates
    mean_diff, ci_low, ci_high = compute_paired_bootstrap_ci(vec1, vec2, n_boots=n_bootstraps)
    cohens_dz = compute_cohens_d_z(vec1, vec2)
    
    return {
        "n": n_samples,
        "df": df_val,
        "t_stat": t_stat,
        "p_analytic": p_analytic,
        "p_perm": p_perm,
        "total_perms": total_perms,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_dz": cohens_dz
    }

def apply_holm_bonferroni(panel_results):
    """Applies Holm-Bonferroni step-down correction across m=3 panels."""
    m_tests = len(panel_results)
    
    # Sort panel keys by unadjusted p_perm
    sorted_keys = sorted(panel_results.keys(), key=lambda k: panel_results[k]["p_perm"])
    prev_p_holm = 0.0
    
    for rank_idx, key in enumerate(sorted_keys):
        p_raw = panel_results[key]["p_perm"]
        multiplier = m_tests - rank_idx
        p_candidate = p_raw * multiplier
        
        p_holm = max(p_candidate, prev_p_holm)
        p_holm = min(1.0, p_holm)
        prev_p_holm = p_holm
        
        panel_results[key]["p_holm"] = p_holm

def draw_raincloud_subplot(ax, df_source, data_x, title, x_labels, grp_values, panel_label, stats_res):
    palette_mapping = {
        "StimOFF": "#E0E0E0",      # Light Silver
        "STN-DBS": "#A8B8E8",      # Periwinkle
        "STN+SN-DBS": "#8FBC8F",   # Muted Green
    }
    box_colors = [palette_mapping.get(label, "gray") for label in x_labels]

    color_improved    = "#D6604D"  # Vermillion
    color_minor_chang = "#E6AD4A"  # Gold
    color_deteriorat  = "#404040"  # Ink Black

    # Boxplot data
    bp = ax.boxplot(data_x, patch_artist=True, vert=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(bp[element], color="dimgray")

    # Violinplot data
    vp = ax.violinplot(data_x, vert=True, widths=0.7, showextrema=False)
    for idx, body in enumerate(vp["bodies"]):
        if idx == 0:
            body.get_paths()[0].vertices[:, 0] = np.clip(
                body.get_paths()[0].vertices[:, 0], idx, idx + 1
            )
        else:
            body.get_paths()[0].vertices[:, 0] = np.clip(
                body.get_paths()[0].vertices[:, 0], idx + 1, idx + 2
            )

        body.set_facecolor(box_colors[idx])
        body.set_alpha(0.4)

    # Scatterplot data with connecting lines
    for idx in range(len(df_source.index)):
        x = np.arange(1, len(data_x) + 1).astype(float)

        for i in range(len(data_x)):
            if i == 0:
                line_color = (
                    color_improved
                    if grp_values.iloc[idx] == 2
                    else (color_minor_chang if grp_values.iloc[idx] == 1 else color_deteriorat)
                )
            elif i == 1:
                line_color = (
                    color_improved
                    if grp_values.iloc[idx] == 2
                    else (color_minor_chang if grp_values.iloc[idx] == 1 else color_deteriorat)
                )
            else:
                line_color = (
                    color_improved
                    if grp_values.iloc[idx] == 2
                    else (color_minor_chang if grp_values.iloc[idx] == 1 else color_deteriorat)
                )

            ax.scatter(x[i], data_x[i].iloc[idx], s=12, c="black", zorder=3)

            if i < len(data_x) - 1:
                ax.plot(
                    [x[i], x[i + 1]],
                    [data_x[i].iloc[idx], data_x[i + 1].iloc[idx]],
                    c=line_color,
                    linewidth=1.0,
                    alpha=0.6,
                    zorder=2,
                )

    # Journal Panel Labels
    ax.text(
        -0.05,
        1.10,
        panel_label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="right"
    )

    # Format axes & labels
    ax.set_xticks(np.arange(1, len(data_x) + 1, 1))
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=15)
    ax.set_ylabel("FoG-AC Score", fontweight="bold", fontsize=15)

    ax.figure.canvas.draw()
    y_start, y_end = ax.get_ylim()

    tick_start = np.floor(y_start / 2) * 2
    tick_end = np.ceil(y_end / 2) * 2
    tick_range = np.arange(tick_start, tick_end + 1, 2)

    ax.set_yticks(tick_range)
    ax.set_ylim(bottom=tick_start, top=tick_end)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)

    # Title & Subtitle Hierarchical Placement
    ax.set_title(title, pad=42, fontsize=12, fontweight="bold")
    
    # Text Annotation displaying exact p-values, Holm adjustment, Cohen's dz, and Bootstrap 95% CI
    stats_annotation = (
        f"Mean $\\Delta$: {stats_res['mean_diff']:+.2f} [Boot 95% CI: {stats_res['ci_low']:+.2f}, {stats_res['ci_high']:+.2f}]\n"
        f"Cohen's $d_z = {stats_res['cohens_dz']:.2f}$  |  $t({stats_res['df']}) = {stats_res['t_stat']:.2f}$\n"
        f"$p_{{\\mathrm{{analytic}}}} = {stats_res['p_analytic']:.4f}$  |  $p_{{\\mathrm{{perm}}}} = {stats_res['p_perm']:.4f}$  |  $p_{{\\mathrm{{Holm}}}} = {stats_res['p_holm']:.4f}$"
    )
    ax.text(
        0.5, 
        1.01, 
        stats_annotation, 
        transform=ax.transAxes, 
        fontsize=10, 
        fontweight="normal", 
        color="dimgray", 
        ha="center",
        va="bottom"
    )
    ax.grid(False)


# --- SHARED GLOBAL LEGEND CONFIGURATION ---
color_improved    = "#D6604D"
color_minor_chang = "#E6AD4A"
color_deteriorat  = "#404040"

global_legend_elements = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color_improved, markeredgecolor='none', markersize=11, label='Improvement'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color_minor_chang, markeredgecolor='none', markersize=11, label='Minor Change'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color_deteriorat, markeredgecolor='none', markersize=11, label='Deterioration')
]

# =====================================================================
# RUN FUNCTION FOR PIPELINE
# =====================================================================
def run_paired_condition_analysis(df_input, run_title):
    print("\n" + "="*85)
    print(f"   {run_title.upper()}   ")
    print("="*85)
    
    df_red = df_input[df_input["grp1"] == 2]
    
    panels_data = {
        "A": {
            "df": df_input,
            "x_data": [df_input["StimOFF"], df_input["STN"]],
            "title": "StimOFF to STN-DBS",
            "x_labels": ["StimOFF", "STN-DBS"],
            "grp": df_input["grp1"]
        },
        "B": {
            "df": df_red,
            "x_data": [df_red["StimOFF"], df_red["STN+SNr"]],
            "title": "StimOFF to STN+SN-DBS\n(Red Responders)",
            "x_labels": ["StimOFF", "STN+SN-DBS"],
            "grp": df_red["grp2"]
        },
        "C": {
            "df": df_red,
            "x_data": [df_red["STN"], df_red["STN+SNr"]],
            "title": "STN-DBS to STN+SN-DBS\n(Red Responders)",
            "x_labels": ["STN-DBS", "STN+SN-DBS"],
            "grp": df_red["grp3"]
        }
    }
    
    # 1. Compute Statistics
    results = {}
    for p_label, p_info in panels_data.items():
        results[p_label] = run_panel_statistics(p_info["x_data"])
        
    # 2. Apply Holm-Bonferroni Correction (m=3)
    apply_holm_bonferroni(results)
    
    # 3. Print Audit Table to Terminal
    audit_rows = []
    for p_label in ["A", "B", "C"]:
        r = results[p_label]
        t = panels_data[p_label]["title"].replace('\n', ' ')
        audit_rows.append({
            "Panel": p_label,
            "Comparison": t,
            "N": r["n"],
            "df": r["df"],
            "Mean Diff": f"{r['mean_diff']:+.2f}",
            "Boot 95% CI": f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]",
            "Cohen's dz": f"{r['cohens_dz']:.2f}",
            "t-stat": f"{r['t_stat']:.2f}",
            "p_analytic": f"{r['p_analytic']:.4f}",
            "p_perm": f"{r['p_perm']:.4f}",
            "p_Holm": f"{r['p_holm']:.4f}"
        })
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print("="*85)
    
    # 4. Draw Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.8))
    for idx, p_label in enumerate(["A", "B", "C"]):
        p_info = panels_data[p_label]
        draw_raincloud_subplot(
            axes[idx],
            p_info["df"],
            p_info["x_data"],
            p_info["title"],
            p_info["x_labels"],
            p_info["grp"],
            p_label,
            results[p_label]
        )
        
    fig.legend(handles=global_legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=True, facecolor='whitesmoke', edgecolor='none', prop={'weight': 'bold', 'size': 11})
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

# =====================================================================
# RUN PIPELINES
# =====================================================================
# Run 1: All Patients
run_paired_condition_analysis(df_clinscore, "Paired Analysis: All Patients")

# Run 2: Excluding Patient PD08 (Target N=8 Cohort)
df_filtered = df_clinscore[df_clinscore["Patient-ID"].str.upper() != "PD08"].copy()
run_paired_condition_analysis(df_filtered, "Paired Analysis: Excluding PD08 (N=8 Cohort)")

#%%

# Filter data for patients where grp3 is 2 and 0, and select patient IDs along with clinical scores
grp3_2_data = df_clinscore[df_clinscore['grp3'] == 2][['Patient-ID', 'STN_vs_STN+SNr']]
grp3_1_data = df_clinscore[df_clinscore['grp3'] == 1][['Patient-ID', 'STN_vs_STN+SNr']]
grp3_0_data = df_clinscore[df_clinscore['grp3'] == 0][['Patient-ID', 'STN_vs_STN+SNr']]

# Display patient IDs associated with grp3_2_scores
print("Patient IDs for grp3 == 2:")
print(grp3_2_data)

print("Patient IDs for grp3 == 1:")
print(grp3_1_data)

# Display patient IDs associated with grp3_0_scores
print("Patient IDs for grp3 == 0:")
print(grp3_0_data)


#%% 3D Coordinate Analysis (X, Y, Z) vs Absolute ΔFoG-AC (N=8 Cohort)

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, norm, theilslopes

# Set global font to Arial
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'

# =====================================================================
# 1. STRUCTURE ALIGNMENT & COHORT SELECTION (N=8)
# =====================================================================
df_SN_clean = df_SN.copy()
df_clinscore_clean = df_clinscore.copy()

df_SN_clean["Patient-ID_upper"] = df_SN_clean["Patient-ID"].astype(str).str.upper()
df_clinscore_clean["Patient-ID_upper"] = df_clinscore_clean["Patient-ID"].astype(str).str.upper()

# Target cohort defined by unique patients in df_SN minus PD08
target_patients = set(df_SN_clean["Patient-ID_upper"].unique()) - {"PD08"}

df_SN_cohort = df_SN_clean[df_SN_clean["Patient-ID_upper"].isin(target_patients)].copy()
df_clin_cohort = df_clinscore_clean[df_clinscore_clean["Patient-ID_upper"].isin(target_patients)].copy()

# Average coordinates per patient across df_SN entries
avg_coords = df_SN_cohort.groupby("Patient-ID_upper")[["X_mm", "Y_mm", "Z_mm"]].mean().reset_index()

# Merge coordinates with the pre-calculated STN_vs_STN+SNr difference
df_analysis = pd.merge(
    avg_coords,
    df_clin_cohort[["Patient-ID_upper", "STN_vs_STN+SNr"]].drop_duplicates(subset=["Patient-ID_upper"]),
    on="Patient-ID_upper"
)

y_data = df_analysis["STN_vs_STN+SNr"].values
patient_ids = df_analysis["Patient-ID_upper"].values
n_cohort = len(df_analysis)

# Explicit mapping from original patient IDs to Ptn01..Ptn08
sorted_pids = sorted(df_analysis["Patient-ID_upper"].unique())
display_label_map = {pid: f"Ptn{i+1:02d}" for i, pid in enumerate(sorted_pids)}

# Setup parameters & color assignment
color_stn_sn_green = "#8FBC8F"
n_bootstraps = 10000
rng = np.random.default_rng(seed=42)

def fisher_ci(rho, n, alpha=0.05):
    if abs(rho) >= 1.0:
        return rho, rho
    z = np.arctanh(rho)
    se_z = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_low, z_high = z - z_crit * se_z, z + z_crit * se_z
    return np.tanh(z_low), np.tanh(z_high)

axes_info = [
    ("X_mm", "X Coordinate (mm) (Medial → Lateral)", "Lateral (X)"),
    ("Y_mm", "Y Coordinate (mm) (Posterior → Anterior)", "Anterior-Posterior (Y)"),
    ("Z_mm", "Z Coordinate (mm) (Ventral → Dorsal)", "Dorsal-Ventral (Z)")
]

all_axis_results = {}
lopo_tables = {}

# =====================================================================
# 2. RUN STATISTICAL ANALYSIS FOR X, Y, AND Z AXES
# =====================================================================
for col_name, axis_label, short_name in axes_info:
    x_data = df_analysis[col_name].values
    
    # Observed Correlation & Analytic p (Full Cohort)
    obs_rho, obs_p_analytic = spearmanr(x_data, y_data)
    
    # Exact Permutation Test (8! = 40,320 exact reassignments, Full Cohort)
    all_y_perms = itertools.permutations(y_data)
    count_extreme = 0
    total_perms = 0
    obs_abs_rho = np.abs(obs_rho)

    for y_perm in all_y_perms:
        perm_rho, _ = spearmanr(x_data, y_perm)
        if np.abs(perm_rho) >= obs_abs_rho - 1e-12:
            count_extreme += 1
        total_perms += 1

    perm_p = count_extreme / total_perms
    
    # Fisher z-transform CI (Full Cohort)
    fisher_ci_low, fisher_ci_high = fisher_ci(obs_rho, n_cohort)
    
    # Bootstrap 95% CI (10k iterations, Full Cohort)
    boot_rhos = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        boot_idx = rng.choice(n_cohort, size=n_cohort, replace=True)
        x_boot = x_data[boot_idx]
        y_boot = y_data[boot_idx]
        if len(np.unique(x_boot)) < 2 or len(np.unique(y_boot)) < 2:
            boot_rhos[i] = np.nan
        else:
            r, _ = spearmanr(x_boot, y_boot)
            boot_rhos[i] = r
    boot_rhos = boot_rhos[~np.isnan(boot_rhos)]
    boot_ci_low, boot_ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    
    # Leave-One-Patient-Out (LOPO) Loop
    lopo_list = []
    n_lopo = n_cohort - 1  # Subgroup size = 7
    
    for i in range(n_cohort):
        excluded_pid = patient_ids[i]
        x_sub = np.delete(x_data, i)
        y_sub = np.delete(y_data, i)
        
        r_sub, p_sub = spearmanr(x_sub, y_sub)
        
        # 1. LOPO Fisher z-transform CI (n=7)
        f_low_sub, f_high_sub = fisher_ci(r_sub, n_lopo)
        
        # 2. LOPO Bootstrap 95% CI (10k resamples, n=7)
        boot_rhos_sub = np.zeros(n_bootstraps)
        for b in range(n_bootstraps):
            b_idx = rng.choice(n_lopo, size=n_lopo, replace=True)
            x_b = x_sub[b_idx]
            y_b = y_sub[b_idx]
            if len(np.unique(x_b)) < 2 or len(np.unique(y_b)) < 2:
                boot_rhos_sub[b] = np.nan
            else:
                r_b, _ = spearmanr(x_b, y_b)
                boot_rhos_sub[b] = r_b
        boot_rhos_sub = boot_rhos_sub[~np.isnan(boot_rhos_sub)]
        b_low_sub, b_high_sub = np.percentile(boot_rhos_sub, [2.5, 97.5])
        
        lopo_list.append({
            "Excluded_Patient": excluded_pid,
            "Display_ID": display_label_map.get(excluded_pid, excluded_pid),
            "Rho": r_sub,
            "p_val": p_sub,
            "Boot_CI_Low": b_low_sub,
            "Boot_CI_High": b_high_sub,
            "Fisher_CI_Low": f_low_sub,
            "Fisher_CI_High": f_high_sub,
            "Delta_Rho": r_sub - obs_rho
        })
    df_lopo = pd.DataFrame(lopo_list)
    
    all_axis_results[short_name] = {
        "obs_rho": obs_rho,
        "obs_p_analytic": obs_p_analytic,
        "perm_p": perm_p,
        "fisher_ci": (fisher_ci_low, fisher_ci_high),
        "boot_ci": (boot_ci_low, boot_ci_high),
        "lopo_range": (df_lopo["Rho"].min(), df_lopo["Rho"].max()),
        "x_data": x_data,
        "col_name": col_name,
        "axis_label": axis_label
    }
    lopo_tables[short_name] = df_lopo

# =====================================================================
# 2B. MULTIPLICITY CORRECTIONS (BONFERRONI & HOLM-BONFERRONI ACROSS 3 AXES)
# =====================================================================
m_tests = len(axes_info)  # 3 tests (X, Y, Z)

# 1. Standard Bonferroni: multiply p by m (capped at 1.0)
for short_name in all_axis_results:
    p_raw = all_axis_results[short_name]["perm_p"]
    all_axis_results[short_name]["p_bonferroni"] = min(1.0, p_raw * m_tests)

# 2. Holm-Bonferroni step-down procedure
sorted_axes = sorted(all_axis_results.keys(), key=lambda k: all_axis_results[k]["perm_p"])
prev_p_holm = 0.0

for rank_idx, short_name in enumerate(sorted_axes):
    p_raw = all_axis_results[short_name]["perm_p"]
    multiplier = m_tests - rank_idx  # 3 for 1st, 2 for 2nd, 1 for 3rd
    p_candidate = p_raw * multiplier
    
    # Enforce monotonicity: Holm p-value cannot decrease down the step-down chain
    p_holm = max(p_candidate, prev_p_holm)
    p_holm = min(1.0, p_holm)
    prev_p_holm = p_holm
    
    all_axis_results[short_name]["p_holm"] = p_holm

# =====================================================================
# 3. PRINT FORMATTED TABLES (WITH MULTIPLICITY SUMMARY)
# =====================================================================
print("\n" + "="*85)
print("              SPATIAL COORDINATE MULTIPLICITY CORRECTION SUMMARY (m=3)              ")
print("="*85)
summary_rows = []
for short_name, res in all_axis_results.items():
    summary_rows.append({
        "Axis": short_name,
        "Spearman Rho": f"{res['obs_rho']:.4f}",
        "Unadjusted p_perm": f"{res['perm_p']:.4f}",
        "Bonferroni p": f"{res['p_bonferroni']:.4f}",
        "Holm-Bonferroni p": f"{res['p_holm']:.4f}"
    })
print(pd.DataFrame(summary_rows).to_string(index=False))
print("="*85)

for short_name, res in all_axis_results.items():
    print("\n" + "="*85)
    print(f"     {short_name.upper()} COORDINATE vs. ABSOLUTE ΔFoG-AC ANALYSIS (N=8)     ")
    print("="*85)
    print(f"Observed Spearman correlation (rho) : {res['obs_rho']:.4f}")
    print(f"Analytic p-value (t-dist)           : {res['obs_p_analytic']:.4f}")
    print(f"Exact Permutation p-value (8! = 40k): {res['perm_p']:.4f}")
    print(f"Bonferroni Adjusted p-value (m=3)   : {res['p_bonferroni']:.4f}")
    print(f"Holm-Bonferroni Adjusted p-value    : {res['p_holm']:.4f}")
    print(f"Fisher z-transform 95% CI            : [{res['fisher_ci'][0]:.4f}, {res['fisher_ci'][1]:.4f}]")
    print(f"Bootstrap 95% CI (10k resamples)     : [{res['boot_ci'][0]:.4f}, {res['boot_ci'][1]:.4f}]")
    print(f"LOPO Rho Range                       : [{res['lopo_range'][0]:.4f}, {res['lopo_range'][1]:.4f}] (Δ = {res['lopo_range'][1] - res['lopo_range'][0]:.4f})")
    print("-" * 85)
    print(f"LEAVE-ONE-PATIENT-OUT (LOPO) ESTIMATES ({short_name}):")
    print(lopo_tables[short_name][["Display_ID", "Rho", "p_val", "Boot_CI_Low", "Boot_CI_High", "Fisher_CI_Low", "Fisher_CI_High", "Delta_Rho"]].to_string(index=False))
    print("="*85)

# =====================================================================
# 4. FIGURE 1: 3-PANEL CORRELATION SCATTER PLOTS (X, Y, Z)
# =====================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6.5))

for idx, (short_name, res) in enumerate(all_axis_results.items()):
    ax = axes1[idx]
    x_vals = res["x_data"]
    
    # Horizontal threshold lines at +3 and -3
    ax.axhline(3, color="red", linestyle="--", linewidth=0.5, zorder=1, label="Threshold (±3)")
    ax.axhline(-3, color="red", linestyle="--", linewidth=0.5, zorder=1)
    
    # Scatter points with custom patient ID annotations
    ax.scatter(x_vals, y_data, color=color_stn_sn_green, s=120, alpha=0.9, zorder=3)
    for pid, x_v, y_v in zip(patient_ids, x_vals, y_data):
        display_label = display_label_map.get(pid, pid)
        ax.annotate(display_label, (x_v, y_v), textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight="bold", fontfamily="Arial")
    
    m_ts, b_ts, _, _ = theilslopes(y_data, x_vals)
    x_grid = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_grid, m_ts * x_grid + b_ts, color=color_stn_sn_green, linestyle="--", linewidth=2.0, zorder=2)

    ax.set_title(f"{short_name} vs. Absolute ΔFoG-AC", fontweight="bold", fontsize=14, pad=38, fontfamily="Arial")
    ax.set_xlabel(res["axis_label"], fontweight="bold", fontsize=15, labelpad=10, fontfamily="Arial")
    ax.set_ylabel("Δ FoG-AC (STNmono - STN+SNr-DBS)", fontweight="bold", fontsize=15, labelpad=10, fontfamily="Arial")

    stats_subtitle = (
        f"$\\rho = {res['obs_rho']:.2f}$  |  "
        f"$p_{{\\mathrm{{perm}}}} = {res['perm_p']:.4f}$  |  "
        f"$p_{{\\mathrm{{Holm}}}} = {res['p_holm']:.4f}$\n"
        f"Boot 95% CI: [{res['boot_ci'][0]:.2f}, {res['boot_ci'][1]:.2f}]"
    )
    ax.text(
        0.5, 1.02, stats_subtitle, transform=ax.transAxes,
        ha="center", va="bottom", fontsize=11, fontweight="bold", fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="whitesmoke", alpha=0.9)
    )    

    # Format ticks: Enlarge and bold font explicitly
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.setp(ax.get_xticklabels(), fontweight="bold", fontfamily="Arial")
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontfamily="Arial")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4, color="#C0C0C0", zorder=0)
    
plt.tight_layout()
plt.show()

# =====================================================================
# 5. FIGURE 2: ENHANCED LOPO FOREST PLOTS WITH BOTTOM TABLE PANEL (X, Y, Z)
# =====================================================================
fig2 = plt.figure(figsize=(18, 10.5))
gs = GridSpec(2, 3, height_ratios=[1.7, 1.0], hspace=0.38, wspace=0.25)

y_ticks = np.arange(n_cohort)

for idx, (short_name, res) in enumerate(all_axis_results.items()):
    # -----------------------------------------------------------------
    # Upper panel: Forest Plot
    # -----------------------------------------------------------------
    ax_plot = fig2.add_subplot(gs[0, idx])
    df_l = lopo_tables[short_name]
    obs_r = res["obs_rho"]
    min_r, max_r = res["lopo_range"]
    
    # Highlight full LOPO range with a soft shaded band
    ax_plot.axvspan(min_r, max_r, color=color_stn_sn_green, alpha=0.25)
    
    # Baseline Full Cohort Line
    ax_plot.axvline(obs_r, color="#404040", linestyle="--", linewidth=2)
    
    # Scatter points for individual exclusions
    ax_plot.plot(df_l["Rho"], y_ticks, "o", color=color_stn_sn_green, markersize=9, zorder=3)
    
    # Annotate exact rho value next to each point
    for y_pos, rho_val in zip(y_ticks, df_l["Rho"]):
        ax_plot.annotate(f"{rho_val:.2f}", (rho_val, y_pos), textcoords="offset points", 
                        xytext=(7, -3), fontsize=10, fontweight="bold", fontfamily="Arial")
    
    ax_plot.set_yticks(y_ticks)
    if idx == 0:
        lopo_labels = [f"Excl. {pid}" for pid in df_l["Display_ID"]]
        ax_plot.set_yticklabels(lopo_labels, fontweight="bold", fontsize=15, fontfamily="Arial")
    else:
        ax_plot.set_yticklabels([])
    
    ax_plot.set_title(f"LOPO Influence: {short_name}", fontweight="bold", fontsize=14, pad=38, fontfamily="Arial")
    ax_plot.set_xlabel("Spearman $\\rho$ (Recalculated)", fontweight="bold", fontsize=15, labelpad=10, fontfamily="Arial")
    ax_plot.set_xlim(-1.0, 1.1)
    
    # Subtitle with stats box above plot
    lopo_subtitle = (
        f"Full Cohort $\\rho = {obs_r:.2f}$\n"
        f"LOPO Range: [{min_r:.2f}, {max_r:.2f}]  ($\\Delta = {max_r - min_r:.2f}$)"
    )
    ax_plot.text(
        0.5, 1.02, lopo_subtitle, transform=ax_plot.transAxes,
        ha="center", va="bottom", fontsize=11, fontweight="bold", fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="whitesmoke", alpha=0.9)
    )

    # Format ticks: Enlarge and bold font explicitly
    ax_plot.tick_params(axis="both", which="major", labelsize=12)
    plt.setp(ax_plot.get_xticklabels(), fontweight="bold", fontfamily="Arial")
    plt.setp(ax_plot.get_yticklabels(), fontweight="bold", fontfamily="Arial")

    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.grid(True, linestyle=":", linewidth=0.6, alpha=0.4, color="#C0C0C0", zorder=0)

    # -----------------------------------------------------------------
    # Lower panel: Detailed Summary Table (Includes Boot & Fisher CIs)
    # -----------------------------------------------------------------
    ax_tbl = fig2.add_subplot(gs[1, idx])
    ax_tbl.axis("off")
    
    table_data = []
    for _, row in df_l.iterrows():
        table_data.append([
            f"Excl. {row['Display_ID']}",
            f"{row['Rho']:.3f}",
            f"{row['p_val']:.3f}",
            f"[{row['Boot_CI_Low']:.2f}, {row['Boot_CI_High']:.2f}]",
            f"[{row['Fisher_CI_Low']:.2f}, {row['Fisher_CI_High']:.2f}]",
            f"{row['Delta_Rho']:+.3f}"
        ])
    
    col_labels = ["Excluded", "Spearman $\\rho$", "$p$-value", "Boot 95% CI", "Fisher 95% CI", "$\\Delta\\rho$"]
    
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.20, 1.75)
    
    # Styling table cells
    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(fontfamily="Arial")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#404040")
        else:
            if r % 2 == 0:
                cell.set_facecolor("#F9F9F9")
            else:
                cell.set_facecolor("white")

plt.show()


#%% VTA/E-Field Atlas Intersection vs. ΔFoG-AC Analysis (Complete Suite + LOPO Analysis)

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, norm
from scipy.stats.mstats import theilslopes

# Set global font to Arial
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'

# =====================================================================
# 1. CONFIGURATION & COHORT SELECTION (EXPLICIT N=8 TARGET PATIENTS)
# =====================================================================
data_dir = r"D:\combined_nigrasteer_Nov2023\combined_nigrasteer_SN"

# EXACT N=8 PATIENT LIST FROM YOUR TARGET COHORT
target_patients = ["ID07", "ID08", "ID10", "PD01", "PD05", "PD07", "PD09", "PD10"]

# Prepare df_clinscore
df_clinscore_clean = df_clinscore.copy()
df_clinscore_clean["Patient-ID_upper"] = df_clinscore_clean["Patient-ID"].astype(str).str.upper()

# Filter df_clinscore explicitly for target_patients
df_clin_cohort = (
    df_clinscore_clean[df_clinscore_clean["Patient-ID_upper"].isin([p.upper() for p in target_patients])]
    [["Patient-ID_upper", "STN_vs_STN+SNr"]]
    .drop_duplicates(subset=["Patient-ID_upper"])
    .set_index("Patient-ID_upper")
    .reindex([p.upper() for p in target_patients])  # Preserve exact order of target_patients
    .reset_index()
)

# Build df_analysis with exactly N=8 rows
df_analysis = pd.DataFrame({
    "Patient-ID": df_clin_cohort["Patient-ID_upper"],
    "Delta_FoG_AC": df_clin_cohort["STN_vs_STN+SNr"].astype(float)
})

# Display mapping (Ptn01..Ptn08)
sorted_pids = sorted(df_analysis["Patient-ID"].unique())
display_label_map = {pid: f"Ptn{i+1:02d}" for i, pid in enumerate(sorted_pids)}

# Setup parameters & color assignment
color_stn_sn_green = "#8FBC8F"  # Muted Dark Sea Green
n_bootstraps = 10000
rng = np.random.default_rng(seed=42)

def fisher_ci(rho, n, alpha=0.05):
    if abs(rho) >= 1.0:
        return rho, rho
    z = np.arctanh(rho)
    se_z = 1.0 / np.sqrt(n - 3) if n > 3 else 0
    z_crit = norm.ppf(1 - alpha / 2)
    z_low, z_high = z - z_crit * se_z, z + z_crit * se_z
    return np.tanh(z_low), np.tanh(z_high)

# File mappings for MAT extraction
vat_files = {
    "left_snc_8ptns.mat": "Left_SNc_VAT", "left_snr_8ptns.mat": "Left_SNr_VAT",
    "right_snc_8ptns.mat": "Right_SNc_VAT", "right_snr_8ptns.mat": "Right_SNr_VAT"
}

efield_files = {
    "left_snc_efield_8ptns.mat": "Left_SNc_Efield", "left_snr_efield_8ptns.mat": "Left_SNr_Efield",
    "right_snc_efield_8ptns.mat": "Right_SNc_Efield", "right_snr_efield_8ptns.mat": "Right_SNr_Efield"
}

# =====================================================================
# 2. DATA EXTRACTION AND METRIC DERIVATION
# =====================================================================
for file_name, label in vat_files.items():
    path = os.path.join(data_dir, file_name)
    if os.path.exists(path):
        mat = sio.loadmat(path)
        struct = mat['stats'][0, 0][1]
        df_analysis[f"{label}_Raw"] = [float(v) for v in struct['both'][0, 0].flatten()]
        df_analysis[f"{label}_Norm"] = [float(v) for v in struct['nboth'][0, 0].flatten()]

for file_name, label in efield_files.items():
    path = os.path.join(data_dir, file_name)
    if os.path.exists(path):
        mat = sio.loadmat(path)
        struct = mat['stats'][0, 0][1]
        df_analysis[f"{label}_Raw"] = [float(v) for v in struct['both'][0, 0].flatten()]
        df_analysis[f"{label}_Norm"] = [float(v) for v in struct['nboth'][0, 0].flatten()]

# ---------------------------------------------------------------------
# A. BILATERAL SUBREGION COMBINATIONS
# ---------------------------------------------------------------------
df_analysis["Bilateral_SNc_VAT_Raw"] = df_analysis["Left_SNc_VAT_Raw"] + df_analysis["Right_SNc_VAT_Raw"]
df_analysis["Bilateral_SNc_VAT_Norm"] = df_analysis["Left_SNc_VAT_Norm"] + df_analysis["Right_SNc_VAT_Norm"]
df_analysis["Bilateral_SNr_VAT_Raw"] = df_analysis["Left_SNr_VAT_Raw"] + df_analysis["Right_SNr_VAT_Raw"]
df_analysis["Bilateral_SNr_VAT_Norm"] = df_analysis["Left_SNr_VAT_Norm"] + df_analysis["Right_SNr_VAT_Norm"]

df_analysis["Bilateral_SNc_Efield_Raw"] = df_analysis["Left_SNc_Efield_Raw"] + df_analysis["Right_SNc_Efield_Raw"]
df_analysis["Bilateral_SNc_Efield_Norm"] = df_analysis["Left_SNc_Efield_Norm"] + df_analysis["Right_SNc_Efield_Norm"]
df_analysis["Bilateral_SNr_Efield_Raw"] = df_analysis["Left_SNr_Efield_Raw"] + df_analysis["Right_SNr_Efield_Raw"]
df_analysis["Bilateral_SNr_Efield_Norm"] = df_analysis["Left_SNr_Efield_Norm"] + df_analysis["Right_SNr_Efield_Norm"]

# ---------------------------------------------------------------------
# B. COMBINED WHOLE SUBSTANTIA NIGRA (SN = SNc + SNr) COMBINATIONS
# ---------------------------------------------------------------------
# Left Whole SN
df_analysis["Left_SN_Total_VAT_Raw"] = df_analysis["Left_SNc_VAT_Raw"] + df_analysis["Left_SNr_VAT_Raw"]
df_analysis["Left_SN_Total_VAT_Norm"] = df_analysis["Left_SNc_VAT_Norm"] + df_analysis["Left_SNr_VAT_Norm"]
df_analysis["Left_SN_Total_Efield_Raw"] = df_analysis["Left_SNc_Efield_Raw"] + df_analysis["Left_SNr_Efield_Raw"]
df_analysis["Left_SN_Total_Efield_Norm"] = df_analysis["Left_SNc_Efield_Norm"] + df_analysis["Left_SNr_Efield_Norm"]

# Right Whole SN
df_analysis["Right_SN_Total_VAT_Raw"] = df_analysis["Right_SNc_VAT_Raw"] + df_analysis["Right_SNr_VAT_Raw"]
df_analysis["Right_SN_Total_VAT_Norm"] = df_analysis["Right_SNc_VAT_Norm"] + df_analysis["Right_SNr_VAT_Norm"]
df_analysis["Right_SN_Total_Efield_Raw"] = df_analysis["Right_SNc_Efield_Raw"] + df_analysis["Right_SNr_Efield_Raw"]
df_analysis["Right_SN_Total_Efield_Norm"] = df_analysis["Right_SNc_Efield_Norm"] + df_analysis["Right_SNr_Efield_Norm"]

# Bilateral Whole SN
df_analysis["Bilateral_SN_Total_VAT_Raw"] = df_analysis["Bilateral_SNc_VAT_Raw"] + df_analysis["Bilateral_SNr_VAT_Raw"]
df_analysis["Bilateral_SN_Total_VAT_Norm"] = df_analysis["Bilateral_SNc_VAT_Norm"] + df_analysis["Bilateral_SNr_VAT_Norm"]
df_analysis["Bilateral_SN_Total_Efield_Raw"] = df_analysis["Bilateral_SNc_Efield_Raw"] + df_analysis["Bilateral_SNr_Efield_Raw"]
df_analysis["Bilateral_SN_Total_Efield_Norm"] = df_analysis["Bilateral_SNc_Efield_Norm"] + df_analysis["Bilateral_SNr_Efield_Norm"]

vta_execution_stats = {}

# =====================================================================
# 3. PLOTTING & LOPO COMPUTATION ENGINE
# =====================================================================
def run_and_plot_lopo_suite(metrics_list, figure_label_id, main_title=""):
    n_panels = len(metrics_list)
    y_data = df_analysis["Delta_FoG_AC"].values
    patient_ids = df_analysis["Patient-ID"].values
    n_cohort = len(df_analysis)
    
    # -----------------------------------------------------------------
    # FIGURE A: SCATTER PLOTS WITH THEIL-SEN REGRESSION
    # -----------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 6.5))
    if n_panels == 1:
        axes1 = [axes1]
        
    vta_execution_stats[figure_label_id] = {}
    lopo_results_dict = {}

    for idx, (col_name, display_label) in enumerate(metrics_list):
        ax = axes1[idx]
        x_data = df_analysis[col_name].values
        
        # Color coding for structural trendlines
        if "SNc" in col_name:
            color_line = "#8C6BB1"  # Muted Amethyst Purple for SNc
        elif "SNr" in col_name:
            color_line = color_stn_sn_green  # Muted Dark Sea Green for SNr
        else:
            color_line = "#4682B4"  # Steel Blue for Whole SN
            
        # Theil-Sen Robust Regression
        m_ts, b_ts, _, _ = theilslopes(y_data, x_data)
        
        x_min_bounded = x_data.min()
        x_max_bounded = x_data.max() + (x_data.std() * 0.1) if x_data.std() > 0 else x_data.max() + 0.1
        x_grid = np.linspace(x_min_bounded, x_max_bounded, 500)
        
        ax.plot(x_grid, m_ts * x_grid + b_ts, color=color_line, linestyle="--", linewidth=2, zorder=2)
        
        ax.scatter(x_data, y_data, color=color_stn_sn_green, s=120, edgecolors="none", alpha=0.9, zorder=3)
        
        for pid, x_v, y_v in zip(patient_ids, x_data, y_data):
            disp_label = display_label_map.get(pid, pid)
            ax.annotate(disp_label, (x_v, y_v), textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight="bold")
            
        # Full Cohort Spearman
        obs_rho, obs_p = spearmanr(x_data, y_data)
        
        # LOPO Resampling Routine
        lopo_list = []
        n_lopo = n_cohort - 1
        for i in range(n_cohort):
            excluded_pid = patient_ids[i]
            x_sub = np.delete(x_data, i)
            y_sub = np.delete(y_data, i)
            
            r_sub, p_sub = spearmanr(x_sub, y_sub)
            f_low_sub, f_high_sub = fisher_ci(r_sub, n_lopo)
            
            # Bootstrap 10k resamples for sub-model
            boot_rhos_sub = np.zeros(n_bootstraps)
            for b in range(n_bootstraps):
                b_idx = rng.choice(n_lopo, size=n_lopo, replace=True)
                x_b, y_b = x_sub[b_idx], y_sub[b_idx]
                if len(np.unique(x_b)) < 2 or len(np.unique(y_b)) < 2:
                    boot_rhos_sub[b] = np.nan
                else:
                    boot_rhos_sub[b], _ = spearmanr(x_b, y_b)
            boot_rhos_sub = boot_rhos_sub[~np.isnan(boot_rhos_sub)]
            b_low_sub, b_high_sub = np.percentile(boot_rhos_sub, [2.5, 97.5])
            
            lopo_list.append({
                "Excluded_Patient": excluded_pid,
                "Display_ID": display_label_map.get(excluded_pid, excluded_pid),
                "Rho": r_sub,
                "p_val": p_sub,
                "Boot_CI_Low": b_low_sub,
                "Boot_CI_High": b_high_sub,
                "Fisher_CI_Low": f_low_sub,
                "Fisher_CI_High": f_high_sub,
                "Delta_Rho": r_sub - obs_rho
            })
            
        df_lopo = pd.DataFrame(lopo_list)
        lopo_results_dict[col_name] = (df_lopo, obs_rho, display_label)
        
        vta_execution_stats[figure_label_id][col_name] = {
            "n_pts": n_cohort, "rho": obs_rho, "p_spearman": obs_p,
            "lopo_min": df_lopo["Rho"].min(), "lopo_max": df_lopo["Rho"].max()
        }
        
        stats_subtitle = f"Full Cohort Spearman: $\\rho = {obs_rho:.2f}$ ($p = {obs_p:.3f}$)"
        ax.set_title(f"{display_label.split(' (')[0]} Target Distribution", fontweight="bold", fontsize=12, pad=38)
        
        ax.text(
            0.5, 1.03, stats_subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11, color="#404040", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="whitesmoke", alpha=0.9)
        )
        
        ax.set_xlabel(display_label, fontweight="bold", fontsize=11, labelpad=8)
        ax.set_ylabel("Δ FoG-AC (STNmono - STN+SNr-DBS)", fontweight="bold", fontsize=11)
        
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4, color="#C0C0C0", zorder=0)

    if main_title:
        fig1.suptitle(main_title, fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # FIGURE B: GRIDSPEC LOPO FOREST PLOT + SUMMARY TABLES
    # -----------------------------------------------------------------
    fig2 = plt.figure(figsize=(7.5 * n_panels, 10.5))
    gs = GridSpec(2, n_panels, height_ratios=[1.7, 1.0], hspace=0.38, wspace=0.25)
    y_ticks = np.arange(n_cohort)

    for idx, (col_name, display_label) in enumerate(metrics_list):
        df_l, obs_r, disp_lbl = lopo_results_dict[col_name]
        min_r, max_r = df_l["Rho"].min(), df_l["Rho"].max()
        
        # Forest Plot Upper Panel
        ax_plot = fig2.add_subplot(gs[0, idx])
        ax_plot.axvspan(min_r, max_r, color=color_stn_sn_green, alpha=0.25)
        ax_plot.axvline(obs_r, color="#404040", linestyle="--", linewidth=2)
        ax_plot.plot(df_l["Rho"], y_ticks, "o", color=color_stn_sn_green, markersize=9, zorder=3)
        
        for y_pos, rho_val in zip(y_ticks, df_l["Rho"]):
            ax_plot.annotate(f"{rho_val:.2f}", (rho_val, y_pos), textcoords="offset points", xytext=(7, -3), fontsize=10, fontweight="bold")
            
        ax_plot.set_yticks(y_ticks)
        if idx == 0:
            lopo_labels = [f"Excl. {pid}" for pid in df_l["Display_ID"]]
            ax_plot.set_yticklabels(lopo_labels, fontweight="bold", fontsize=12)
        else:
            ax_plot.set_yticklabels([])
            
        ax_plot.set_title(f"LOPO Influence: {disp_lbl.split(' (')[0]}", fontweight="bold", fontsize=12, pad=38)
        ax_plot.set_xlabel("Spearman $\\rho$ (Recalculated)", fontweight="bold", fontsize=12, labelpad=10)
        ax_plot.set_xlim(-1.0, 1.1)
        
        lopo_subtitle = f"Full Cohort $\\rho = {obs_r:.2f}$\nLOPO Range: [{min_r:.2f}, {max_r:.2f}]  ($\\Delta = {max_r - min_r:.2f}$)"
        ax_plot.text(0.5, 1.02, lopo_subtitle, transform=ax_plot.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="whitesmoke", alpha=0.9))
        
        ax_plot.tick_params(axis="both", which="major", labelsize=11)
        plt.setp(ax_plot.get_xticklabels(), fontweight="bold")
        ax_plot.spines["top"].set_visible(False)
        ax_plot.spines["right"].set_visible(False)
        ax_plot.grid(True, linestyle=":", linewidth=0.6, alpha=0.4, color="#C0C0C0", zorder=0)

        # Detailed Summary Table Lower Panel
        ax_tbl = fig2.add_subplot(gs[1, idx])
        ax_tbl.axis("off")
        
        table_data = []
        for _, row in df_l.iterrows():
            table_data.append([
                f"Excl. {row['Display_ID']}",
                f"{row['Rho']:.3f}",
                f"{row['p_val']:.3f}",
                f"[{row['Boot_CI_Low']:.2f}, {row['Boot_CI_High']:.2f}]",
                f"[{row['Fisher_CI_Low']:.2f}, {row['Fisher_CI_High']:.2f}]",
                f"{row['Delta_Rho']:+.3f}"
            ])
            
        col_labels = ["Excluded", "Spearman $\\rho$", "$p$-value", "Boot 95% CI", "Fisher 95% CI", "$\\Delta\\rho$"]
        tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.15, 1.65)
        
        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(fontfamily="Arial")
            if r == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#404040")
            else:
                cell.set_facecolor("#F9F9F9" if r % 2 == 0 else "white")

    plt.show()

# =====================================================================
# 4. EXECUTE SUITE WITH LOPO ANALYSIS
# =====================================================================
plt.close('all')

# ---------------------------------------------------------------------
# SECTION I: BILATERAL SUBREGION ANALYSES (SNc vs SNr)
# ---------------------------------------------------------------------
print("--> Executing Section I: Bilateral Subregion Models...")

run_and_plot_lopo_suite([
    ("Bilateral_SNc_VAT_Raw", "Bilateral SNc VAT ($mm^3$)"),
    ("Bilateral_SNr_VAT_Raw", "Bilateral SNr VAT ($mm^3$)")
], "Fig1_Bilateral_VAT_Raw", "Bilateral VAT Raw Volumes vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Bilateral_SNc_VAT_Norm", "Bilateral SNc VAT (Normalized)"),
    ("Bilateral_SNr_VAT_Norm", "Bilateral SNr VAT (Normalized)")
], "Fig2_Bilateral_VAT_Norm", "Bilateral VAT Normalized Intersections vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Bilateral_SNc_Efield_Raw", "Bilateral SNc E-Field Intensity (Raw)"),
    ("Bilateral_SNr_Efield_Raw", "Bilateral SNr E-Field Intensity (Raw)")
], "Fig3_Bilateral_Efield_Raw", "Bilateral E-Field Raw Intensities vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Bilateral_SNc_Efield_Norm", "Bilateral SNc E-Field Intensity (Normalized)"),
    ("Bilateral_SNr_Efield_Norm", "Bilateral SNr E-Field Intensity (Normalized)")
], "Fig4_Bilateral_Efield_Norm", "Bilateral E-Field Normalized Intensities vs. ΔFoG-AC")

# ---------------------------------------------------------------------
# SECTION II: HEMISPHERIC ISOLATED ANALYSES (RAW & NORMALIZED)
# ---------------------------------------------------------------------
print("--> Executing Section II: Hemispheric Models (Left vs Right - Raw & Normalized)...")

run_and_plot_lopo_suite([
    ("Left_SNc_VAT_Raw", "Left SNc VAT ($mm^3$)"),
    ("Left_SNr_VAT_Raw", "Left SNr VAT ($mm^3$)")
], "Fig5_Left_VAT_Raw", "Left Hemisphere VAT Raw Volumes vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Left_SNc_Efield_Raw", "Left SNc E-Field Intensity (Raw)"),
    ("Left_SNr_Efield_Raw", "Left SNr E-Field Intensity (Raw)")
], "Fig6_Left_Efield_Raw", "Left Hemisphere E-Field Raw Intensities vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Left_SNc_VAT_Norm", "Left SNc VAT (Normalized)"),
    ("Left_SNr_VAT_Norm", "Left SNr VAT (Normalized)")
], "Fig7_Left_VAT_Norm", "Left Hemisphere VAT Normalized Intersections vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Left_SNc_Efield_Norm", "Left SNc E-Field Intensity (Normalized)"),
    ("Left_SNr_Efield_Norm", "Left SNr E-Field Intensity (Normalized)")
], "Fig8_Left_Efield_Norm", "Left Hemisphere E-Field Normalized Intensities vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Right_SNc_VAT_Raw", "Right SNc VAT ($mm^3$)"),
    ("Right_SNr_VAT_Raw", "Right SNr VAT ($mm^3$)")
], "Fig9_Right_VAT_Raw", "Right Hemisphere VAT Raw Volumes vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Right_SNc_Efield_Raw", "Right SNc E-Field Intensity (Raw)"),
    ("Right_SNr_Efield_Raw", "Right SNr E-Field Intensity (Raw)")
], "Fig10_Right_Efield_Raw", "Right Hemisphere E-Field Raw Intensities vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Right_SNc_VAT_Norm", "Right SNc VAT (Normalized)"),
    ("Right_SNr_VAT_Norm", "Right SNr VAT (Normalized)")
], "Fig11_Right_VAT_Norm", "Right Hemisphere VAT Normalized Intersections vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Right_SNc_Efield_Norm", "Right SNc E-Field Intensity (Normalized)"),
    ("Right_SNr_Efield_Norm", "Right SNr E-Field Intensity (Normalized)")
], "Fig12_Right_Efield_Norm", "Right Hemisphere E-Field Normalized Intensities vs. ΔFoG-AC")

# ---------------------------------------------------------------------
# SECTION III: COMBINED WHOLE SUBSTANTIA NIGRA (SNc + SNr = SN)
# ---------------------------------------------------------------------
print("--> Executing Section III: Combined Whole Substantia Nigra (SN) Models...")

run_and_plot_lopo_suite([
    ("Left_SN_Total_VAT_Raw", "Left Whole SN VAT ($mm^3$)"),
    ("Right_SN_Total_VAT_Raw", "Right Whole SN VAT ($mm^3$)"),
    ("Bilateral_SN_Total_VAT_Raw", "Bilateral Whole SN VAT ($mm^3$)")
], "Fig13_Whole_SN_VAT_Raw", "Whole Substantia Nigra (SNc+SNr) VAT Raw Volumes vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Left_SN_Total_VAT_Norm", "Left Whole SN VAT (Normalized)"),
    ("Right_SN_Total_VAT_Norm", "Right Whole SN VAT (Normalized)"),
    ("Bilateral_SN_Total_VAT_Norm", "Bilateral Whole SN VAT (Normalized)")
], "Fig14_Whole_SN_VAT_Norm", "Whole Substantia Nigra (SNc+SNr) VAT Normalized Intersections vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Left_SN_Total_Efield_Raw", "Left Whole SN E-Field (Raw)"),
    ("Right_SN_Total_Efield_Raw", "Right Whole SN E-Field (Raw)"),
    ("Bilateral_SN_Total_Efield_Raw", "Bilateral Whole SN E-Field (Raw)")
], "Fig15_Whole_SN_Efield_Raw", "Whole Substantia Nigra (SNc+SNr) E-Field Raw Intensities vs. ΔFoG-AC")

run_and_plot_lopo_suite([
    ("Left_SN_Total_Efield_Norm", "Left Whole SN E-Field (Normalized)"),
    ("Right_SN_Total_Efield_Norm", "Right Whole SN E-Field (Normalized)"),
    ("Bilateral_SN_Total_Efield_Norm", "Bilateral Whole SN E-Field (Normalized)")
], "Fig16_Whole_SN_Efield_Norm", "Whole Substantia Nigra (SNc+SNr) E-Field Normalized Intensities vs. ΔFoG-AC")

# =====================================================================
# TERMINAL AUDIT SUMMARY (SPEARMAN + LOPO SENSITIVITY RANGE)
# =====================================================================
print("\n" + "="*95)
print("          COMPLETE VTA/E-FIELD & LOPO SENSITIVITY VERIFICATION SUMMARY          ")
print("="*95)
print(f"Target Analysis Cohort (N={len(df_analysis)}): {df_analysis['Patient-ID'].tolist()}")

for fig_id, targets in vta_execution_stats.items():
    print(f"\n[{fig_id} MODEL BOUNDARIES]")
    for col_metric, data in targets.items():
        print(f"  --> {col_metric:<30} | Full Cohort \u03c1 = {data['rho']:>6.3f} (p = {data['p_spearman']:>7.5f}) | LOPO Range: [{data['lopo_min']:>6.3f}, {data['lopo_max']:>6.3f}] (\u0394 = {data['lopo_max'] - data['lopo_min']:>5.3f})")
print("="*95 + "\n")

#%% HMAT-SN connectivity: raw clin score change + LOPO + Exact Permutations + Multiplicity Control

import os
import itertools
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, norm, theilslopes

# Set global font to Arial
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. CONFIGURATION AND DATA ALIGNMENT (N=8 Cohort)
# =====================================================================
data_dir = r"D:\combined_nigrasteer_Nov2023\combined_nigrasteer_SN"
structures = ["M1", "S1", "SMA", "preSMA", "PMd", "PMv", "PPN"]

color_stn_sn_green = "#8FBC8F"
n_bootstraps = 10000
rng = np.random.default_rng(seed=42)

# Prepare cohort and clinical score alignment
df_SN_clean = df_SN.copy()
df_clinscore_clean = df_clinscore.copy()

df_SN_clean["Patient-ID_upper"] = df_SN_clean["Patient-ID"].astype(str).str.upper()
df_clinscore_clean["Patient-ID_upper"] = df_clinscore_clean["Patient-ID"].astype(str).str.upper()

target_patients = set(df_SN_clean["Patient-ID_upper"].unique()) - {"PD08"}

df_clin_cohort = df_clinscore_clean[df_clinscore_clean["Patient-ID_upper"].isin(target_patients)].copy()
df_clin_cohort = df_clin_cohort.drop_duplicates(subset=["Patient-ID_upper"]).sort_values("Patient-ID_upper").reset_index(drop=True)

y_data = df_clin_cohort["STN_vs_STN+SNr"].values  # Raw ΔFoG-AC Score
patient_ids = df_clin_cohort["Patient-ID_upper"].values
n_cohort = len(df_clin_cohort)

# Explicit mapping from original patient IDs to Ptn01..Ptn08
sorted_pids = sorted(df_clin_cohort["Patient-ID_upper"].unique())
display_label_map = {pid: f"Ptn{i+1:02d}" for i, pid in enumerate(sorted_pids)}

def fisher_ci(rho, n, alpha=0.05):
    if abs(rho) >= 1.0:
        return rho, rho
    z = np.arctanh(rho)
    se_z = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_low, z_high = z - z_crit * se_z, z + z_crit * se_z
    return np.tanh(z_low), np.tanh(z_high)

def compute_fdr_q_values(p_values):
    """Computes Benjamini-Hochberg FDR q-values for an array of p-values."""
    p_vals = np.asarray(p_values)
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    
    q_vals = np.zeros(m)
    prev_q = 1.0
    for i in range(m - 1, -1, -1):
        q_calc = (sorted_p[i] * m) / (i + 1)
        prev_q = min(prev_q, q_calc)
        q_vals[i] = prev_q
        
    # Restore original order
    original_q_vals = np.zeros(m)
    original_q_vals[sorted_indices] = q_vals
    return np.clip(original_q_vals, 0.0, 1.0)

# =====================================================================
# 2. STANDALONE FIGURE GENERATION PIPELINE WITH MULTIPLICITY CONTROL
# =====================================================================
def run_standalone_connectivity_figures(is_normalized=True):
    model_type = "Normalized" if is_normalized else "Raw"
    print("\n" + "="*95)
    print(f"   GENERATING STANDALONE FIGURES ({model_type.upper()} CONNECTIVITY, N=8)   ")
    print("="*95)
    
    struct_results = {}
    lopo_tables_dict = {}
    
    # -----------------------------------------------------------------
    # STEP A: RUN ALL 7 STRUCTURE CALCULATIONS & EXACT PERMUTATIONS
    # -----------------------------------------------------------------
    for struct in structures:
        left_file = os.path.join(data_dir, f"left_{struct}_8ptns.mat")
        right_file = os.path.join(data_dir, f"right_{struct}_8ptns.mat")
        
        if not (os.path.exists(left_file) and os.path.exists(right_file)):
            continue
            
        mat_L = sio.loadmat(left_file)['stats'][0, 0]['fccorr'][0, 0]
        mat_R = sio.loadmat(right_file)['stats'][0, 0]['fccorr'][0, 0]
        
        if is_normalized:
            x_data = mat_L['nleft'].flatten() + mat_R['nright'].flatten()
        else:
            x_data = mat_L['left'].flatten() + mat_R['right'].flatten()
            
        # --- OBSERVED SPEARMAN CORRELATION & ANALYTIC P-VALUE ---
        obs_rho, obs_p_analytic = spearmanr(x_data, y_data)
        
        # --- EXACT PERMUTATION TEST (8! = 40,320 REASSIGNMENTS) ---
        all_y_perms = itertools.permutations(y_data)
        count_extreme = 0
        total_perms = 0
        obs_abs_rho = np.abs(obs_rho)

        for y_perm in all_y_perms:
            perm_rho, _ = spearmanr(x_data, y_perm)
            if np.abs(perm_rho) >= obs_abs_rho - 1e-12:
                count_extreme += 1
            total_perms += 1

        perm_p = count_extreme / total_perms
        
        # --- CIs ---
        fisher_ci_low, fisher_ci_high = fisher_ci(obs_rho, n_cohort)
        
        # Non-parametric Bootstrap Percentile CI (10k resamples)
        boot_rhos = np.zeros(n_bootstraps)
        for i in range(n_bootstraps):
            boot_idx = rng.choice(n_cohort, size=n_cohort, replace=True)
            x_boot, y_boot = x_data[boot_idx], y_data[boot_idx]
            if len(np.unique(x_boot)) < 2 or len(np.unique(y_boot)) < 2:
                boot_rhos[i] = np.nan
            else:
                r, _ = spearmanr(x_boot, y_boot)
                boot_rhos[i] = r
        boot_rhos = boot_rhos[~np.isnan(boot_rhos)]
        boot_ci_low, boot_ci_high = np.percentile(boot_rhos, [2.5, 97.5])
        
        # --- LOPO ANALYSIS ---
        lopo_list = []
        n_lopo = n_cohort - 1
        for i in range(n_cohort):
            excluded_pid = patient_ids[i]
            x_sub = np.delete(x_data, i)
            y_sub = np.delete(y_data, i)
            r_sub, p_sub = spearmanr(x_sub, y_sub)
            f_low_sub, f_high_sub = fisher_ci(r_sub, n_lopo)
            
            lopo_list.append({
                "Excluded_Patient": excluded_pid,
                "Display_ID": display_label_map.get(excluded_pid, excluded_pid),
                "Rho": r_sub,
                "p_val": p_sub,
                "Fisher_CI_Low": f_low_sub,
                "Fisher_CI_High": f_high_sub,
                "Delta_Rho": r_sub - obs_rho
            })
        df_lopo = pd.DataFrame(lopo_list)
        
        struct_results[struct] = {
            "x_data": x_data,
            "obs_rho": obs_rho,
            "obs_p_analytic": obs_p_analytic,
            "perm_p": perm_p,
            "fisher_ci": (fisher_ci_low, fisher_ci_high),
            "boot_ci": (boot_ci_low, boot_ci_high),
            "min_r": df_lopo["Rho"].min(),
            "max_r": df_lopo["Rho"].max()
        }
        lopo_tables_dict[struct] = df_lopo

    # -----------------------------------------------------------------
    # STEP B: APPLY MULTIPLICITY CORRECTIONS ACROSS HMAT FAMILY (m=7)
    # -----------------------------------------------------------------
    valid_structs = list(struct_results.keys())
    m_family = len(valid_structs)
    raw_p_perms = [struct_results[s]["perm_p"] for s in valid_structs]
    
    # 1. Benjamini-Hochberg FDR Control
    fdr_q_values = compute_fdr_q_values(raw_p_perms)
    
    # 2. Holm-Bonferroni Step-Down
    sorted_indices = np.argsort(raw_p_perms)
    holm_p_values = np.zeros(m_family)
    prev_holm = 0.0
    for rank_idx, idx in enumerate(sorted_indices):
        p_raw = raw_p_perms[idx]
        multiplier = m_family - rank_idx
        p_candidate = p_raw * multiplier
        prev_holm = max(prev_holm, p_candidate)
        holm_p_values[idx] = min(1.0, prev_holm)

    for i, struct in enumerate(valid_structs):
        struct_results[struct]["q_fdr"] = fdr_q_values[i]
        struct_results[struct]["p_holm"] = holm_p_values[i]

    # -----------------------------------------------------------------
    # STEP C: PRINT TERMINAL MULTIPLICITY AUDIT TABLE (WITH P_ANALYTIC)
    # -----------------------------------------------------------------
    print("\n" + "="*95)
    print(f"      HMAT CORTICAL CONNECTIVITY MULTIPLICITY SUMMARY ({model_type.upper()}, m={m_family})      ")
    print("="*95)
    audit_rows = []
    for s in valid_structs:
        r = struct_results[s]
        audit_rows.append({
            "Structure": s,
            "Spearman Rho": f"{r['obs_rho']:.4f}",
            "p_analytic": f"{r['obs_p_analytic']:.4f}",
            "Unadjusted p_perm": f"{r['perm_p']:.4f}",
            "FDR q-value": f"{r['q_fdr']:.4f}",
            "Holm-Bonferroni p": f"{r['p_holm']:.4f}"
        })
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print("="*95)

    # -----------------------------------------------------------------
    # STEP D: GENERATE INDIVIDUAL STANDALONE FIGURES FOR EACH STRUCTURE
    # -----------------------------------------------------------------
    for struct in valid_structs:
        res = struct_results[struct]
        df_lopo = lopo_tables_dict[struct]
        x_data = res["x_data"]
        obs_rho = res["obs_rho"]
        obs_p_analytic = res["obs_p_analytic"]
        perm_p = res["perm_p"]
        q_fdr = res["q_fdr"]
        boot_ci_low, boot_ci_high = res["boot_ci"]
        min_r, max_r = res["min_r"], res["max_r"]

        print("\n" + "-"*85)
        print(f" Structure: {struct} ({model_type} Model)")
        print(f" Rho = {obs_rho:.4f} | p_analytic = {obs_p_analytic:.4f} | Perm p = {perm_p:.4f} | FDR q = {q_fdr:.4f} | Boot 95% CI = [{boot_ci_low:.3f}, {boot_ci_high:.3f}]")
        print(f" LOPO Range = [{min_r:.4f}, {max_r:.4f}] (Delta = {max_r - min_r:.4f})")
        print("-"*85)
        print(df_lopo[["Display_ID", "Rho", "p_val", "Fisher_CI_Low", "Fisher_CI_High", "Delta_Rho"]].to_string(index=False))

        # -------------------------------------------------------------
        # FIGURE 1: STANDALONE SCATTER PLOT WITH THEIL-SEN REGRESSION
        # -------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(6.5, 6.0))
        
        # Horizontal threshold lines
        ax1.axhline(3, color="red", linestyle="--", linewidth=0.5, zorder=1, label="Threshold (±3)")
        ax1.axhline(-3, color="red", linestyle="--", linewidth=0.5, zorder=1)
        
        # Scatter points & Annotations with display IDs (Ptn01..Ptn08)
        ax1.scatter(x_data, y_data, color=color_stn_sn_green, s=130, alpha=0.9, zorder=3)
        for pid, x_v, y_v in zip(patient_ids, x_data, y_data):
            display_label = display_label_map.get(pid, pid)
            ax1.annotate(display_label, (x_v, y_v), textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight="bold", fontfamily="Arial")
        
        # Non-parametric Theil-Sen Robust Trendline
        m_ts, b_ts, _, _ = theilslopes(y_data, x_data)
        x_grid = np.linspace(x_data.min(), x_data.max(), 100)
        ax1.plot(x_grid, m_ts * x_grid + b_ts, color=color_stn_sn_green, linestyle="--", linewidth=2.0, zorder=2)
        
        ax1.set_title(f"{struct} ({model_type}) vs. Absolute ΔFoG-AC", fontweight="bold", fontsize=14, pad=38, fontfamily="Arial")
        ax1.set_xlabel(f"Bilateral Ipsilateral {struct} Connectivity ({model_type})", fontweight="bold", fontsize=12, labelpad=10, fontfamily="Arial")
        ax1.set_ylabel("Δ FoG-AC (STNmono - STN+SNr-DBS)", fontweight="bold", fontsize=13, labelpad=10, fontfamily="Arial")

        stats_subtitle = (
            f"$\\rho = {obs_rho:.2f}$  |  "
            f"$p_{{\\mathrm{{analytic}}}} = {obs_p_analytic:.4f}$  |  "
            f"$p_{{\\mathrm{{perm}}}} = {perm_p:.4f}$  |  "
            f"$q_{{\\mathrm{{FDR}}}} = {q_fdr:.4f}$\n"
            f"Boot 95% CI: [{boot_ci_low:.2f}, {boot_ci_high:.2f}]"
        )
        ax1.text(
            0.5, 1.02, stats_subtitle, transform=ax1.transAxes,
            ha="center", va="bottom", fontsize=10, fontweight="bold", fontfamily="Arial",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="whitesmoke", alpha=0.9)
        )    

        ax1.tick_params(axis="both", which="major", labelsize=11)
        plt.setp(ax1.get_xticklabels(), fontweight="bold", fontfamily="Arial")
        plt.setp(ax1.get_yticklabels(), fontweight="bold", fontfamily="Arial")

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.4, color="#C0C0C0", zorder=0)
        
        plt.tight_layout()
        plt.show()

        # -------------------------------------------------------------
        # FIGURE 2: STANDALONE LOPO FOREST PLOT & TABLE
        # -------------------------------------------------------------
        fig2 = plt.figure(figsize=(7.0, 9.5))
        gs = GridSpec(2, 1, height_ratios=[1.7, 1.0], hspace=0.38)
        y_ticks = np.arange(n_cohort)

        # Upper panel: Forest Plot
        ax_plot = fig2.add_subplot(gs[0])
        ax_plot.axvspan(min_r, max_r, color=color_stn_sn_green, alpha=0.25)
        ax_plot.axvline(obs_rho, color="#404040", linestyle="--", linewidth=2)
        ax_plot.plot(df_lopo["Rho"], y_ticks, "o", color=color_stn_sn_green, markersize=9, zorder=3)
        
        for y_pos, rho_val in zip(y_ticks, df_lopo["Rho"]):
            ax_plot.annotate(f"{rho_val:.2f}", (rho_val, y_pos), textcoords="offset points", 
                            xytext=(7, -3), fontsize=10, fontweight="bold", fontfamily="Arial")
        
        # Use formatted display IDs (Ptn01..Ptn08) on the y-axis
        lopo_labels = [f"Excl. {pid}" for pid in df_lopo["Display_ID"]]
        ax_plot.set_yticks(y_ticks)
        ax_plot.set_yticklabels(lopo_labels, fontweight="bold", fontsize=12, fontfamily="Arial")
        
        ax_plot.set_title(f"LOPO Influence: {struct} ({model_type})", fontweight="bold", fontsize=14, pad=38, fontfamily="Arial")
        ax_plot.set_xlabel("Spearman $\\rho$ (Recalculated)", fontweight="bold", fontsize=13, labelpad=10, fontfamily="Arial")
        ax_plot.set_xlim(-1.0, 1.1)
        
        lopo_subtitle = (
            f"Full Cohort $\\rho = {obs_rho:.2f}$\n"
            f"LOPO Range: [{min_r:.2f}, {max_r:.2f}]  ($\\Delta = {max_r - min_r:.2f}$)"
        )
        ax_plot.text(
            0.5, 1.02, lopo_subtitle, transform=ax_plot.transAxes,
            ha="center", va="bottom", fontsize=11, fontweight="bold", fontfamily="Arial",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="whitesmoke", alpha=0.9)
        )

        ax_plot.tick_params(axis="both", which="major", labelsize=11)
        plt.setp(ax_plot.get_xticklabels(), fontweight="bold", fontfamily="Arial")

        ax_plot.spines["top"].set_visible(False)
        ax_plot.spines["right"].set_visible(False)
        ax_plot.grid(True, linestyle=":", linewidth=0.6, alpha=0.4, color="#C0C0C0", zorder=0)

        # Lower panel: Detailed Summary Table using Display_ID
        ax_tbl = fig2.add_subplot(gs[1])
        ax_tbl.axis("off")
        
        table_data = []
        for _, row in df_lopo.iterrows():
            table_data.append([
                f"Excl. {row['Display_ID']}",
                f"{row['Rho']:.3f}",
                f"{row['p_val']:.3f}",
                f"[{row['Fisher_CI_Low']:.2f}, {row['Fisher_CI_High']:.2f}]",
                f"{row['Delta_Rho']:+.3f}"
            ])
        
        col_labels = ["Excluded", "Spearman $\\rho$", "$p$-value", "Fisher 95% CI", "$\\Delta\\rho$"]
        
        tbl = ax_tbl.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center"
        )
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.0)
        tbl.scale(1.0, 1.3)
        
        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(fontfamily="Arial")
            if r == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#404040")
            else:
                if r % 2 == 0:
                    cell.set_facecolor("#F9F9F9")
                else:
                    cell.set_facecolor("white")

        plt.show()

# =====================================================================
# 3. RUN PRIMARY (NORMALIZED) & SENSITIVITY (RAW) ANALYSES
# =====================================================================
# 1. Normalized Connectivity Pipeline
run_standalone_connectivity_figures(is_normalized=True)

# 2. Raw Connectivity Pipeline (Sensitivity Analysis)
run_standalone_connectivity_figures(is_normalized=False)

#%%

# =====================================================================
# EXTRACT EXACT PMv VALUES FOR FIGURE 1E
# =====================================================================
struct = "PMv"
left_file = os.path.join(data_dir, f"left_{struct}_8ptns.mat")
right_file = os.path.join(data_dir, f"right_{struct}_8ptns.mat")

mat_L = sio.loadmat(left_file)['stats'][0, 0]['fccorr'][0, 0]
mat_R = sio.loadmat(right_file)['stats'][0, 0]['fccorr'][0, 0]

# Extract Left, Right, and Bilateral Sums
norm_left = mat_L['nleft'].flatten()
norm_right = mat_R['nright'].flatten()
norm_bilateral = norm_left + norm_right  # <-- Exact X-axis for Fig 1E

raw_left = mat_L['left'].flatten()
raw_right = mat_R['right'].flatten()
raw_bilateral = raw_left + raw_right

# Build the exact table
df_fig1e_exact = pd.DataFrame({
    "Patient_ID": patient_ids,
    "Display_ID": [display_label_map.get(pid, pid) for pid in patient_ids],
    "PMv_Normalized_Bilateral": norm_bilateral,
    "PMv_Normalized_Left": norm_left,
    "PMv_Normalized_Right": norm_right,
    "PMv_Raw_Bilateral": raw_bilateral,
    "Delta_FoG_AC": y_data  # <-- Exact Y-axis for Fig 1E
})

print("\n" + "="*90)
print("              EXACT DATA VALUES FOR FIGURE 1E (PMv CONNECTIVITY)              ")
print("="*90)
print(df_fig1e_exact.to_string(index=False))
print("="*90)

# Optional: Save to CSV to attach directly for your boss
df_fig1e_exact.to_csv("Figure_1E_PMv_Exact_Values.csv", index=False)
print("Saved to: Figure_1E_PMv_Exact_Values.csv")

