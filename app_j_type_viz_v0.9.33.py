# J-type-like Web Viz (v0.9.33)
# ---------------------------------------------------
# Changelog (from v0.9.30 -> v0.9.33):
# 1) 新增 Mode 切換（Basic / Advanced），Basic 僅顯示常用設定；Advanced 顯示完整控制。
# 2) pairwise 線避重邏輯維持原設定（固定間距 + 手動參數）。
# 3) Bar 與 Box 圖皆支援同時顯示上下 (n=xx)，提供獨立顏色與 alpha 控制。
# 4) 預設套用 Publication-ready 風格；Style preset 預設為 "Publication-ready"。
#
# 其餘功能（Bar/Box/Scatter/導出/summary/向量字體/Okabe–Ito 等）皆沿用 v0.9.30。

from matplotlib import cycler as _cycler
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import io, math, re, warnings
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# --- helper: color with alpha (keeps back-compat with hex/color names) ---
from matplotlib.colors import to_rgba as _to_rgba
def _rgba(color, alpha):
    r, g, b, _ = _to_rgba(color)
    return (r, g, b, float(alpha))


# ---------------- rcParams for vector text -----------------
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

# ===================== Publication-ready Style =====================
def apply_publication_style(enable_pub_style: bool):
    if not enable_pub_style:
        return
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#000000",
        "grid.color": "#DDDDDD",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300
    })
    try:
        import seaborn as sns
        sns.set_palette("colorblind")
    except Exception:
        pass

# ===================== Helpers =====================
def apply_legend_style(ax,
                       transparent_bg=True,
                       show_border=False,
                       border_color="#000000",
                       border_width=1.0):
    leg = ax.get_legend() or getattr(ax, "legend_", None)
    if leg is None:
        return

    frame = leg.get_frame()
    # 背景透明
    if transparent_bg:
        frame.set_facecolor("none")
        frame.set_alpha(0.0)
    else:
        frame.set_facecolor("#FFFFFF")
        frame.set_alpha(0.9)

    # ✅ 邊框控制區
    if show_border:
        frame.set_edgecolor(border_color)
        frame.set_linewidth(float(border_width))
        frame.set_alpha(1.0)  # 確保透明時仍顯示邊線
        frame.set_zorder(10)
        try:
            frame.set_boxstyle("round,pad=0.3")  # 可選：圓角卡片風格
        except Exception:
            pass
    else:
        frame.set_edgecolor("none")


NUMERIC_KINDS = set("biufc")

def is_numeric_series(s: pd.Series) -> bool:
    try:
        return s.dtype.kind in NUMERIC_KINDS
    except Exception:
        return False

def sem(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n <= 1:
        return np.nan
    return np.nanstd(a, ddof=1) / max(1.0, np.sqrt(n))

def polyfit_regression(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    b, a = np.polyfit(x[mask], y[mask], 1)
    yhat = a + b * x[mask]
    ss_res = np.sum((y[mask] - yhat) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b, a, r2

def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    if name.endswith(".jmp"):
        raise RuntimeError(".jmp not directly supported. Export as CSV/XLSX.")
    return pd.read_csv(uploaded)

def example_dataframe():
    rng = np.random.default_rng(42)
    n = 180
    df = pd.DataFrame({
        "Group": rng.choice(["A", "B", "C"], n, p=[.4, .4, .2]),
        "X": rng.choice(["Day1", "Day2", "Day3"], n),
        "Y": rng.normal(100, 15, n) + rng.choice([0, 5, -5], n)
    })
    return df

def p_to_stars(value, thr1=0.05, thr2=0.01, thr3=0.001, show_ns=False) -> str:
    if isinstance(value, str):
        if value.strip().lower() == "ns":
            return "ns"
        try:
            p = float(value)
        except Exception:
            return ""
    else:
        try:
            p = float(value)
        except Exception:
            return ""
    if p < thr3:
        return "***"
    if p < thr2:
        return "**"
    if p < thr1:
        return "*"
    return "ns" if show_ns else ""

# --- sync whisker caps (讓帽子線寬 = Box 寬) ---
def sync_whisker_caps_to_boxwidth(bp, box_width=0.6, cap_lw=None, cap_color=None):
    if not bp or "caps" not in bp:
        return
    caps = bp["caps"]
    n_pairs = len(caps) // 2
    for i in range(n_pairs):
        cap_top = caps[2*i]
        cap_bot = caps[2*i + 1]
        x_mid = np.mean(cap_top.get_xdata())
        half_w = box_width / 2.0
        new_x = [x_mid - half_w, x_mid + half_w]
        cap_top.set_xdata(new_x); cap_bot.set_xdata(new_x)
        if cap_lw is not None:
            cap_top.set_linewidth(cap_lw); cap_bot.set_linewidth(cap_lw)
        if cap_color is not None:
            cap_top.set_color(cap_color); cap_bot.set_color(cap_color)

# ---- v0.9.25 Box pairwise addons: stair overlap avoidance ----
def _assign_layers_for_pairs_box(pairs, tol=1e-12):
    layers = []
    out_layers = []
    for (x1, x2) in pairs:
        a, b = (x1, x2) if x1 <= x2 else (x2, x1)
        placed = False
        for li, layer in enumerate(layers):
            if any(not (b < la - tol or a > lb + tol) for (la, lb) in layer):
                continue
            layer.append((a, b))
            out_layers.append(li)
            placed = True
            break
        if not placed:
            layers.append([(a, b)])
            out_layers.append(len(layers) - 1)
    return out_layers

def _format_eq(b: float, a: float, r2: float, show_r2=True, show_eq=True) -> str:
    parts = []
    if show_eq:
        parts.append(f"y = {a:.3g} + {b:.3g}·x")
    if show_r2 and (r2 == r2):
        parts.append(f"R²={r2:.2f}")
    return ", ".join(parts) if parts else ""

def sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r'[\\/:*?"<>|]+', '-', s)
    s = re.sub(r'[_-]{2,}', lambda m: m.group(0)[0], s)
    s = s.strip("_-")
    return s or "Figure"

def safe_float(v: Any, default: float) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default

# --- Count utilities used by Bar & Box ---
def _compute_category_counts(df, x_col, group_col=None):
    if group_col and group_col not in (None, "None"):
        df_tmp = df.dropna(subset=[x_col, group_col])
        df_tmp["combo"] = df_tmp[x_col].astype(str) + "_" + df_tmp[group_col].astype(str)
        levels = sorted(df_tmp["combo"].unique())
        counts = df_tmp["combo"].value_counts()
        positions = list(range(len(levels)))
        n_list = [int(counts.get(lbl, 0)) for lbl in levels]
        return positions, levels, n_list
    else:
        x_levels = list(pd.Index(df[x_col].dropna().astype(str).unique()))
        counts = df[x_col].dropna().astype(str).value_counts()
        positions = list(range(len(x_levels)))
        n_list = [int(counts.get(lbl, 0)) for lbl in x_levels]
        return positions, x_levels, n_list

def draw_sample_counts(ax,
                       positions, labels, n_list,
                       show_top=True, show_bottom=True,
                       top_offset_rel=0.03, bottom_offset_rel=0.06,
                       top_color="#000000", bottom_color="#000000",
                       top_alpha=1.0, bottom_alpha=1.0,
                       fontsize=10, fmt="n={n}"):
    """Draw sample counts on top and/or bottom of each category with separate colors/alphas."""
    y_min, y_max = ax.get_ylim()
    y_range = max(1e-12, (y_max - y_min))
    y_top_base = y_max
    y_bottom_base = y_min
    for xi, (xpos, lbl, nval) in enumerate(zip(positions, labels, n_list)):
        text_str = fmt.format(n=nval)
        if show_top:
            y_text_top = y_top_base + y_range * max(0.0, top_offset_rel)
            ax.text(xpos, y_text_top, text_str,
                    ha="center", va="bottom", fontsize=fontsize,
                    color=top_color, alpha=top_alpha, clip_on=False, zorder=6)
        if show_bottom:
            ax.text(
                xpos, -0.15 - bottom_offset_rel, text_str,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=fontsize, color=bottom_color, alpha=bottom_alpha,
                clip_on=False, zorder=6)

# Rotation helper
def apply_x_label_rotation(ax, rotation):
    try:
        x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        auto_rotation = 45 if (rotation == 0 and (len(x_labels) > 5 or any(len(lbl) > 6 for lbl in x_labels))) else rotation
        ax.set_xticklabels(x_labels, rotation=auto_rotation,
                           ha='right' if auto_rotation > 0 else 'center')
    except Exception:
        pass

# ===================== UI =====================
st.set_page_config(page_title="J-type-like Web Viz v0.9.33", layout="wide")

# --- CSS（三欄可捲動 + scrollbar-gutter） ---
st.markdown("""
<style>
[data-testid="stHorizontalBlock"] { align-items: flex-start !important; }
[data-testid="stVerticalBlock"] {
    height: 90vh !important; overflow-y: auto !important;
    scrollbar-gutter: stable both-edges;
    padding: 0.5rem 1.2rem 0.5rem 0.8rem !important;
    background: #f9f9fb !important; border-radius: 8px;
    box-shadow: inset 0 0 6px rgba(0,0,0,0.05); box-sizing: border-box;
}
[data-testid="stVerticalBlock"]::-webkit-scrollbar { width: 12px; }
[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
    border-radius: 8px; background-clip: padding-box;
    background: rgba(0,0,0,.25); border: 3px solid transparent;
}
[data-testid="stVerticalBlock"]::-webkit-scrollbar-track { background: transparent; }
[data-testid="stVerticalBlock"] { scrollbar-width: thin; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div {
    padding-top: 0.2rem !important;
    padding-bottom: 0.4rem !important;
}
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🧭 J-type Web Viz v0.9.33")
    st.header("1) Upload data")
    file_types = ["csv", "xlsx", "xls", "jmp"]
    up = st.file_uploader("Upload .csv / .xlsx / .jmp", type=file_types)
    use_example = st.checkbox("Use example data", value=up is None)

# --- Load Data ---
if use_example:
    df = example_dataframe()
    st.info("Using built-in example data")
elif up is not None:
    try:
        df = read_uploaded_file(up)
    except Exception as e:
        st.error(f"Failed: {e}")
        st.stop()
else:
    st.warning("Please upload or use example data.")
    st.stop()

# --- Mode 切換（Basic / Advanced）---
mode = st.radio("Mode", ["Basic", "Advanced"], horizontal=True, index=0)
# --- CSS：修正 scroll-col 無法輸入問題 ---
st.markdown("""
<style>
scroll-col {
    max-height: 90vh;           /* 固定高度，允許滾動 */
    overflow-y: auto;
    padding-right: 0.8rem;
    background-color: #f9f9fb;
    border-radius: 8px;
    box-shadow: inset 0 0 6px rgba(0,0,0,0.05);

    /* 🚀 關鍵修正：讓內部元件可被點擊與輸入 */
    position: relative;         
    z-index: 0;                 
}
</style>
""", unsafe_allow_html=True)

# --- Create Columns ---
colA, colB, colC = st.columns([2.5, 2, 3])

with colA:
    st.markdown('<div class="scroll-col">', unsafe_allow_html=True)
    st.subheader("2) Rename Columns & Category Labels")
    rename_map: Dict[str, str] = {}
    with st.expander("Rename columns", expanded=(mode == "Advanced")):
        for col in df.columns:
            newname = st.text_input(f"Rename {col}", value=col, key=f"rename_{col}")
            rename_map[col] = newname
    if rename_map:
        df = df.rename(columns=rename_map)

    numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
    cat_cols = [c for c in df.columns if not is_numeric_series(df[c])]

    if len(df.columns) < 2:
        st.error("Dataset must have at least two columns.")
        st.stop()

    st.markdown("---")

    # 3) Plot Settings
    st.subheader("3) Plot Settings")
    plot_type = st.radio("Plot type", ["Bar (mean ± SE)", "Scatter", "Box (show samples)"], horizontal=True)
    group_col = st.selectbox("Group (optional)", [None]+df.columns.tolist(), key="group_col")

    # ---------- BAR ----------
    if plot_type.startswith("Bar"):
        x_col = st.selectbox("X (categorical)", (cat_cols or df.columns.tolist()), key="bar_x")
        y_col = st.selectbox("Y (numeric)", (numeric_cols or df.columns.tolist()), key="bar_y")
        if y_col not in df.columns or not is_numeric_series(df[y_col]):
            st.error("Selected Y must be numeric.")
            st.stop()
        show_se = st.checkbox("Show SE bars", True)

        st.markdown("### ⚙️ Bar & Layout Settings (inch)")
        if mode == "Advanced":
            bar_width_inch = st.number_input("Bar width (inch)", value=0.25, min_value=0.05, max_value=2.0, step=0.05)
            bar_spacing_inch = st.number_input("Between-bar spacing (inch)", value=0.15, min_value=0.0, max_value=2.0, step=0.05)
            margin_left_inch = st.number_input("Left margin (inch)", value=0.5, min_value=0.0, max_value=3.0, step=0.1)
            margin_right_inch = st.number_input("Right margin (inch)", value=0.5, min_value=0.0, max_value=3.0, step=0.1)
        else:
            bar_width_inch, bar_spacing_inch = 0.25, 0.15
            margin_left_inch, margin_right_inch = 0.5, 0.5

        st.markdown("### 🧱 Bar Edge & Errorbar Style")
        if mode == "Advanced":
            bar_edgecolor = st.color_picker("Bar edge color", "#000000")
            bar_linewidth = st.slider("Bar edge linewidth", 0.0, 4.0, 0.6)
            err_capsize = st.slider("Errorbar capsize (pt)", 0.0, 20.0, 5.0)
            err_linewidth = st.slider("Errorbar linewidth", 0.2, 4.0, 1.2)
        else:
            bar_edgecolor, bar_linewidth, err_capsize, err_linewidth = "#000000", 0.6, 5.0, 1.2

        st.markdown("### 🟡 Overlay sample points")
        bar_show_points = st.checkbox("Show samples on bars", False)
        if mode == "Advanced":
            bar_point_size = st.slider("Point size", 5, 200, 60)
            bar_point_alpha = st.slider("Point alpha", 0.1, 1.0, 0.6)
            bar_point_jitter = st.slider("Horizontal jitter (axes fraction)", 0.0, 0.2, 0.06)
        else:
            bar_point_size, bar_point_alpha, bar_point_jitter = 60, 0.6, 0.06

        # ---- v0.9.31: (n=xx) 上下同時顯示（Bar） ----
        st.markdown("### 🔢 (n=xx) Labels – Bar")
        show_n_labels = st.checkbox("Enable (n=xx) for bars", True)
        if show_n_labels:
            coln1, coln2 = st.columns(2)
            with coln1:
                show_n_top_bar = st.checkbox("Show top (n=xx)", True)
                n_top_color_bar = st.color_picker("Top (n) color", "#222222")
                n_top_alpha_bar = st.slider("Top (n) alpha", 0.1, 1.0, 0.95)
                n_top_offset_bar = st.slider("Top (n) offset (rel y-range)", 0.0, 0.5, 0.03)
            with coln2:
                show_n_bottom_bar = st.checkbox("Show bottom (n=xx)", True)
                n_bottom_color_bar = st.color_picker("Bottom (n) color", "#000000")
                n_bottom_alpha_bar = st.slider("Bottom (n) alpha", 0.1, 1.0, 0.9)
                n_bottom_offset_bar = st.slider("Bottom (n) offset (rel y-range)", 0.0, 0.5, 0.08)
            n_fontsize = st.slider("Font size for (n=xx)", 6, 20, 10)
            n_bold = st.checkbox("Bold (n=xx)", False)
            show_n_inside = st.checkbox("Also show inside-bar (if bar>0)", False)
            n_inside_offset = st.slider("Inside-bar (n) offset (fraction of bar height)", 0.00, 0.50, 0.15)
            n_inside_bg = st.checkbox("Inside-bar (n) white background", False)

        st.markdown("### 🧮 Bottom-center n summary")
        bottom_n_summary = st.selectbox("Show bottom-center n summary", ["Off", "By X-level", "By Group"], index=0)
        bottom_n_font = st.slider("Bottom summary font size", 6, 20, 10)
        bottom_n_offset = st.slider("Bottom summary offset (relative y-range)", 0.0, 0.5, 0.08)

        st.markdown("### 🔢 Value Labels on Bars")
        if mode == "Advanced":
            show_value_labels = st.checkbox("Show mean value on bar tops", False)
            value_decimals = st.number_input("Decimals", value=2, step=1, min_value=0, max_value=6)
            value_fontsize = st.slider("Value label fontsize", 6, 24, 10)
            value_bold = st.checkbox("Value label bold", False)
        else:
            show_value_labels, value_decimals, value_fontsize, value_bold = False, 2, 10, False

    # ---------- BOX ----------
    elif plot_type.startswith("Box"):
        x_col = st.selectbox("X (categorical)", (cat_cols or df.columns.tolist()), key="box_x")
        y_col = st.selectbox("Y (numeric)", (numeric_cols or df.columns.tolist()), key="box_y")
        show_points = st.checkbox("Show individual sample points", True)
        if mode == "Advanced":
            point_size = st.slider("Point size", 5, 100, 30)
            point_alpha = st.slider("Point alpha", 0.2, 1.0, 0.6)
            jitter_width = st.slider("Jitter width", 0.0, 0.5, 0.2)
        else:
            point_size, point_alpha, jitter_width = 30, 0.6, 0.2

        box_width = st.slider("Box width", 0.1, 0.9, 0.6)
        box_edge_lw = st.slider("Box edge linewidth", 0.5, 5.0, 1.5, 0.1)
        box_edge_color = st.color_picker("Box edge color", "#333333")
        box_fill_color = st.color_picker("Box fill color", "#4C78A8")
        box_alpha = st.slider("Box fill alpha", 0.1, 1.0, 0.9, 0.05)

        # ---- v0.9.31: (n=xx) 上下同時顯示（Box） ----
        st.markdown("### 🔢 (n=xx) Labels – Box")
        box_show_n = st.checkbox("Show (n=xx) per category", True)
        if box_show_n:
            colbn1, colbn2 = st.columns(2)
            with colbn1:
                box_show_top = st.checkbox("Show top (n=xx) – Box", True)
                box_top_color = st.color_picker("Top (n) color – Box", "#222222")
                box_top_alpha = st.slider("Top (n) alpha – Box", 0.1, 1.0, 0.95)
                box_top_offset_rel = st.slider("Top (n) offset (rel y-range) – Box", 0.0, 0.5, 0.03)
            with colbn2:
                box_show_bottom = st.checkbox("Show bottom (n=xx) – Box", True)
                box_bottom_color = st.color_picker("Bottom (n) color – Box", "#000000")
                box_bottom_alpha = st.slider("Bottom (n) alpha – Box", 0.1, 1.0, 0.9)
                box_bottom_offset_rel = st.slider("Bottom (n) offset (rel y-range) – Box", 0.0, 0.5, 0.10)
            box_n_fontsize = st.slider("n label fontsize (Box)", 6, 24, 10)

        box_xtick_offset_rel = st.slider(
            "X axis category label vertical offset (rel y-range)", 0.0, 0.3, 0.05, 0.01
        )

    # ---------- SCATTER ----------
    else:
        if len(numeric_cols) < 2:
            st.error("Scatter requires at least two numeric columns.")
            st.stop()
        x_col = st.selectbox("X (numeric)", numeric_cols, key="scat_x")
        y_col = st.selectbox("Y (numeric)", [c for c in numeric_cols if c != x_col], key="scat_y")
        add_reg = st.checkbox("Show regression line", True)
        show_r2 = st.checkbox("Legend: show R²", True)
        show_eq = st.checkbox("Legend: show equation", False)
        if mode == "Advanced":
            point_size = st.slider("Point size", 10, 200, 50)
            point_alpha = st.slider("Point alpha", 0.1, 1.0, 0.7)
        else:
            point_size, point_alpha = 50, 0.7

    # Global Font
    st.markdown("### 🖋️ Global Font")
    font_options = ["Arial", "Helvetica", "Times New Roman", "Calibri", "Liberation Sans"]
    font_family = st.selectbox("Global font family", font_options, index=0)
    plt.rcParams.update({"font.family": font_family})

    # Colors, shapes, legend
    st.markdown("### 🎨 Colors, Shapes & Legend Labels")
    group_colors: Dict[Any, str] = {}
    group_shapes: Dict[Any, str] = {}
    group_labels: Dict[Any, str] = {}
    legend_custom_order: List[str] = []
    legend_fontsize = 10
    legend_title_fontsize = 10

    if group_col not in (None, "None"):
        g_levels = list(pd.Index(df[group_col].dropna().astype(str).unique()))
    
        # ✅ 統一改成橫列卡片排版（所有圖型皆用）    
        # 一排最多顯示幾個卡片（可自行改成 3~5）
        per_row = 4
        for row_start in range(0, len(g_levels), per_row):
            row_groups = g_levels[row_start: row_start + per_row]
            cols = st.columns(len(row_groups))
            for col_i, g in enumerate(row_groups):
                with cols[col_i]:
                    st.markdown(f"**{g}**")
                    group_colors[g] = st.color_picker(
                        "Color", value="#1f77b4", key=f"gcol_{g}"
                    )
                    group_shapes[g] = st.selectbox(
                        "Shape",
                        {"o": "○ Circle", "s": "□ Square", "^": "△ Triangle", "D": "◇ Diamond"},
                        key=f"shape_{g}"
                    )
                    group_labels[g] = st.text_input(
                        "Legend label", value=g, key=f"glabel_{g}"
                    )
                    st.markdown("---")
    
        with st.expander("Legend order", expanded=False if mode == "Basic" else True):
            st.caption("自訂圖例順序（以逗號分隔；使用上方右列『Label for ...』文字）")
        
            default_order = ", ".join([group_labels[g] for g in g_levels])
            order_input = st.text_input("Custom legend order", value=default_order)
            legend_custom_order = [x.strip() for x in order_input.split(",") if x.strip()]
        
            st.markdown("---")
            st.markdown("**Legend font settings**")
        
            # 🔹 第一橫排
            row1 = st.columns(1)
            with row1[0]:
                legend_fontsize = st.slider("Legend font size", 6, 24, 10)
                legend_title_fontsize = st.slider("Legend title font size", 6, 28, 10)

            st.markdown("---")
            st.markdown("**Legend appearance**")
            row1 = st.columns(1)
            with row1[0]:
                legend_bg_transparent = st.checkbox("Transparent bg", True)
                legend_show_border = st.checkbox("Show border", False)
                legend_border_color = st.color_picker("Border color", "#000000", disabled=not legend_show_border)
                legend_border_width = st.slider("Border width", 0.2, 4.0, 1.0, 0.1, disabled=not legend_show_border)


    # X-level colors (Bar)
    if plot_type.startswith("Bar"):
        st.markdown("### 🎨 X-level Colors")
        x_levels = list(pd.Index(df[x_col].dropna().astype(str).unique()))
        x_colors: Dict[str, str] = {}
        for xv in x_levels:
            x_colors[xv] = st.color_picker(f"Color for {xv}", value="#1f77b4", key=f"xcolor_{xv}")
        override_by_x = st.checkbox("Override group colors by X-level", False)
    else:
        x_colors = {}
        override_by_x = False

    st.markdown("### 📏 Figure size")
    fig_width = st.slider("Width (inch)", 4.0, 15.0, 8.0)
    fig_height = st.slider("Height (inch)", 3.0, 10.0, 5.0)
    st.markdown('</div>', unsafe_allow_html=True)

# 4) Axis and labels
with colB:
    st.markdown('<div class="scroll-col">', unsafe_allow_html=True)
    st.subheader("4) Axis & Text Settings")
    main_title = st.text_input("Main title", value=f"{y_col} vs {x_col}")
    xlabel = st.text_input("X-axis label", value=x_col)
    ylabel = st.text_input("Y-axis label", value=y_col)
    font_size = st.slider("Base font size", 8, 24, 12)
    bold = st.checkbox("Bold axis labels", False)

    x_tick_rotation = st.slider("X tick rotation (deg)", 0, 90, 0)
    x_tick_fontsize = st.slider("X tick fontsize", 6, 20, 10)
    y_tick_fontsize = st.slider("Y tick fontsize", 6, 20, 10)
    xlabel_pad = st.slider("X-axis labelpad (distance)", 0, 120, 50)

    y0_line  = st.checkbox("Draw horizontal line at y=0", False)
    y0_color = st.color_picker("y=0 line color", "#000000")
    x_axis_lw = st.slider("X axis width", 0.2, 5.0, 1.0, 0.1)
    x_axis_color = st.color_picker("X axis color", "#000000")
    y_axis_lw = st.slider("Y axis width", 0.2, 5.0, 1.0, 0.1)
    y_axis_color = st.color_picker("Y axis color", "#000000")
    tick_color = st.color_picker("Tick/Title color", "#000000")

    label_map: Dict[str, str] = {}
    custom_order = []
    if plot_type.startswith("Bar") or plot_type.startswith("Box"):
        x_levels_default = list(pd.Index(df[x_col].dropna().astype(str).unique()))
        with st.expander("Edit X-category labels & order", expanded=False if mode == "Basic" else True):
            for xv in x_levels_default:
                label_map[xv] = st.text_input(f"Label for {xv}", value=xv, key=f"xlabel_{xv}")
            st.caption("Enter desired X order (comma-separated):")
            order_input = st.text_input("Custom X order", value=", ".join(x_levels_default))
            custom_order = [x.strip() for x in order_input.split(",") if x.strip() in x_levels_default]
    else:
        st.markdown("### 📈 X-axis (numeric)")
        x_min = st.number_input("X min", value=float(df[x_col].min()))
        x_max = st.number_input("X max", value=float(df[x_col].max()))
        x_step = st.number_input("X tick step", value=1.0)
        x_dec = st.number_input("X decimals", value=1, step=1)

    y_min = st.number_input("Y min", value=float(df[y_col].min()))
    y_max = st.number_input("Y max", value=float(df[y_col].max()))
    y_step = st.number_input("Y tick step", value=1.0)
    y_dec = st.number_input("Y decimals", value=1, step=1)
    lock_nice_ticks = st.checkbox("Lock nice ticks (align to step)", False)

    st.markdown("### 🧭 Grid")
    grid_x = st.checkbox("Show x-grid", False)
    grid_y = st.checkbox("Show y-grid", True)
    grid_linewidth = st.slider("Grid linewidth", 0.2, 2.5, 0.6)

    st.markdown("### 📍 Axis intersection control")
    fix_y_intercept = st.checkbox("Fix Y-axis intersection at specific value", False)
    if fix_y_intercept:
        y_intercept_value = st.number_input("Y value to align with X-axis", value=0.0)
    else:
        y_intercept_value = None

    st.markdown("### 🧱 Spines")
    show_spine_left   = st.checkbox("Show left spine", True,  key="sp_left")
    show_spine_bottom = st.checkbox("Show bottom spine", True, key="sp_bottom")
    show_spine_right  = st.checkbox("Show right spine", False, key="sp_right")
    show_spine_top    = st.checkbox("Show top spine", False,  key="sp_top")

    # 5) Manual Stats + Pairwise
    st.markdown("---")
    st.subheader("5) Manual Statistics Annotation")
    show_stats = st.checkbox("Show stats on plot", True)
    position = st.selectbox("Annotation position", ["top-left", "top-right", "bottom-left", "bottom-right"])
    stat_font_size = st.slider("Stats font size", 8, 24, 12)
    stat_color = st.color_picker("Stats text color", value="#000000")
    t_in    = st.text_input("t value", key="t_in")
    f_in    = st.text_input("F value", key="f_in")
    p_in    = st.text_input("p value", key="p_in")
    note_in = st.text_input("note",     key="note_in")
    # === Legend position control ===
    legend_loc = st.selectbox(
        "Legend position (if grouping)",
        options=[
            "best", "upper right", "upper left", "lower right", "lower left",
            "upper center", "lower center", "center left", "center right", "center", "none"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("5b) Pairwise Significance Lines & Stars")
    enable_sig = st.checkbox("Enable pairwise significance", False)
    hide_ns = st.checkbox("Hide non-significant (p>=thr)", True)
    thr_1 = st.number_input("p < threshold (*)", value=0.05, format="%.5f")
    thr_2 = st.number_input("p < threshold (**)", value=0.01, format="%.5f")
    thr_3 = st.number_input("p < threshold (***)", value=0.001, format="%.5f")

    sig_line_width = st.slider("Pairwise line width", 0.5, 5.0, 1.5)
    sig_line_color_default = st.color_picker("Pairwise default line color", "#000000")
    sig_stack_gap = st.slider("Stacking gap per comparison (relative y-range)", 0.01, 0.25, 0.05)
    sig_line_lift = st.slider("Lift above tallest bar (relative y-range)", 0.0, 0.4, 0.06)
    sig_tick_length = st.slider("End tick length (relative y-range)", 0.002, 0.05, 0.01)
    sig_star_font = st.slider("Pairwise text size", 8, 32, 12)
    sig_star_bold = st.checkbox("Pairwise text bold", True)
    sig_star_extra_offset = st.slider("Text extra offset (relative y-range)", 0.0, 0.15, 0.02)
    max_pairs = 12
    pair_count = st.number_input("Number of comparisons", min_value=0, max_value=max_pairs, value=0, step=1)
    sig_pairs: List[Dict[str, Any]] = []
    if enable_sig and pair_count > 0 and plot_type.startswith("Bar"):
        st.caption("Select targets to compare. For grouped bars, pick both X and Group for each side.")
        for i in range(int(pair_count)):
            with st.expander(f"Comparison #{i+1}", expanded=False):
                if group_col in (None, "None"):
                    x_lvls = list(pd.Index(df[x_col].dropna().astype(str).unique()))
                    c1 = st.selectbox("X1", x_lvls, key=f"sig_x1_{i}")
                    c2 = st.selectbox("X2", [x for x in x_lvls if x != st.session_state.get(f'sig_x1_{i}', x_lvls[0])], key=f"sig_x2_{i}")
                    pval = st.text_input("p-value (number or 'ns')", key=f"sig_p_{i}")
                    line_color = st.color_picker("Line color", sig_line_color_default, key=f"sig_color_{i}")
                    sig_pairs.append({"x1": c1, "g1": None, "x2": c2, "g2": None, "p": pval, "color": line_color})
                else:
                    x_lvls = list(pd.Index(df[x_col].dropna().astype(str).unique()))
                    g_lvls = list(pd.Index(df[group_col].dropna().astype(str).unique()))
                    row1c = st.columns(2)
                    with row1c[0]:
                        x1 = st.selectbox("X1", x_lvls, key=f"sig_x1_{i}")
                    with row1c[1]:
                        g1 = st.selectbox("Group1", g_lvls, key=f"sig_g1_{i}")
                    row2c = st.columns(2)
                    with row2c[0]:
                        x2 = st.selectbox("X2", x_lvls, key=f"sig_x2_{i}")
                    with row2c[1]:
                        g2 = st.selectbox("Group2", g_lvls, key=f"sig_g2_{i}")
                    pval = st.text_input("p-value (number or 'ns')", key=f"sig_p_{i}")
                    line_color = st.color_picker("Line color", sig_line_color_default, key=f"sig_color_{i}")
                    sig_pairs.append({"x1": x1, "g1": g1, "x2": x2, "g2": g2, "p": pval, "color": line_color})
    elif enable_sig and pair_count > 0 and plot_type.startswith("Box"):
        st.caption("Select targets to compare (Box).")
        x_lvls = list(pd.Index(df[x_col].dropna().astype(str).unique()))
        for i in range(int(pair_count)):
            with st.expander(f"Comparison (Box) #{i+1}", expanded=False):
                c1 = st.selectbox("X1", x_lvls, key=f"sig_box_x1_{i}")
                c2 = st.selectbox("X2", [x for x in x_lvls if x != st.session_state.get(f'sig_box_x1_{i}', x_lvls[0])], key=f"sig_box_x2_{i}")
                pval = st.text_input("p-value (number or 'ns')", key=f"sig_box_p_{i}")
                line_color = sig_line_color_default
                sig_pairs.append({"x1": c1, "g1": None, "x2": c2, "g2": None, "p": pval, "color": line_color})

    # Plotting
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('none')
    plt.rcParams.update({"font.size": font_size})
    fontweight = 'bold' if bold else 'normal'

    # Axes look
    ax.spines["bottom"].set_linewidth(x_axis_lw); ax.spines["bottom"].set_color(x_axis_color)
    ax.spines["top"].set_linewidth(x_axis_lw);    ax.spines["top"].set_color(x_axis_color)
    ax.spines["left"].set_linewidth(y_axis_lw);   ax.spines["left"].set_color(y_axis_color)
    ax.spines["right"].set_linewidth(y_axis_lw);  ax.spines["right"].set_color(y_axis_color)
    ax.tick_params(axis='x', width=max(0.2, x_axis_lw), colors=tick_color)
    ax.tick_params(axis='y', width=max(0.2, y_axis_lw), colors=tick_color)
    ax.xaxis.label.set_color(tick_color); ax.yaxis.label.set_color(tick_color); ax.title.set_color(tick_color)
    ax.spines["top"].set_visible(bool(show_spine_top))
    ax.spines["right"].set_visible(bool(show_spine_right))
    ax.spines["bottom"].set_visible(bool(show_spine_bottom))
    ax.spines["left"].set_visible(bool(show_spine_left))
    ax.xaxis.set_ticks_position('bottom' if show_spine_bottom else ('top' if show_spine_top else 'none'))
    ax.yaxis.set_ticks_position('left' if show_spine_left else ('right' if show_spine_right else 'none'))

    needed_top = y_max
    needed_bottom = y_min

    def _draw_bottom_center_summary(ax, text, y_min_val, y_max_val, y_offset_rel, fontsize):
        y_range = y_max_val - y_min_val
        y_pos = y_min_val - y_range * max(0.0, y_offset_rel)
        ax.text(0.5, y_pos, text, ha="center", va="top", fontsize=fontsize,
                transform=ax.get_xaxis_transform())
        return y_pos

    def _safe_text(ax, *args, **kwargs):
        fs = safe_float(kwargs.pop("fontsize", 10), 10.0)
        fs = max(1.0, fs)
        color = kwargs.pop("color", "#000000")
        return ax.text(*args, fontsize=fs, color=color, **kwargs)

    # --- BAR ---
    bar_centers_map: Dict[Union[str, tuple], float] = {}
    bar_tops_map: Dict[Union[str, tuple], float] = {}
    xpos = None
    summary_table = None

    if plot_type.startswith("Bar"):
        if custom_order:
            df[x_col] = pd.Categorical(df[x_col].astype(str), categories=custom_order, ordered=True)

        grouped = df.groupby([x_col, group_col])[y_col] if group_col not in (None, "None") else df.groupby(x_col)[y_col]

        if group_col in (None, "None"):
            means = grouped.mean()
            ses = grouped.apply(sem)
            counts = grouped.count()
            xpos = np.arange(len(means))
            bar_width = max(0.02, bar_width_inch / max(0.1, fig_width))

            summary_table = pd.DataFrame({
                x_col: means.index.astype(str),
                "Mean": means.values,
                "SE": ses.values,
                "N": [int(counts[ix]) for ix in means.index]
            })

            for i, (xv, mean) in enumerate(zip(means.index, means.values)):
                se = ses.values[i]
                safe_yerr = None if (not show_se or (not (se == se) or se <= 0)) else se
                color = x_colors.get(str(xv), "#1f77b4")
                ax.bar(xpos[i], mean, yerr=safe_yerr, color=color, width=bar_width,
                       edgecolor=bar_edgecolor, linewidth=bar_linewidth,
                       capsize=err_capsize, ecolor=bar_edgecolor, error_kw=dict(lw=err_linewidth))

                bar_centers_map[str(xv)] = float(xpos[i])
                top = (mean if np.isfinite(mean) else 0.0) + (safe_yerr if safe_yerr else 0.0)
                bar_tops_map[str(xv)] = top

                if show_value_labels and np.isfinite(mean):
                    ax.text(xpos[i], top + (y_max - y_min)*0.01,
                            f"{mean:.{int(value_decimals)}f}",
                            ha="center", va="bottom",
                            fontsize=value_fontsize, fontweight=("bold" if value_bold else "normal"))

            xticks = [label_map.get(str(l), str(l)) for l in means.index]
            ax.set_xticks(xpos); ax.set_xticklabels(xticks, rotation=x_tick_rotation, fontsize=x_tick_fontsize)

            # overlay sample points
            if bar_show_points:
                jitter = bar_point_jitter
                for i, xv in enumerate(means.index):
                    vals = df.loc[df[x_col].astype(str) == str(xv), y_col].dropna().to_numpy()
                    if len(vals) == 0: continue
                    x_center = xpos[i]
                    rng = np.random.default_rng(i+2026)
                    jitter_vals = (rng.random(len(vals)) - 0.5) * 2 * jitter
                    ax.scatter(x_center + jitter_vals, vals,
                               s=bar_point_size, alpha=bar_point_alpha, color=x_colors.get(str(xv), "#1f77b4"),
                               edgecolors="k", linewidths=0.3, zorder=3)

            # --- v0.9.31: (n=xx) Top/Bottom 同時顯示（Bar, ungrouped） ---
            if show_n_labels:
                y_min_val, y_max_val = ax.get_ylim()
                y_range = max(1e-12, (y_max_val - y_min_val))
                for x, m in zip(xpos, means.index):
                    nval = int(counts[m])
                    if show_n_inside:
                        mean_val = float(means.loc[m])
                        if np.isfinite(mean_val) and mean_val > 0:
                            y_text = mean_val * max(0.0, n_inside_offset)
                            _safe_text(ax, x, y_text, f"(n={nval})",
                                       ha="center", va="bottom",
                                       fontweight=("bold" if n_bold else "normal"),
                                       fontsize=n_fontsize,
                                       color=x_colors.get(str(m), "#1f77b4") if n_inside_bg else "#000000",
                                       bbox=(dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none') if n_inside_bg else None))
                    if show_n_top_bar:
                        ax.text(x, y_max_val + y_range * max(0.0, n_top_offset_bar),
                                f"(n={nval})", ha="center", va="bottom",
                                fontsize=n_fontsize, fontweight=("bold" if n_bold else "normal"),
                                color=n_top_color_bar, alpha=n_top_alpha_bar, clip_on=False)
                        needed_top = max(needed_top, y_max_val + y_range * (n_top_offset_bar + 0.01))
                    if show_n_bottom_bar:
                        # --- 修改後：將 (n=xx) 顯示在 X 軸標籤下方 ---
                        ax.text(
                            x, -0.01 - n_bottom_offset_bar, f"(n={nval})",
                            transform=ax.get_xaxis_transform(),  # X 用 data, Y 用 axes
                            ha="center", va="top",
                            fontsize=n_fontsize,
                            fontweight=("bold" if n_bold else "normal"),
                            color=n_bottom_color_bar, alpha=n_bottom_alpha_bar,
                            clip_on=False)

            if bottom_n_summary != "Off":
                if bottom_n_summary == "By X-level":
                    items = [f"{label_map.get(str(ix), str(ix))}: n={int(counts[ix])}" for ix in means.index]
                else:
                    total_n = int(df[y_col].dropna().shape[0])
                    items = [f"Total n={total_n}"]
                y_pos = _draw_bottom_center_summary(ax, " ; ".join(items), y_min, y_max, bottom_n_offset, bottom_n_font)
                if (y_pos is not None):
                    needed_bottom = min(needed_bottom, y_pos - (y_max - y_min)*0.02)

        else:
            means = grouped.mean().unstack(group_col)
            ses = grouped.apply(sem).unstack(group_col)
            counts = df.groupby([x_col, group_col])[y_col].count().unstack(group_col)
            x_levels = list(means.index)
            g_levels = list(means.columns)
            n_groups = len(g_levels)
            bar_width = max(0.02, bar_width_inch / max(0.1, fig_width))
            spacing = max(0.0, bar_spacing_inch / max(0.1, fig_width))
            xpos = np.arange(len(x_levels))
            total_width = n_groups*bar_width + (n_groups-1)*spacing
            offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_groups)

            recs = []
            for xv in x_levels:
                for g in g_levels:
                    recs.append({
                        x_col: str(xv),
                        str(group_col): str(g),
                        "Mean": float(means.loc[xv, g]) if pd.notna(means.loc[xv, g]) else np.nan,
                        "SE": float(ses.loc[xv, g]) if pd.notna(ses.loc[xv, g]) else np.nan,
                        "N": int(counts.loc[xv, g]) if (xv in counts.index and g in counts.columns and pd.notna(counts.loc[xv, g])) else 0
                    })
            summary_table = pd.DataFrame.from_records(recs)

            for i, g in enumerate(g_levels):
                for xi, xv in enumerate(x_levels):
                    base_color = group_colors.get(str(g), "#1f77b4")
                    color = x_colors.get(str(xv), base_color) if override_by_x else base_color
                    val = float(means.loc[xv, g]) if pd.notna(means.loc[xv, g]) else np.nan
                    se = float(ses.loc[xv, g]) if pd.notna(ses.loc[xv, g]) else np.nan
                    safe_yerr = None if (not show_se or (not (se == se) or se <= 0)) else se

                    center = xpos[xi]+offsets[i]
                    ax.bar(center, val, yerr=safe_yerr, width=bar_width,
                           edgecolor=bar_edgecolor, linewidth=bar_linewidth,
                           capsize=err_capsize, ecolor=bar_edgecolor,
                           error_kw=dict(lw=err_linewidth),
                           label=str(group_labels.get(g, g)) if xi == 0 else "", color=color)

                    bar_centers_map[(str(xv), str(g))] = float(center)
                    top = (val if np.isfinite(val) else 0.0) + (safe_yerr if safe_yerr else 0.0)
                    bar_tops_map[(str(xv), str(g))] = top

                    if bar_show_points:
                        vals = df.loc[(df[x_col].astype(str) == str(xv)) & (df[group_col].astype(str) == str(g)), y_col].dropna().to_numpy()
                        if len(vals) > 0:
                            rng = np.random.default_rng(hash((str(xv), str(g))) % (2**32-1))
                            jit = (rng.random(len(vals)) - 0.5) * 2 * bar_point_jitter
                            ax.scatter(center + jit, vals, s=bar_point_size, alpha=bar_point_alpha, color=color, edgecolors="k", linewidths=0.3, zorder=3)

                    if show_value_labels and np.isfinite(val):
                        ax.text(center, top + (y_max - y_min)*0.01, f"{val:.{int(value_decimals)}f}",
                                ha="center", va="bottom", fontsize=value_fontsize,
                                fontweight=("bold" if value_bold else "normal"))

            xticks = [label_map.get(str(l), str(l)) for l in x_levels]
            ax.set_xticks(xpos); ax.set_xticklabels(xticks, rotation=x_tick_rotation, fontsize=x_tick_fontsize)

            handles, labels = ax.get_legend_handles_labels()
            if legend_custom_order:
                order_map = {lab: i for i, lab in enumerate(legend_custom_order)}
                order_idx = sorted(range(len(labels)), key=lambda k: (order_map.get(labels[k], 10**9), k))
                handles = [handles[i] for i in order_idx]
                labels = [labels[i] for i in order_idx]
            if labels:
                legend_title = f"{group_col}" + (" (colors by X-level)" if override_by_x else "")
                leg = ax.legend(handles, labels, title=legend_title,
                                fontsize=legend_fontsize, title_fontsize=legend_title_fontsize)
                
                # ✅ 統一由共用函式控制外觀
                apply_legend_style(ax,
                                   transparent_bg=legend_bg_transparent,
                                   show_border=legend_show_border,
                                   border_color=legend_border_color,
                                   border_width=legend_border_width)

            # --- v0.9.31: (n=xx) Top/Bottom 同時顯示（Bar, grouped） ---
            if show_n_labels:
                y_min_val, y_max_val = ax.get_ylim()
                y_range = max(1e-12, (y_max_val - y_min_val))
                for xi, xv in enumerate(x_levels):
                    for gi, g in enumerate(g_levels):
                        center = xpos[xi]+offsets[gi]
                        nval = int(counts.loc[xv, g]) if (xv in counts.index and g in counts.columns and pd.notna(counts.loc[xv, g])) else 0
                        val = float(means.loc[xv, g]) if pd.notna(means.loc[xv, g]) else np.nan
                        if show_n_inside and np.isfinite(val) and val > 0:
                            y_text = val *  max(0.0, n_inside_offset)
                            _safe_text(ax, center, y_text, f"(n={int(nval)})",
                                       ha="center", va="bottom",
                                       fontweight=("bold" if n_bold else "normal"),
                                       fontsize=n_fontsize,
                                       color=group_colors.get(str(g), "#1f77b4") if n_inside_bg else "#000000",
                                       bbox=(dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none') if n_inside_bg else None))
                        if show_n_top_bar:
                            ax.text(center, y_max_val + y_range * max(0.0, n_top_offset_bar),
                                    f"(n={int(nval)})", ha="center", va="bottom",
                                    fontsize=n_fontsize, fontweight=("bold" if n_bold else "normal"),
                                    color=n_top_color_bar, alpha=n_top_alpha_bar, clip_on=False)
                            needed_top = max(needed_top, y_max_val + y_range * (n_top_offset_bar + 0.01))
                        if show_n_bottom_bar:
                            y_bottom_text = y_min_val - y_range * max(0.0, n_bottom_offset_bar)
                            ax.text(center, y_bottom_text, f"(n={int(nval)})",
                                    ha="center", va="top", fontsize=n_fontsize,
                                    fontweight=("bold" if n_bold else "normal"),
                                    color=n_bottom_color_bar, alpha=n_bottom_alpha_bar, clip_on=False)
                            needed_bottom = min(needed_bottom, y_bottom_text - y_range*0.02)

            if bottom_n_summary != "Off":
                if bottom_n_summary == "By X-level":
                    items = [f"{label_map.get(str(xv), str(xv))}: n={int(counts.loc[xv, g_levels].sum())}" for xv in x_levels]
                else:
                    items = [f"{group_labels.get(str(g), str(g))}: n={int(counts[g].sum())}" for g in g_levels]
                y_pos = _draw_bottom_center_summary(ax, " ; ".join(items), y_min, y_max, bottom_n_offset, bottom_n_font)
                if (y_pos is not None):
                    needed_bottom = min(needed_bottom, y_pos - (y_max - y_min)*0.02)

    # --- BOX ---
    elif plot_type.startswith("Box"):
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            if group_col in (None, "None"):
                sns.boxplot(
                    data=df, x=x_col, y=y_col, ax=ax,
                    fliersize=0, widths=box_width,
                    boxprops=dict(facecolor=box_fill_color, alpha=box_alpha, edgecolor=box_edge_color, linewidth=box_edge_lw),
                    whiskerprops=dict(color=box_edge_color, linewidth=box_edge_lw),
                    capprops=dict(color=box_edge_color, linewidth=box_edge_lw),
                    medianprops=dict(color=box_edge_color, linewidth=box_edge_lw)
                )
                if show_points:
                    sns.stripplot(
                        data=df, x=x_col, y=y_col, ax=ax,
                        color=group_colors.get("default", box_edge_color),
                        size=point_size / 10, alpha=point_alpha, jitter=jitter_width
                    )
            else:
                palette = {str(k): v for k, v in group_colors.items() if k is not None}
                sns.boxplot(
                    data=df, x=x_col, y=y_col, hue=group_col, ax=ax,
                    palette=palette if palette else None,
                    fliersize=0, widths=box_width,
                    boxprops=dict(facecolor=box_fill_color, alpha=box_alpha, edgecolor=box_edge_color, linewidth=box_edge_lw),
                    whiskerprops=dict(color=box_edge_color, linewidth=box_edge_lw),
                    capprops=dict(color=box_edge_color, linewidth=box_edge_lw),
                    medianprops=dict(color=box_edge_color, linewidth=box_edge_lw)
                )
                if show_points:
                    sns.stripplot(
                        data=df, x=x_col, y=y_col, hue=group_col, ax=ax,
                        dodge=True, palette=palette if palette else None,
                        size=point_size / 10, alpha=point_alpha,
                        jitter=jitter_width, legend=False
                    )
                if ax.get_legend() is None and getattr(ax, "legend_", None) is None:
                    if legend_loc != "none":
                        ax.legend(title=str(group_col), loc=legend_loc, frameon=False)
                else:
                    leg = ax.get_legend()
                    if legend_loc != "none":
                        leg.set_bbox_to_anchor(None)
                        leg.set_loc(legend_loc)
                        leg.set_frame_on(False)

                
                # ✅ 統一控制 legend 透明背景與邊框
                apply_legend_style(
                    ax,
                    transparent_bg=legend_bg_transparent,
                    show_border=legend_show_border,
                    border_color=legend_border_color,
                    border_width=legend_border_width
                )

            try:
                for patch in ax.artists:
                    if hasattr(patch, "set_facecolor"):
                        patch.set_facecolor(box_fill_color)
                        patch.set_alpha(box_alpha)
                        patch.set_edgecolor(box_edge_color)
                        patch.set_linewidth(box_edge_lw)
            except Exception:
                pass
            try:
                for line in ax.lines:
                    line.set_color(box_edge_color)
                    line.set_linewidth(box_edge_lw)
            except Exception:
                pass

            # --- v0.9.31 修正版: Box (n=xx) 上/下 同時顯示 ---
            if box_show_n:
                try:
                    positions, x_levels, n_list = _compute_category_counts(df, x_col)
                    for x_pos, xv, nval in zip(positions, x_levels, n_list):
                        # 上方樣本數：仍以資料座標為基準 (受 y 軸影響)
                        if box_show_top:
                            y_min_val, y_max_val = ax.get_ylim()
                            y_range = max(1e-12, (y_max_val - y_min_val))
                            y_text = y_max_val + y_range * box_top_offset_rel
                            ax.text(
                                x_pos, y_text, f"(n={nval})",
                                ha="center", va="bottom",
                                fontsize=int(box_n_fontsize),
                                color=box_top_color, alpha=box_top_alpha,
                                clip_on=False
                            )
            
                        # 下方樣本數：固定在 X 軸標籤下方，不受 Y 軸變化影響
                        if box_show_bottom:
                            ax.text(
                                x_pos, -0.12 - box_bottom_offset_rel, f"(n={nval})",
                                transform=ax.get_xaxis_transform(),  # ✅ 關鍵: 鎖定 X 軸座標系統
                                ha="center", va="top",
                                fontsize=int(box_n_fontsize),
                                color=box_bottom_color, alpha=box_bottom_alpha,
                                clip_on=False
                            )
            
                    # 自動延伸上方 y 軸，防止上方標籤被裁切
                    y_min_val, y_max_val = ax.get_ylim()
                    y_range = max(1e-12, (y_max_val - y_min_val))
                    extra = max(box_top_offset_rel, box_bottom_offset_rel) * 1.5
                    ax.set_ylim(y_min_val - y_range * extra, y_max_val + y_range * extra)
            
                except Exception as e:
                    warnings.warn(f"Box n-label draw failed: {e}")


        except Exception as e:
            warnings.warn(f"Seaborn not available or failed ({e}), falling back to matplotlib boxplot).")
            cats = list(pd.Index(df[x_col].dropna().astype(str).unique()))
            data = [df.loc[df[x_col].astype(str) == c, y_col].dropna().to_numpy() for c in cats]
            bp = ax.boxplot(data, positions=np.arange(len(cats)),
                            widths=box_width, showfliers=False, patch_artist=True)
            for b in bp.get('boxes', []):
                b.set_facecolor(box_fill_color); b.set_alpha(box_alpha)
                b.set_edgecolor(box_edge_color); b.set_linewidth(box_edge_lw)
            for k in ('whiskers', 'caps', 'medians'):
                for ln in bp.get(k, []):
                    ln.set_color(box_edge_color); ln.set_linewidth(box_edge_lw)
            try:
                sync_whisker_caps_to_boxwidth(bp, box_width=box_width, cap_lw=box_edge_lw, cap_color=box_edge_color)
            except Exception as e:
                warnings.warn(f"Whisker cap sync skipped (mpl): {e}")

            xticks = [label_map.get(str(l), str(l)) for l in cats]
            ax.set_xticks(np.arange(len(cats)))
            ax.set_xticklabels(xticks, rotation=x_tick_rotation, fontsize=x_tick_fontsize)

            if show_points:
                for i, c in enumerate(cats):
                    vals = df.loc[df[x_col].astype(str) == c, y_col].dropna().to_numpy()
                    if len(vals) == 0: continue
                    rng = np.random.default_rng(i+123)
                    jit = (rng.random(len(vals))-0.5)*2*jitter_width
                    ax.scatter(i + jit, vals, s=point_size, alpha=point_alpha,
                               color=group_colors.get("default", "#1f77b4"),
                               edgecolors="k", linewidths=0.3)

        gb_cols = [x_col] + ([] if group_col in (None, "None") else [group_col])
        summary_table = (df.groupby(gb_cols)[y_col]
                           .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
                           .reset_index())
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)

        # X tick vertical offset (visual n-label spacing)
        try:
            if plot_type.startswith("Box") and box_xtick_offset_rel > 0:
                y_min_val, y_max_val = ax.get_ylim()
                y_range = max(1e-12, (y_max_val - y_min_val))
                offset_val = y_range * box_xtick_offset_rel
                for label in ax.get_xticklabels():
                    pos = label.get_position()
                    label.set_position((pos[0], pos[1] - offset_val / y_range))
        except Exception as e:
            warnings.warn(f"Box X tick reposition failed: {e}")

    # --- SCATTER ---
    else:
        if group_col in (None, "None"):
            color = group_colors.get("default", "#1f77b4")
            _x = pd.to_numeric(df[x_col], errors="coerce")
            _y = pd.to_numeric(df[y_col], errors="coerce")
            _mask = _x.notna() & _y.notna()
            if _mask.sum() == 0:
                st.warning(f"No numeric data to plot for X='{x_col}', Y='{y_col}'.")
            else:
                ax.scatter(_x[_mask], _y[_mask], s=point_size, alpha=point_alpha, color=color, marker="o")
            if add_reg:
                coef = polyfit_regression(_x.to_numpy(), _y.to_numpy())
                if coef:
                    b, a, r2 = coef
                    xs = np.linspace(df[x_col].min(), df[x_col].max(), 200)
                    ax.plot(xs, a+b*xs, color=color, linewidth=2)
                    label = _format_eq(b, a, r2, show_r2, show_eq)
                    if label:
                        leg = ax.legend([label], fontsize=legend_fontsize)
                        
        else:
            legend_entries = []
            from matplotlib.lines import Line2D
            for gk, gdf in df.groupby(group_col):
                gk = str(gk)
                color = group_colors.get(gk, "#1f77b4")
                shape = group_shapes.get(gk, "o")[0]
                disp = group_labels.get(gk, gk)
                _x = pd.to_numeric(gdf[x_col], errors="coerce")
                _y = pd.to_numeric(gdf[y_col], errors="coerce")
                _mask = _x.notna() & _y.notna()
                if _mask.sum() > 0:
                    ax.scatter(_x[_mask], _y[_mask], s=point_size, alpha=point_alpha, color=color, marker=shape)
                if add_reg:
                    coef = polyfit_regression(gdf[x_col].to_numpy(), gdf[y_col].to_numpy())
                    if coef:
                        b, a, r2 = coef
                        xs = np.linspace(gdf[x_col].min(), gdf[x_col].max(), 200)
                        ax.plot(xs, a+b*xs, color=color, linewidth=2, alpha=0.9)
                        label = f"{disp} (n={len(gdf)})"
                        extra = _format_eq(b, a, r2, show_r2, show_eq)
                        if extra: label = f"{label}, {extra}"
                    else:
                        label = f"{disp} (n={len(gdf)})"
                else:
                    label = f"{disp} (n={len(gdf)})"
                legend_entries.append((label, color, shape))
            if legend_entries:
                if legend_custom_order:
                    order_map = {lab: i for i, lab in enumerate(legend_custom_order)}
                    legend_entries.sort(key=lambda t: (order_map.get(t[0].split(" (n=")[0], 10**9)))
                handles = [Line2D([0], [0], marker=m, color='w', label=l,
                                  markerfacecolor=c, markeredgecolor='k', markersize=8)
                           for l, c, m in legend_entries]
                leg = ax.legend(handles=handles, title=str(group_col),
                                fontsize=legend_fontsize, title_fontsize=legend_title_fontsize)
                
                apply_legend_style(ax,
                                   transparent_bg=legend_bg_transparent,
                                   show_border=legend_show_border,
                                   border_color=legend_border_color,
                                   border_width=legend_border_width)


    # Axis & stats (post drawing)
    ax.set_title(main_title, fontweight=fontweight)

    if plot_type.startswith("Scatter") and 'x_min' in locals():
        ax.set_xlim(x_min, x_max)
        xticks = np.arange(x_min, x_max + 1e-9, x_step if 'x_step' in locals() else 1.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.{int(x_dec)}f}" for x in xticks], fontsize=x_tick_fontsize, rotation=x_tick_rotation)

    line_y_values = []
    # --- Pairwise (Bar) fixed stacking rule ---
    if enable_sig and pair_count > 0 and plot_type.startswith("Bar") and (len(sig_pairs) > 0):
        def _get_center_top(xv, gv=None) -> Tuple[Optional[float], Optional[float]]:
            key = (str(xv), str(gv)) if gv is not None else str(xv)
            return bar_centers_map.get(key, None), bar_tops_map.get(key, None)
    
        y_range_user = max(1e-12, (y_max - y_min))
        extra_for_top_n = (float(n_top_offset_bar) + 0.01) if (show_n_labels and show_n_top_bar) else 0.0
    
        line_y_values = []  # ✅ 重新初始化，避免重複 append
        for idx, comp in enumerate(sig_pairs):
            c1, t1 = _get_center_top(comp["x1"], comp["g1"])
            c2, t2 = _get_center_top(comp["x2"], comp["g2"])
            if c1 is None or c2 is None or t1 is None or t2 is None:
                continue
            tallest = max(t1, t2)
            # ✅ 階梯式上移，防止重疊
            y_line = tallest + y_range_user * (sig_line_lift + extra_for_top_n + idx * sig_stack_gap)
            line_y_values.append((c1, c2, y_line))
            needed_top = max(needed_top, y_line + y_range_user * (sig_star_extra_offset + 0.06))
    
    # --- Draw pairwise significance lines (Bar, single bracket, default transparent) ---
    if enable_sig and pair_count > 0 and plot_type.startswith("Bar") and len(sig_pairs) > 0:
        try:
            y_min, y_max = ax.get_ylim()
            y_range_user = max(1e-12, (y_max - y_min))
        except Exception:
            y_range_user = 1.0
    
        sig_line_height = locals().get("sig_line_height", 0.02)
        sig_star_extra_offset = locals().get("sig_star_extra_offset", 0.02)
        sig_line_width = locals().get("sig_line_width", 1.6)
        sig_star_fontsize = locals().get("sig_star_fontsize", 12)
        sig_star_color = locals().get("sig_star_color", "black")
        sig_line_color = locals().get("sig_line_color", "black")
        sig_line_alpha = float(locals().get("sig_line_alpha", 0.0))  # ← 預設透明
    
        for comp, entry in zip(sig_pairs, line_y_values):
            if entry is None or len(entry) != 3:
                continue
            c1, c2, y_line = entry
            try:
                y_top = y_line + y_range_user * sig_line_height
                ax.plot([c1, c1, c2, c2],
                        [y_line, y_top, y_top, y_line],
                        lw=float(sig_line_width),
                        color=_rgba(sig_line_color, sig_line_alpha),
                        clip_on=False, zorder=10)
                if comp.get("p_str"):
                    mid = (c1 + c2) / 2.0
                    ax.text(mid,
                            y_top + y_range_user * sig_star_extra_offset,
                            comp["p_str"],
                            ha="center", va="bottom",
                            fontsize=float(sig_star_fontsize),
                            color=sig_star_color,
                            clip_on=False, zorder=11)
            except Exception as e:
                warnings.warn(f"Bar pairwise draw failed for {comp}: {e}")



    # --- Pairwise (Box) – unified stair-gap stacking rule ---
    if enable_sig and pair_count > 0 and plot_type.startswith("Box") and (len(sig_pairs) > 0):
        x_levels = list(pd.Index(df[x_col].dropna().astype(str).unique()))
        x_index = {lvl: i for i, lvl in enumerate(x_levels)}
        box_line_triplets = []
        y_range_user = max(1e-12, (y_max - y_min))
    
        extra_for_top_n = (float(box_top_offset_rel) + 0.01) if (box_show_n and box_show_top) else 0.0
        line_counter = 0
    
        for comp in sig_pairs:
            x1 = str(comp.get("x1")); x2 = str(comp.get("x2"))
            if (x1 not in x_index) or (x2 not in x_index):
                box_line_triplets.append(None)
                continue
    
            # --- 取該組的最高點 ---
            try:
                t1 = df.loc[df[x_col].astype(str) == x1, y_col].max()
                t2 = df.loc[df[x_col].astype(str) == x2, y_col].max()
            except Exception:
                t1, t2 = y_max, y_max
    
            tallest = max(t1, t2) if (pd.notna(t1) and pd.notna(t2)) else y_max
            # --- 階梯式上移（與 Bar 相同演算法） ---
            y_line = tallest + y_range_user * (sig_line_lift + extra_for_top_n + line_counter * sig_stack_gap)
            box_line_triplets.append((float(x_index[x1]), float(x_index[x2]), float(y_line)))
            line_counter += 1
            needed_top = max(needed_top, y_line + y_range_user * (sig_star_extra_offset + 0.06))
    
        # --- Draw pairwise significance lines (Box, single bracket, default transparent) ---
        for comp, trip in zip(sig_pairs, box_line_triplets):
            if trip is None or len(trip) != 3:
                continue
            a, b, y_line = trip
            try:
                sig_line_height = locals().get("sig_line_height", 0.02)
                sig_star_extra_offset = locals().get("sig_star_extra_offset", 0.02)
                sig_line_width = locals().get("sig_line_width", 1.6)
                sig_star_fontsize = locals().get("sig_star_fontsize", 12)
                sig_star_color = locals().get("sig_star_color", "black")
                sig_line_color = locals().get("sig_line_color", "black")
                sig_line_alpha = float(locals().get("sig_line_alpha", 0.0))  # ← 預設透明
        
                y_top = y_line + y_range_user * sig_line_height
                ax.plot([a, a, b, b],
                        [y_line, y_top, y_top, y_line],
                        lw=float(sig_line_width),
                        color=_rgba(sig_line_color, sig_line_alpha),
                        clip_on=False, zorder=10)
        
                if comp.get("p_str"):
                    mid = (a + b) / 2.0
                    ax.text(mid,
                            y_top + y_range_user * sig_star_extra_offset,
                            comp["p_str"],
                            ha="center", va="bottom",
                            fontsize=float(sig_star_fontsize),
                            color=sig_star_color,
                            clip_on=False, zorder=11)
            except Exception as e:
                warnings.warn(f"Box pairwise draw failed for {comp}: {e}")




    y_min_plot = y_min
    y_max_plot = max(y_max, needed_top)
    if needed_bottom < y_min_plot:
        y_min_plot = needed_bottom
    if fix_y_intercept and (y_intercept_value is not None):
        y_min_plot = min(y_min_plot, y_intercept_value)
        y_max_plot = max(y_max_plot, y_intercept_value)
    if lock_nice_ticks:
        step = max(1e-12, y_step)
        y_min_aligned = math.floor(y_min_plot / step) * step
        y_max_aligned = math.ceil(y_max_plot / step) * step
        y_min_plot = min(y_min_aligned, y_min)
        y_max_plot = max(y_max_aligned, y_max)
    if not np.isfinite(y_min_plot) or not np.isfinite(y_max_plot) or y_max_plot <= y_min_plot:
        y_min_plot = y_min
        y_max_plot = y_max if y_max > y_min else (y_min + 1.0)

    ax.set_ylim(y_min_plot, y_max_plot)

    # Optional baseline at y=0
    try:
        if y0_line and np.isfinite(y_min_plot) and np.isfinite(y_max_plot) and (y_min_plot < 0 < y_max_plot):
            try:
                _lw = float(ax.spines.get("bottom", ax.spines["left"]).get_linewidth())
                if not np.isfinite(_lw) or _lw <= 0: _lw = 1.0
            except Exception:
                _lw = 1.0
            try:
                from matplotlib.colors import is_color_like as _is_color_like
                _color = y0_color if _is_color_like(y0_color) else "#000000"
            except Exception:
                _color = "#000000"
            ax.axhline(y=0.0, xmin=0, xmax=1, color=_color, linewidth=max(0.5, _lw * 0.8), alpha=0.7, zorder=2)
    except Exception as e:
        st.warning(f"Draw y=0 baseline skipped: {e}")

    try:
        if fix_y_intercept and (y_intercept_value is not None):
            ax.spines["bottom"].set_position(("data", float(y_intercept_value)))
        else:
            if abs(float(y_min)) < 1e-12:
                ax.spines["bottom"].set_position(("data", 0.0))
            else:
                if plot_type.startswith("Bar"):
                    ax.spines["bottom"].set_position(("outward", 0))
    except Exception as e:
        st.warning(f"Axis alignment adjustment skipped: {e}")

    if plot_type.startswith("Bar") and xpos is not None:
        n_cats = len(xpos)
        ax.set_xlim(-0.5, n_cats - 0.5)  # 原來用 inch margin，簡化到類別邊界視覺更穩定

    ax.set_xlabel(xlabel, labelpad=xlabel_pad, fontweight=fontweight)
    ax.set_ylabel(ylabel, fontweight=fontweight)

    # Draw pairwise lines (Bar)
    if enable_sig and pair_count > 0 and plot_type.startswith("Bar") and (len(sig_pairs) > 0):
        for idx, comp in enumerate(sig_pairs):
            entry = None if idx >= len(line_y_values) else line_y_values[idx]
            if entry is None: continue
            c1, c2, y_line = entry
            stars_raw = p_to_stars(comp.get("p", ""), thr_1, thr_2, thr_3, show_ns=(not hide_ns))
            if hide_ns and (stars_raw == "ns") and (isinstance(comp.get("p", ""), str) and comp.get("p", "").strip().lower() != "ns"):
                continue
            stars = stars_raw if stars_raw else str(comp.get("p", "")).strip()
            color = comp.get("color", sig_line_color_default) or sig_line_color_default
            ax.plot([c1, c2], [y_line, y_line], color=color, lw= sig_line_width)
            y_rng = ax.get_ylim()[1] - ax.get_ylim()[0]
            tick = y_rng * sig_tick_length
            ax.plot([c1, c1], [y_line, y_line - tick], color=color, lw=sig_line_width)
            ax.plot([c2, c2], [y_line, y_line - tick], color=color, lw=sig_line_width)
            ax.text((c1+c2)/2.0, y_line + y_rng*sig_star_extra_offset, stars,
                    ha="center", va="bottom", color=color,
                    fontsize=sig_star_font, fontweight=("bold" if sig_star_bold else "normal"))

    # Pairwise (Box) lines
    if enable_sig and pair_count > 0 and plot_type.startswith("Box") and (len(sig_pairs) > 0):
        x_levels = list(pd.Index(df[x_col].dropna().astype(str).unique()))
        x_index = {lvl: i for i, lvl in enumerate(x_levels)}
        y_rng_full = ax.get_ylim()[1] - ax.get_ylim()[0]
        pos_pairs, bases, stars, colors = [], [], [], []
        for comp in sig_pairs:
            x1 = str(comp.get("x1")); x2 = str(comp.get("x2"))
            if (x1 not in x_index) or (x2 not in x_index): continue
            pos_pairs.append((float(x_index[x1]), float(x_index[x2])))
            t1 = df.loc[df[x_col].astype(str) == x1, y_col].max()
            t2 = df.loc[df[x_col].astype(str) == x2, y_col].max()
            tallest = max(t1, t2) if (pd.notna(t1) and pd.notna(t2)) else ax.get_ylim()[1]
            bases.append(float(tallest) + y_rng_full * sig_line_lift)
            stars.append(p_to_stars(comp.get("p", ""), thr_1, thr_2, thr_3, show_ns=(not hide_ns)))
            colors.append(comp.get("color", sig_line_color_default) or sig_line_color_default)
        if pos_pairs:
            layers = _assign_layers_for_pairs_box(pos_pairs)
            step = y_rng_full * sig_stack_gap
            for (i, (x1, x2)) in enumerate(pos_pairs):
                y_line = bases[i] + layers[i] * step
                s_txt = stars[i]
                if hide_ns and s_txt == "ns": s_txt = ""
                color = colors[i]
                ax.plot([x1, x2], [y_line, y_line], color=color, lw=sig_line_width)
                tick = y_rng_full * sig_tick_length
                ax.plot([x1, x1], [y_line, y_line - tick], color=color, lw=sig_line_width)
                ax.plot([x2, x2], [y_line, y_line - tick], color=color, lw=sig_line_width)
                if s_txt:
                    ax.text((x1+x2)/2.0, y_line + y_rng_full*sig_star_extra_offset, s_txt,
                            ha="center", va="bottom", color=color,
                            fontsize=sig_star_font, fontweight=("bold" if sig_star_bold else "normal"))

    # yticks
    try:
        yticks = np.arange(y_min_plot, y_max_plot + 1e-9, max(1e-12, y_step))
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.{int(y_dec)}f}" for y in yticks], fontsize=y_tick_fontsize)
    except Exception:
        pass

    ax.grid(axis='x', which='both', alpha=0.3 if grid_x else 0.0, linewidth=grid_linewidth)
    ax.grid(axis='y', which='both', alpha=0.3 if grid_y else 0.0, linewidth=grid_linewidth)

    st.markdown('</div>', unsafe_allow_html=True)

# --- 🧩 安全可輸入版 colC 模組 ---
with colC:
    # ✅ 移除 HTML <div>，改用安全 CSS + Streamlit 原生容器
    st.markdown("""
    <style>
    .safe-scroll {
        max-height: 90vh;
        overflow-y: auto;
        padding-right: 0.8rem;
        background-color: #f9f9fb;
        border-radius: 8px;
        box-shadow: inset 0 0 6px rgba(0,0,0,0.05);
        position: relative;   /* 保留互動層 */
        z-index: 0;           /* 防止遮擋 */
    }
    </style>
    """, unsafe_allow_html=True)

    # ✅ 這裡用 markdown 包裝樣式，但不再用不安全 div 包住所有 widget
    st.subheader("6) Plot Preview & Export")

    # --- 自動為 X 軸下方的 (n=xx) 保留空間 ---
    try:
        extra_bottom = 0.14 + float(n_bottom_offset_bar)
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, extra_bottom))
    except NameError:
        pass

    st.pyplot(fig, clear_figure=False)
    # --- build PNG buffer for enlarged preview ---
    import io
    _png_buf = io.BytesIO()
    try:
        # 確保圖已渲染，再存成 PNG
        fig.canvas.draw()
        fig.savefig(_png_buf, format="png", dpi=200, bbox_inches="tight")
        _png_bytes = _png_buf.getvalue()
    
        with st.expander("🔎 放大預覽（非全螢幕）", expanded=False):
            st.image(_png_bytes, caption="Large preview", use_container_width=True)
    except Exception as e:
        st.warning(f"預覽圖片產生失敗：{e}")


    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"{sanitize_filename(ylabel)}_{sanitize_filename(xlabel)}"
    if group_col not in (None, "None"):
        base_name += f"_{sanitize_filename(str(group_col))}"
    file_pdf = f"{base_name}_{ts}.pdf"
    file_png = f"{base_name}_{ts}.png"
    file_svg = f"{base_name}_{ts}.svg"
    file_csv = f"{base_name}_{ts}_summary.csv"

    # --- Export Style ---
    st.sidebar.markdown("---")
    enable_pub_style = st.checkbox("📰 Export style preset: Publication-ready", value=True)
    apply_publication_style(enable_pub_style)

    style_preset = st.selectbox(
        "🎨 Style preset",
        ["Default", "Publication-ready", "Presentation"],
        index=1,
        help="Apply predefined aesthetic preset; won’t override your manual tweaks."
    )

    _okabe_ito = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    if style_preset in ["Publication-ready", "Presentation"]:
        plt.rcParams["axes.prop_cycle"] = _cycler(color=_okabe_ito)
        try:
            plt.rcParams["font.family"] = "Helvetica"
        except Exception:
            pass
        plt.rcParams["axes.linewidth"] = 1.5 if style_preset == "Publication-ready" else 1.2
        plt.rcParams["grid.linewidth"] = 0.4 if style_preset == "Publication-ready" else 0.6
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["figure.dpi"] = 300 if style_preset == "Publication-ready" else 200
        st.sidebar.caption("Preset applied: Okabe–Ito palette + clean axes/grid.")

    x_label_rotation = st.slider("X-axis label rotation (°)", min_value=0, max_value=90, value=0, step=5)
    for _ax in fig.axes:
        apply_x_label_rotation(_ax, x_label_rotation)

    # --- Export PDF/PNG/SVG ---
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', transparent=True)
    pdf_buf.seek(0)
    st.download_button("📄 Download PDF", data=pdf_buf, file_name=file_pdf, mime="application/pdf", use_container_width=True)

    png_dpi = st.slider("PNG export DPI", 72, 600, 300)
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=int(png_dpi), bbox_inches="tight", transparent=True)
    png_buf.seek(0)
    st.download_button("🖼️ Download PNG", data=png_buf, file_name=file_png, mime="image/png", use_container_width=True)

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight", transparent=True)
    svg_buf.seek(0)
    st.download_button("🧩 Download SVG", data=svg_buf, file_name=file_svg, mime="image/svg+xml", use_container_width=True)



    # --- Figure Caption Module v5 ---
    st.markdown("---")
    st.subheader("🖋️ Figure Caption Generator")
    
    caption_style = st.selectbox(
        "Journal style",
        ["eLife", "PNAS", "Nature", "Science", "Current Biology"],
        index=0,
        help="Select the journal tone for caption phrasing."
    )
    
    stat_indicator = st.radio(
        "Display error indicator as:",
        ["±SE", "±SEM", "±SD"],
        index=0,
        help="Choose how the variability measure should appear in captions."
    )
    
    auto_caption = st.checkbox("Auto-generate captions", value=True)
    
    # --- 安全名稱取用 ---
    def safe_name(col, rename_map, use_raw):
        try:
            if not use_raw and col in rename_map:
                return rename_map[col]
            return str(col)
        except Exception:
            return str(col)
    
    rename_map = locals().get("rename_map", {})
    x_raw, y_raw, g_raw = str(x_col), str(y_col), (str(group_col) if group_col not in (None, "None") else None)
    x_renamed = safe_name(x_col, rename_map, False)
    y_renamed = safe_name(y_col, rename_map, False)
    g_renamed = safe_name(group_col, rename_map, False) if group_col not in (None, "None") else None
    
    # --- 平均 ± 指標 ---
    def extract_mean_se(summary_table, x_col, indicator):
        try:
            rows = []
            if all(k in summary_table.columns for k in [x_col, "Mean", "SE"]):
                for _, row in summary_table.iterrows():
                    label = str(row[x_col])
                    val_m = row["Mean"]
                    val_s = row["SE"]
                    rows.append(f"{label}: {val_m:.2f} {indicator} {val_s:.2f}")
            return ", ".join(rows)
        except Exception:
            return ""
    
    # --- 抽取所有 p 值 (pairwise) ---
    def extract_pvalues(summary_table):
        try:
            p_cols = [c for c in summary_table.columns if c.lower().startswith("p")]
            if len(p_cols) == 0:
                return ""
            p_col = p_cols[0]
            if "Comparison" in summary_table.columns:
                pairs = []
                for _, row in summary_table.iterrows():
                    if pd.notna(row["Comparison"]) and pd.notna(row[p_col]):
                        try:
                            val = float(row[p_col])
                            p_str = "p<0.001" if val < 0.001 else f"p={val:.3f}"
                        except Exception:
                            p_str = f"p={row[p_col]}"
                        pairs.append(f"{row['Comparison']}: {p_str}")
                return "; ".join(pairs)
            else:
                vals = []
                for _, row in summary_table.iterrows():
                    try:
                        val = float(row[p_col])
                        vals.append("p<0.001" if val < 0.001 else f"p={val:.3f}")
                    except Exception:
                        vals.append(f"p={row[p_col]}")
                return ", ".join(vals)
        except Exception:
            return ""
    
    # --- N info ---
    def extract_ninfo(summary_table, x_col):
        try:
            if "N" not in summary_table.columns:
                return ""
            n_each = [f"{str(row[x_col])}: n={int(row['N'])}" for _, row in summary_table.iterrows()]
            return "(" + ", ".join(n_each) + ")"
        except Exception:
            return ""
    
    # --- Caption 組合模板 ---
    def build_caption(xname, yname, gname, meanse, ptext, ntext, style, indicator):
        s1 = f"{yname} across {xname}" + (f" grouped by {gname}." if gname else ".")
        s2 = f"Data represent mean {indicator} ({meanse})" if meanse else f"Data represent mean {indicator}."
        if ptext:
            s2 += f"; pairwise tests: {ptext}."
        s3 = f" {ntext}" if ntext else ""
        if style == "eLife":
            return f"{s1} {s2}{s3}"
        elif style == "PNAS":
            return f"{s1} Means {indicator} are shown ({meanse}); {ptext}. {ntext}"
        elif style == "Nature":
            return f"{s1} Points indicate group means {indicator}; {ptext}. {ntext}"
        elif style == "Science":
            return f"{s1} Values are shown as mean {indicator} ({meanse}); {ptext}. {ntext}"
        else:  # Current Biology
            return f"{s1} Plotted values are mean {indicator} ({meanse}); {ptext}. {ntext}"
    
    # --- 自動組合 ---
    if auto_caption:
        if 'summary_table' in locals() and summary_table is not None and not summary_table.empty:
            meanse_raw = extract_mean_se(summary_table, x_raw, stat_indicator)
            meanse_renamed = extract_mean_se(summary_table, x_renamed, stat_indicator)
            p_text = extract_pvalues(summary_table)
            n_info = extract_ninfo(summary_table, x_col)
        else:
            # --- 若無 summary_table（如 Scatter）則設為空 ---
            meanse_raw = meanse_renamed = p_text = n_info = ""
    
        # 即使沒有統計結果，也生成 caption
        caption_raw = build_caption(x_raw, y_raw, g_raw, meanse_raw, p_text, n_info, caption_style, stat_indicator)
        caption_renamed = build_caption(x_renamed, y_renamed, g_renamed, meanse_renamed, p_text, n_info, caption_style, stat_indicator)
    else:
        caption_raw, caption_renamed = "", ""

    
    # ---- 1️⃣ Raw Caption ----
    st.markdown("#### 1️⃣ Caption using raw column names")
    caption_raw = st.text_area("Raw caption", caption_raw, height=150)
    st.markdown(f"<div style='padding:.8rem;border:1px solid #ccc;border-radius:6px;'>{caption_raw}</div>", unsafe_allow_html=True)
    
    # ---- 2️⃣ Renamed Caption ----
    st.markdown("#### 2️⃣ Caption using renamed column names")
    caption_renamed = st.text_area("Renamed caption (auto-generated)", caption_renamed, height=150)
    st.markdown(f"<div style='padding:.8rem;border:1px solid #ccc;border-radius:6px;'>{caption_renamed}</div>", unsafe_allow_html=True)
    
   # ---- 3️⃣ Custom editable caption ----
    with st.container(border=True):
        st.markdown("#### 3️⃣ Custom editable caption (manual input)")
    
        caption_custom = st.text_area(
            "Custom caption (editable, write freely here)",
            value="",
            height=180,
            placeholder=f"Write your caption using renamed fields (e.g., {y_renamed} across {x_renamed} ...)",
            key="caption_custom_textarea",
            label_visibility="visible"
        )
    
        # 將輸入即時寫入 session_state，供 ZIP 匯出使用
        st.session_state["caption_custom"] = caption_custom
    
        if caption_custom.strip():
            st.markdown(
                f"<div style='padding:.8rem;border:1px solid #ccc;border-radius:6px;background:#fcfcfd;'>{caption_custom}</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("✏️ Start typing your custom caption above...")
    
    st.caption("All three captions will be included in the ZIP export as separate files.")
    
    # --- ZIP 匯出 ---
    file_caption_raw = f"{base_name}_{ts}_caption_raw.txt"
    file_caption_renamed = f"{base_name}_{ts}_caption_renamed.txt"
    file_caption_custom = f"{base_name}_{ts}_caption_custom.txt"
    
    try:
        import zipfile as _zipfile
        all_zip_buf = io.BytesIO()
        with _zipfile.ZipFile(all_zip_buf, mode="w", compression=_zipfile.ZIP_DEFLATED) as zf:
            if 'pdf_buf' in locals() and pdf_buf is not None:
                zf.writestr(file_pdf, pdf_buf.getvalue())
            if 'png_buf' in locals() and png_buf is not None:
                zf.writestr(file_png, png_buf.getvalue())
            if 'svg_buf' in locals() and svg_buf is not None:
                zf.writestr(file_svg, svg_buf.getvalue())
            if 'summary_table' in locals() and summary_table is not None:
                _csv_io = io.StringIO()
                _exp_tbl = summary_table.copy()
                _exp_tbl.to_csv(_csv_io, index=False)
                zf.writestr(file_csv, _csv_io.getvalue())
            # 匯入 caption 三種版本
            zf.writestr(file_caption_raw, caption_raw or "")
            zf.writestr(file_caption_renamed, caption_renamed or "")
            # 這裡改抓 session_state 的內容（確保即時更新）
            zf.writestr(file_caption_custom, st.session_state.get("caption_custom", ""))
        all_zip_buf.seek(0)
        st.download_button(
            "📦 Download ALL (ZIP)",
            data=all_zip_buf,
            file_name=f"{base_name}_{ts}_ALL.zip",
            mime="application/zip",
            use_container_width=True
        )
    except Exception as _e:
        st.caption(f"ZIP export skipped: {_e}")





    st.markdown("---")
    st.caption("v0.9.34 • Added figure caption generator (eLife default, +PNAS/Nature/Science/Current Biology styles).")
    st.markdown('</div>', unsafe_allow_html=True)
