# J-type-like Web Viz v0.10.1 — workflow UI refactor of supplied v0.9.36
# Run: streamlit run jtype_web_viz_v0_10_1.py
# Python >= 3.12; tested dependency versions are in requirements.txt.
from __future__ import annotations
import io
import json
import math
import re
import sys
import platform
import hashlib
import threading
import warnings
import zipfile
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cycler as _cycler
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = '0.10.1'
TEMPLATE_VERSION = 2
TEMPLATE_FILE_EXT = 'jviz_template.json'
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#000000', '#F0E442']


def safe_ticks(low, high, step):
    if not all(np.isfinite(v) for v in (low, high, step)) or step <= 0 or high <= low:
        raise ValueError('Axis bounds must be finite and increasing; tick step must be positive.')
    if (high - low) / step > 1000:
        raise ValueError('More than 1,000 ticks requested. Increase the tick step.')
    return np.arange(low, high + step * 1e-8, step)


def significance_label(value, thr1, thr2, thr3, hide_ns):
    value = str(value).strip()
    if not value:
        return ''
    if value.lower() == 'ns':
        return '' if hide_ns else 'ns'
    try:
        p = float(value)
    except ValueError:
        raise ValueError("Pairwise p values must be a number between 0 and 1, or 'ns'.")
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Pairwise p values must be between 0 and 1.')
    return p_to_stars(p, thr1, thr2, thr3, show_ns=not hide_ns)


class Controls:
    """Stable widget keys plus a separate durable settings store.

    Streamlit removes state for unrendered widgets. The mirror preserves settings
    when changing plot family; Basic/Advanced changes disclosure, never values.
    """
    def __init__(self):
        self.settings = dict(st.session_state.get('_settings', {}))

    def __getattr__(self, kind):
        def widget(label, *args, key, **kwargs):
            candidate = st.session_state.get(key, self.settings.get(key))
            if candidate is not None:
                valid = True
                if kind in ('selectbox', 'radio'):
                    options = list(kwargs.get('options', args[0] if args else []))
                    valid = candidate in options
                elif kind == 'checkbox':
                    valid = isinstance(candidate, bool)
                elif kind in ('text_input', 'text_area', 'color_picker'):
                    valid = isinstance(candidate, str)
                    if kind == 'color_picker':
                        valid = valid and bool(re.fullmatch(r'#[0-9a-fA-F]{6}', candidate))
                elif kind in ('slider', 'number_input'):
                    lo = kwargs.get('min_value', args[0] if len(args) > 0 else None)
                    hi = kwargs.get('max_value', args[1] if len(args) > 1 else None)
                    default = kwargs.get('value', args[2] if len(args) > 2 else None)
                    valid = isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and np.isfinite(candidate)
                    valid = valid and (lo is None or candidate >= lo) and (hi is None or candidate <= hi)
                    if isinstance(default, int) and not isinstance(default, bool):
                        valid = valid and isinstance(candidate, int)
                    elif isinstance(default, float) and valid:
                        candidate = float(candidate)
                if not valid:
                    st.session_state.pop(key, None)
                    self.settings.pop(key, None)
                    st.caption(f'Reset incompatible setting: {label}')
                elif key not in st.session_state:
                    # Restore through the widget default, avoiding two competing
                    # sources (explicit default and Session State API).
                    positional = list(args)
                    if kind in ('selectbox', 'radio'):
                        kwargs['index'] = options.index(candidate)
                    else:
                        position = 2 if kind in ('slider', 'number_input') else 0
                        if len(positional) > position:
                            positional[position] = candidate
                            kwargs.pop('value', None)
                        else:
                            kwargs['value'] = candidate
                    args = tuple(positional)
            value = getattr(st, kind)(label, *args, key=key, **kwargs)
            self.settings[key] = value
            st.session_state['_settings'] = self.settings
            return value
        return widget


def template_payload(settings, mode='Full', scope='Full plot'):
    values = dict(settings)
    if mode == 'Minimal':
        values = {k: v for k, v in values.items() if k in STYLE_KEYS}
    elif scope == 'Global style only':
        values = {k: v for k, v in values.items() if k in STYLE_KEYS}
    return {'template_version': TEMPLATE_VERSION, 'app': 'J-type-like Web Viz',
            'app_version': APP_VERSION, 'saved_at': datetime.now().isoformat(timespec='seconds'),
            'mode': mode, 'scope': scope, 'session_state': values}


def queue_template():
    """Callback: validate first, then apply atomically before widgets are created."""
    upload = st.session_state.get('tpl_uploader')
    if upload is None:
        return
    try:
        tpl = json.loads(upload.getvalue().decode('utf-8'))
        if not isinstance(tpl, dict) or tpl.get('template_version', 1) not in (1, 2):
            raise ValueError('Unsupported template format/version.')
        values = tpl.get('session_state')
        if not isinstance(values, dict):
            raise ValueError('Template must contain a session_state object.')
        allowed = {k: v for k, v in values.items() if k in CONTROL_KEYS or any(k.startswith(p) for p in DYNAMIC_PREFIXES)}
        if tpl.get('mode') == 'Minimal' or tpl.get('scope') == 'Global style only':
            allowed = {k: v for k, v in allowed.items() if k in STYLE_KEYS}
        json.dumps(allowed, allow_nan=False)
        settings = dict(st.session_state.get('_settings', {}))
        settings.update(allowed)
        st.session_state['_settings'] = settings
        for k, v in allowed.items():
            st.session_state.pop(k, None)
        st.session_state['_template_applied'] = True
        st.session_state['_template_notice'] = f'Applied {len(allowed)} saved settings. Legacy templates can restore only settings present in their JSON.'
    except (ValueError, TypeError, UnicodeError) as exc:
        st.session_state['_template_notice'] = f'Template not applied: {exc}'


@st.cache_resource
def render_lock():
    return threading.RLock()


def render_plot(df, settings):
    # rcParams and seaborn jitter use process-wide state: serialize and restore it.
    with render_lock(), plt.rc_context():
        random_state = np.random.get_state()
        before = set(plt.get_fignums())
        try:
            np.random.seed(2026)
            plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
            preset = settings['style_preset']
            apply_publication_style(preset == 'Publication-ready')
            if preset in ('Publication-ready', 'Presentation'):
                plt.rcParams['axes.prop_cycle'] = _cycler(color=PALETTE)
            plt.rcParams.update({'font.family': settings['font_family'], 'font.size': settings['font_size'], 'figure.dpi': 100})
            return _render_plot(df, settings)
        except Exception:
            for number in set(plt.get_fignums()) - before:
                plt.close(number)
            raise
        finally:
            np.random.set_state(random_state)


def figure_bytes(fig, fmt, dpi=300):
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches='tight', transparent=True,
                metadata={'Date': None} if fmt == 'svg' else None)
    return buffer.getvalue()


def make_caption(settings, summary, raw=False):
    reverse = {v: k for k, v in settings['rename_map'].items()}
    def name(v):
        return str(reverse.get(v, v) if raw else v)
    x, y, g = settings['x_col'], settings['y_col'], settings['group_col']
    journal = settings.get('caption_style', 'eLife')
    opening = {'eLife': 'Distribution of', 'PNAS': 'Comparison of', 'Nature': 'Variation in', 'Science': 'Observed', 'Current Biology': 'Analysis of'}[journal]
    sentence = f'{opening} {name(y)} across {name(x)}' + (f', grouped by {name(g)}.' if g else '.')
    family = settings['plot_type']
    if family.startswith('Bar'):
        sentence += ' Bars show means' + (' ± SE.' if settings['show_se'] else '; error bars are hidden.')
        if settings.get('bar_show_points'):
            sentence += ' Points show individual observations.'
    elif family.startswith('Box'):
        sentence += ' Boxes show the median and interquartile range; whiskers use the 1.5×IQR rule.'
        if settings.get('show_points'):
            sentence += ' Points show individual observations.'
    else:
        sentence += ' Points show individual observations.'
        if settings.get('add_reg'):
            sentence += ' Lines show ordinary least-squares fits.'
    if settings.get('enable_sig') and settings.get('sig_pairs'):
        ptext = '; '.join(f"{p['x1']} vs {p['x2']}: p={p['p']}" for p in settings['sig_pairs'] if str(p['p']).strip())
        if ptext:
            sentence += f' User-supplied pairwise annotations: {ptext}.'
    if summary is not None:
        count_key = 'N' if 'N' in summary else 'count'
        if count_key in summary:
            sentence += ' Sample sizes: ' + '; '.join(f"{row[x]}" + (f" / {row[g]}" if g else '') + f": n={int(row[count_key])}" for _, row in summary.iterrows()) + '.'
    return sentence


def export_panel(fig, summary, settings, ui, source_info):
    st.caption('PDF / SVG preserve editable vector text. PNG is raster; all exports use the preview figure.')
    png_dpi = ui.slider('PNG export DPI', 72, 600, 300, key='png_dpi')
    with st.expander('Figure captions', expanded=False):
        caption_style = ui.selectbox('Journal style', ['eLife', 'PNAS', 'Nature', 'Science', 'Current Biology'], key='caption_style')
        st.caption('Caption drafts describe the actual plot. Check journal wording and supplied statistics before submission.')
        auto_caption = ui.checkbox('Auto-generate captions', True, key='auto_caption')
        caption_settings = dict(settings, caption_style=caption_style)
        signature = hashlib.sha256(json.dumps(caption_settings, sort_keys=True, default=str).encode()).hexdigest()
        if auto_caption and st.session_state.get('_caption_signature') != signature:
            for key, raw in [('caption_raw', True), ('caption_renamed', False)]:
                st.session_state.pop(key, None)
                ui.settings[key] = make_caption(caption_settings, summary, raw)
            st.session_state['_caption_signature'] = signature
        caption_raw = ui.text_area('Raw caption', key='caption_raw', height=130)
        caption_renamed = ui.text_area('Renamed caption', key='caption_renamed', height=130)
        caption_custom = ui.text_area('Custom caption', key='caption_custom_textarea', height=130)
    settings = dict(settings, png_dpi=png_dpi, caption_style=caption_style, auto_caption=auto_caption)
    snapshot = dict(ui.settings)
    report = {
        'meta': {'app': 'J-type-like Web Viz', 'app_version': APP_VERSION,
                 'generated_at': datetime.now().isoformat(timespec='seconds'),
                 'python': sys.version.split()[0], 'platform': platform.platform(),
                 'streamlit': st.__version__, 'matplotlib': matplotlib.__version__,
                 'pandas': pd.__version__, 'numpy': np.__version__, 'jitter_seed': 2026,
                 'resolved_font': matplotlib.font_manager.FontProperties(fname=matplotlib.font_manager.findfont(settings['font_family'])).get_name()},
        'data': source_info,
        'used_parameters': _safe_serialize(settings),
        'session_state': snapshot,
        'statistics_note': 't/F/p and pairwise p values are user-supplied annotations; this app does not run significance tests.',
        'captions': {'raw': caption_raw, 'renamed': caption_renamed, 'custom': caption_custom},
    }
    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    txt_text = report_to_txt(report) + '\n\nFull settings and provenance\n' + json_text
    base = sanitize_filename(f"{settings['y_col']}_{settings['x_col']}")
    files = {f'{base}.pdf': figure_bytes(fig, 'pdf'),
             f'{base}.png': figure_bytes(fig, 'png', png_dpi),
             f'{base}.svg': figure_bytes(fig, 'svg'),
             f'{base}_params.json': json_text.encode(), f'{base}_params.txt': txt_text.encode(),
             f'{base}_caption_raw.txt': caption_raw.encode(),
             f'{base}_caption_renamed.txt': caption_renamed.encode(),
             f'{base}_caption_custom.txt': caption_custom.encode(),
             f'{base}.{TEMPLATE_FILE_EXT}': json.dumps(template_payload(snapshot), ensure_ascii=False, indent=2).encode()}
    if summary is not None:
        files[f'{base}_summary.csv'] = summary.to_csv(index=False).encode('utf-8-sig')
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    st.download_button('Download ALL · ZIP', bundle.getvalue(), file_name=f'{base}_ALL.zip', mime='application/zip', type='primary', key='download_all')
    with st.expander('Individual files and parameter reports'):
        for name, data in files.items():
            st.download_button(name, data, file_name=name, key='download_'+name)
    with st.expander('Save reusable template'):
        tpl_name = st.text_input('Template name', 'my_template', key='tpl_name')
        tpl_mode = st.selectbox('Template contents', ['Full', 'Minimal'], key='tpl_mode')
        tpl_scope = st.selectbox('Template scope', ['Full plot', 'Global style only'], key='tpl_scope')
        payload = template_payload(snapshot, tpl_mode, tpl_scope)
        st.download_button('Download template', json.dumps(payload, ensure_ascii=False, indent=2), file_name=f'{sanitize_filename(tpl_name)}.{TEMPLATE_FILE_EXT}', mime='application/json', key='download_template')

CONTROL_KEYS = ['add_reg', 'auto_caption', 'bar_edgecolor', 'bar_linewidth', 'bar_point_alpha', 'bar_point_jitter', 'bar_point_size', 'bar_show_points', 'bar_spacing_inch', 'bar_width_inch', 'bar_x', 'bar_y', 'bold', 'bottom_n_font', 'bottom_n_offset', 'bottom_n_summary', 'box_alpha', 'box_bottom_alpha', 'box_bottom_color', 'box_bottom_offset_rel', 'box_edge_color', 'box_edge_lw', 'box_fill_color', 'box_n_fontsize', 'box_point_alpha', 'box_point_size', 'box_show_bottom', 'box_show_n', 'box_show_top', 'box_top_alpha', 'box_top_color', 'box_top_offset_rel', 'box_width', 'box_x', 'box_xtick_offset_rel', 'box_y', 'caption_custom_textarea', 'caption_raw', 'caption_renamed', 'caption_style', 'custom_x_order', 'enable_sig', 'err_capsize', 'err_linewidth', 'f_in', 'fig_height', 'fig_width', 'fix_y_intercept', 'font_family', 'font_size', 'grid_linewidth', 'grid_x', 'grid_y', 'group_col', 'hide_ns', 'jitter_width', 'legend_bg_transparent', 'legend_border_color', 'legend_border_width', 'legend_fontsize', 'legend_loc', 'legend_order_input', 'legend_show_border', 'legend_title_fontsize', 'lock_nice_ticks', 'main_title', 'margin_left_inch', 'margin_right_inch', 'n_bold', 'n_bottom_alpha_bar', 'n_bottom_color_bar', 'n_bottom_offset_bar', 'n_fontsize', 'n_inside_bg', 'n_inside_offset', 'n_top_alpha_bar', 'n_top_color_bar', 'n_top_offset_bar', 'note_in', 'override_by_x', 'p_in', 'pair_count', 'plot_type', 'png_dpi', 'position', 'scat_x', 'scat_y', 'scatter_point_alpha', 'scatter_point_size', 'show_eq', 'show_n_bottom_bar', 'show_n_inside', 'show_n_labels', 'show_n_top_bar', 'show_points', 'show_r2', 'show_se', 'show_stats', 'show_value_labels', 'sig_line_color_default', 'sig_line_lift', 'sig_line_width', 'sig_stack_gap', 'sig_star_bold', 'sig_star_extra_offset', 'sig_star_font', 'sig_tick_length', 'sp_bottom', 'sp_left', 'sp_right', 'sp_top', 'stat_color', 'stat_font_size', 'style_preset', 't_in', 'thr_1', 'thr_2', 'thr_3', 'tick_color', 'title_fontsize', 'value_bold', 'value_decimals', 'value_fontsize', 'x_axis_color', 'x_axis_lw', 'x_dec', 'x_max', 'x_min', 'x_step', 'x_tick_fontsize', 'x_tick_rotation', 'xlabel', 'xlabel_fontsize', 'xlabel_pad', 'y0_color', 'y0_line', 'y_axis_color', 'y_axis_lw', 'y_dec', 'y_intercept_value', 'y_max', 'y_min', 'y_step', 'y_tick_fontsize', 'ylabel', 'ylabel_fontsize']
STYLE_KEYS = ['add_reg', 'auto_caption', 'bar_edgecolor', 'bar_linewidth', 'bar_point_alpha', 'bar_point_jitter', 'bar_point_size', 'bar_show_points', 'bar_spacing_inch', 'bar_width_inch', 'bold', 'bottom_n_font', 'bottom_n_offset', 'bottom_n_summary', 'box_alpha', 'box_bottom_alpha', 'box_bottom_color', 'box_bottom_offset_rel', 'box_edge_color', 'box_edge_lw', 'box_fill_color', 'box_n_fontsize', 'box_point_alpha', 'box_point_size', 'box_show_bottom', 'box_show_n', 'box_show_top', 'box_top_alpha', 'box_top_color', 'box_top_offset_rel', 'box_width', 'box_xtick_offset_rel', 'caption_style', 'custom_x_order', 'err_capsize', 'err_linewidth', 'fig_height', 'fig_width', 'fix_y_intercept', 'font_family', 'font_size', 'grid_linewidth', 'grid_x', 'grid_y', 'hide_ns', 'jitter_width', 'legend_bg_transparent', 'legend_border_color', 'legend_border_width', 'legend_fontsize', 'legend_loc', 'legend_order_input', 'legend_show_border', 'legend_title_fontsize', 'lock_nice_ticks', 'margin_left_inch', 'margin_right_inch', 'n_bold', 'n_bottom_alpha_bar', 'n_bottom_color_bar', 'n_bottom_offset_bar', 'n_fontsize', 'n_inside_bg', 'n_inside_offset', 'n_top_alpha_bar', 'n_top_color_bar', 'n_top_offset_bar', 'override_by_x', 'png_dpi', 'position', 'scatter_point_alpha', 'scatter_point_size', 'show_eq', 'show_n_bottom_bar', 'show_n_inside', 'show_n_labels', 'show_n_top_bar', 'show_points', 'show_r2', 'show_se', 'show_stats', 'show_value_labels', 'sig_line_color_default', 'sig_line_lift', 'sig_line_width', 'sig_stack_gap', 'sig_star_bold', 'sig_star_extra_offset', 'sig_star_font', 'sig_tick_length', 'sp_bottom', 'sp_left', 'sp_right', 'sp_top', 'stat_color', 'stat_font_size', 'style_preset', 'thr_1', 'thr_2', 'thr_3', 'tick_color', 'title_fontsize', 'value_bold', 'value_decimals', 'value_fontsize', 'x_axis_color', 'x_axis_lw', 'x_dec', 'x_step', 'x_tick_fontsize', 'x_tick_rotation', 'xlabel_fontsize', 'xlabel_pad', 'y0_color', 'y0_line', 'y_axis_color', 'y_axis_lw', 'y_dec', 'y_intercept_value', 'y_step', 'y_tick_fontsize', 'ylabel_fontsize']
DYNAMIC_PREFIXES = ("rename_", "gcol_", "shape_", "glabel_", "xcolor_", "xlabel_", "sig_")

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
    if mask.sum() < 2 or np.ptp(x[mask]) == 0:
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


import re

def format_stat_annotation(t_in="", f_in="", p_in="", note_in=""):
    parts = []

    # =========================
    # t statistic
    # =========================
    if t_in.strip():
        t_raw = t_in.strip()

        # 情況 1：有 df，例如 12=2.31 或 (12)=2.31
        if "=" in t_raw:
            match = re.match(r"\(?\s*(\d+)\s*\)?\s*=\s*([-+]?\d*\.?\d+)", t_raw)
            if match:
                df_val, t_val = match.groups()
                parts.append(
                    rf"$\mathit{{t}}({df_val})={float(t_val):.2f}$"
                )
            else:
                parts.append(
                    rf"$\mathit{{t}}={t_raw}$"
                )

        # 情況 2：只有 t 值，例如 1.2
        else:
            try:
                t_val = float(t_raw)
                parts.append(
                    rf"$\mathit{{t}}={t_val:.2f}$"
                )
            except ValueError:
                parts.append(
                    rf"$\mathit{{t}}={t_raw}$"
                )

    # =========================
    # F statistic
    # =========================
    if f_in.strip():
        f_raw = f_in.strip()

        # 有 df，例如 1,12=4.56 或 (1,12)=4.56
        if "=" in f_raw:
            match = re.match(r"\(?\s*([\d,\s]+)\s*\)?\s*=\s*([-+]?\d*\.?\d+)", f_raw)
            if match:
                df_part, f_val = match.groups()
                df_part = df_part.replace(" ", "")
                parts.append(
                    rf"$\mathit{{F}}({df_part})={float(f_val):.2f}$"
                )
            else:
                parts.append(
                    rf"$\mathit{{F}}={f_raw}$"
                )
        else:
            try:
                f_val = float(f_raw)
                parts.append(
                    rf"$\mathit{{F}}={f_val:.2f}$"
                )
            except ValueError:
                parts.append(
                    rf"$\mathit{{F}}={f_raw}$"
                )

    # =========================
    # p value
    # =========================
    if p_in.strip():
        try:
            p_val = float(p_in.strip())

            if p_val < 0.001:
                parts.append(r"$\mathit{p}<0.001$")
            else:
                parts.append(
                    rf"$\mathit{{p}}={p_val:.3f}$"
                )

        except ValueError:
            parts.append(
                rf"$\mathit{{p}}={p_in.strip()}$"
            )

    # =========================
    # Additional note
    # =========================
    if note_in.strip():
        parts.append(note_in.strip())

    return ", ".join(parts)

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


def _safe_serialize(obj: Any, *, _depth: int = 0, _max_depth: int = 5) -> Any:
    """Best-effort JSON-safe serializer for parameter reports.

    - Avoids blowing up on non-serializable objects (fig/ax/df, etc.)
    - Truncates deeply nested structures.
    """
    if _depth > _max_depth:
        return f"<max_depth:{_max_depth}>"

    # Simple primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Numpy
    try:
        import numpy as _np  # already imported globally; keep local for safety
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
    except Exception:
        pass

    # Pandas
    try:
        import pandas as _pd  # already imported globally; keep local for safety
        if isinstance(obj, _pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "columns": [str(c) for c in obj.columns[:200]],
            }
        if isinstance(obj, _pd.Series):
            return {
                "__type__": "Series",
                "shape": [int(obj.shape[0])],
                "name": str(obj.name),
            }
        if isinstance(obj, _pd.Index):
            return {
                "__type__": "Index",
                "shape": [int(obj.shape[0])],
                "name": str(obj.name),
            }
    except Exception:
        pass

    # Dict-like
    if isinstance(obj, dict):
        out = {}
        for k, v in list(obj.items())[:500]:
            out[str(k)] = _safe_serialize(v, _depth=_depth + 1, _max_depth=_max_depth)
        if len(obj) > 500:
            out["__truncated__"] = f"{len(obj) - 500} keys omitted"
        return out

    # List/Tuple/Set
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out = [_safe_serialize(v, _depth=_depth + 1, _max_depth=_max_depth) for v in seq[:1000]]
        if len(seq) > 1000:
            out.append(f"<truncated:{len(seq)-1000} items>")
        return out

    # Fallback: class name / repr (short)
    try:
        r = repr(obj)
        if len(r) > 300:
            r = r[:297] + "..."
        return {"__type__": type(obj).__name__, "repr": r}
    except Exception:
        return {"__type__": type(obj).__name__}


def report_to_txt(report: Dict[str, Any]) -> str:
    """Human-readable report text (for Notion / lab notes)."""
    meta = report.get("meta", {})
    used = report.get("used_parameters", {})

    lines = []
    lines.append("J-type-like Web Viz — Parameter Report")
    lines.append("=" * 40)
    lines.append(f"App: {meta.get('app','')}")
    lines.append(f"Version: {meta.get('app_version','')}")
    lines.append(f"Generated at: {meta.get('generated_at','')}")
    lines.append(f"Python: {meta.get('python','')}")
    lines.append(f"Platform: {meta.get('platform','')}")
    lines.append(f"Streamlit: {meta.get('streamlit','')}")
    lines.append("")

    lines.append("Used parameters")
    lines.append("-" * 40)
    for k in sorted(used.keys()):
        v = used[k]
        if isinstance(v, dict) and "__type__" in v and len(v) <= 6:
            lines.append(f"{k}: {v}")
        else:
            # pretty one-liner
            s = json.dumps(v, ensure_ascii=False)
            if len(s) > 400:
                s = s[:397] + "..."
            lines.append(f"{k}: {s}")
    lines.append("")

    lines.append("Session state (serialized snapshot)")
    lines.append("-" * 40)
    ss = report.get("session_state", {})
    # Keep it readable: list keys only, plus a few values
    try:
        keys = list(ss.keys())
        lines.append(f"Keys ({len(keys)}): " + ", ".join(keys[:80]) + (" ..." if len(keys) > 80 else ""))
    except Exception:
        lines.append("<unavailable>")
    return "\n".join(lines)


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
        if hasattr(df[x_col].dtype, "categories"):
            x_levels = list(df[x_col].cat.categories)
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


def _render_plot(df, settings):
    """Render a copy using explicit settings; return figure and summary."""
    df = df.copy(deep=True)
    add_reg = settings.get('add_reg')
    bar_edgecolor = settings.get('bar_edgecolor')
    bar_linewidth = settings.get('bar_linewidth')
    bar_point_alpha = settings.get('bar_point_alpha')
    bar_point_jitter = settings.get('bar_point_jitter')
    bar_point_size = settings.get('bar_point_size')
    bar_show_points = settings.get('bar_show_points')
    bar_spacing_inch = settings.get('bar_spacing_inch')
    bar_width_inch = settings.get('bar_width_inch')
    bold = settings.get('bold')
    bottom_n_font = settings.get('bottom_n_font')
    bottom_n_offset = settings.get('bottom_n_offset')
    bottom_n_summary = settings.get('bottom_n_summary')
    box_alpha = settings.get('box_alpha')
    box_bottom_alpha = settings.get('box_bottom_alpha')
    box_bottom_color = settings.get('box_bottom_color')
    box_bottom_offset_rel = settings.get('box_bottom_offset_rel')
    box_edge_color = settings.get('box_edge_color')
    box_edge_lw = settings.get('box_edge_lw')
    box_fill_color = settings.get('box_fill_color')
    box_n_fontsize = settings.get('box_n_fontsize')
    box_show_bottom = settings.get('box_show_bottom')
    box_show_n = settings.get('box_show_n')
    box_show_top = settings.get('box_show_top')
    box_top_alpha = settings.get('box_top_alpha')
    box_top_color = settings.get('box_top_color')
    box_top_offset_rel = settings.get('box_top_offset_rel')
    box_width = settings.get('box_width')
    custom_order = settings.get('custom_order')
    enable_sig = settings.get('enable_sig')
    err_capsize = settings.get('err_capsize')
    err_linewidth = settings.get('err_linewidth')
    f_in = settings.get('f_in')
    fig_height = settings.get('fig_height')
    fig_width = settings.get('fig_width')
    fix_y_intercept = settings.get('fix_y_intercept')
    font_size = settings.get('font_size')
    grid_linewidth = settings.get('grid_linewidth')
    grid_x = settings.get('grid_x')
    grid_y = settings.get('grid_y')
    group_col = settings.get('group_col')
    group_colors = settings.get('group_colors')
    group_labels = settings.get('group_labels')
    group_shapes = settings.get('group_shapes')
    hide_ns = settings.get('hide_ns')
    jitter_width = settings.get('jitter_width')
    label_map = settings.get('label_map')
    legend_bg_transparent = settings.get('legend_bg_transparent')
    legend_border_color = settings.get('legend_border_color')
    legend_border_width = settings.get('legend_border_width')
    legend_custom_order = settings.get('legend_custom_order')
    legend_fontsize = settings.get('legend_fontsize')
    legend_loc = settings.get('legend_loc')
    legend_show_border = settings.get('legend_show_border')
    legend_title_fontsize = settings.get('legend_title_fontsize')
    lock_nice_ticks = settings.get('lock_nice_ticks')
    main_title = settings.get('main_title')
    n_bold = settings.get('n_bold')
    n_bottom_alpha_bar = settings.get('n_bottom_alpha_bar')
    n_bottom_color_bar = settings.get('n_bottom_color_bar')
    n_bottom_offset_bar = settings.get('n_bottom_offset_bar')
    n_fontsize = settings.get('n_fontsize')
    n_inside_bg = settings.get('n_inside_bg')
    n_inside_offset = settings.get('n_inside_offset')
    n_top_alpha_bar = settings.get('n_top_alpha_bar')
    n_top_color_bar = settings.get('n_top_color_bar')
    n_top_offset_bar = settings.get('n_top_offset_bar')
    note_in = settings.get('note_in')
    override_by_x = settings.get('override_by_x')
    p_in = settings.get('p_in')
    pair_count = settings.get('pair_count')
    plot_type = settings.get('plot_type')
    point_alpha = settings.get('point_alpha')
    point_size = settings.get('point_size')
    position = settings.get('position')
    show_eq = settings.get('show_eq')
    show_n_bottom_bar = settings.get('show_n_bottom_bar')
    show_n_inside = settings.get('show_n_inside')
    show_n_labels = settings.get('show_n_labels')
    show_n_top_bar = settings.get('show_n_top_bar')
    show_points = settings.get('show_points')
    show_r2 = settings.get('show_r2')
    show_se = settings.get('show_se')
    show_spine_bottom = settings.get('show_spine_bottom')
    show_spine_left = settings.get('show_spine_left')
    show_spine_right = settings.get('show_spine_right')
    show_spine_top = settings.get('show_spine_top')
    show_stats = settings.get('show_stats')
    show_value_labels = settings.get('show_value_labels')
    sig_line_color_default = settings.get('sig_line_color_default')
    sig_line_lift = settings.get('sig_line_lift')
    sig_line_width = settings.get('sig_line_width')
    sig_pairs = settings.get('sig_pairs')
    sig_stack_gap = settings.get('sig_stack_gap')
    sig_star_bold = settings.get('sig_star_bold')
    sig_star_extra_offset = settings.get('sig_star_extra_offset')
    sig_star_font = settings.get('sig_star_font')
    sig_tick_length = settings.get('sig_tick_length')
    stat_color = settings.get('stat_color')
    stat_font_size = settings.get('stat_font_size')
    t_in = settings.get('t_in')
    thr_1 = settings.get('thr_1')
    thr_2 = settings.get('thr_2')
    thr_3 = settings.get('thr_3')
    tick_color = settings.get('tick_color')
    title_fontsize = settings.get('title_fontsize')
    value_bold = settings.get('value_bold')
    value_decimals = settings.get('value_decimals')
    value_fontsize = settings.get('value_fontsize')
    x_axis_color = settings.get('x_axis_color')
    x_axis_lw = settings.get('x_axis_lw')
    x_col = settings.get('x_col')
    x_colors = settings.get('x_colors')
    x_dec = settings.get('x_dec')
    x_max = settings.get('x_max')
    x_min = settings.get('x_min')
    x_step = settings.get('x_step')
    x_tick_fontsize = settings.get('x_tick_fontsize')
    x_tick_rotation = settings.get('x_tick_rotation')
    xlabel = settings.get('xlabel')
    xlabel_fontsize = settings.get('xlabel_fontsize')
    xlabel_pad = settings.get('xlabel_pad')
    y0_color = settings.get('y0_color')
    y0_line = settings.get('y0_line')
    y_axis_color = settings.get('y_axis_color')
    y_axis_lw = settings.get('y_axis_lw')
    y_col = settings.get('y_col')
    y_dec = settings.get('y_dec')
    y_intercept_value = settings.get('y_intercept_value')
    y_max = settings.get('y_max')
    y_min = settings.get('y_min')
    y_step = settings.get('y_step')
    y_tick_fontsize = settings.get('y_tick_fontsize')
    ylabel = settings.get('ylabel')
    ylabel_fontsize = settings.get('ylabel_fontsize')
    # Plotting
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if plot_type.startswith('Bar'):
        left = settings.get('margin_left_inch', 0.5) / fig_width
        right = 1 - settings.get('margin_right_inch', 0.5) / fig_width
        if left >= right:
            raise ValueError('Left and right margins must leave a positive plotting width.')
        fig.subplots_adjust(left=left, right=right)
    elif plot_type.startswith('Box'):
        ax.tick_params(axis='x', pad=settings.get('box_xtick_offset_rel', 0.05) * fig_height * 72)
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
        ax.text(0.5, -0.25 - y_offset_rel, text, ha="center", va="top",
                fontsize=fontsize, transform=ax.transAxes, clip_on=False)
        return None

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

        grouped = df.groupby([x_col, group_col], observed=True)[y_col] if group_col not in (None, "None") else df.groupby(x_col, observed=True)[y_col]

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
            counts = df.groupby([x_col, group_col], observed=True)[y_col].count().unstack(group_col)
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
                            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{xv}|{g}".encode()).digest()[:4], "big"))
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
        # --- 🔒 Ensure categorical order consistency (v0.9.35 fix) ---
        if custom_order:
            df[x_col] = pd.Categorical(
                df[x_col].astype(str),
                categories=custom_order,
                ordered=True
            )
        try:
            import seaborn as sns
            if group_col in (None, "None"):
                sns.boxplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    order=custom_order if custom_order else None,
                    ax=ax,
                    fliersize=0, width=box_width,
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
                    data=df,
                    x=x_col,
                    y=y_col,
                    hue=group_col,
                    order=custom_order if custom_order else None,
                    ax=ax,
                    palette=palette if palette else None,
                    fliersize=0, width=box_width,
                    boxprops=dict(alpha=box_alpha, edgecolor=box_edge_color, linewidth=box_edge_lw),
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
                    positions, x_levels, n_list = _compute_category_counts(df.dropna(subset=[y_col]), x_col)
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
            if group_col not in (None, "None"):
                raise ValueError(f"Grouped Box rendering failed: {e}") from e
            warnings.warn(f"Seaborn not available or failed ({e}), falling back to matplotlib boxplot).")
            if custom_order:
                cats = custom_order
            else:
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
        summary_table = (df.groupby(gb_cols, observed=True)[y_col]
                           .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
                           .reset_index())
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        # --- FIX: apply custom X labels in Box mode ---
        xticks = ax.get_xticks()
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]

        # 使用 label_map 重新映射
        new_labels = [label_map.get(str(lbl), str(lbl)) for lbl in current_labels]

        ax.set_xticks(xticks)
        ax.set_xticklabels(new_labels, fontsize=x_tick_fontsize, rotation=x_tick_rotation)

    # --- SCATTER ---
    else:
        if group_col in (None, "None"):
            color = group_colors.get("default", "#1f77b4")
            _x = pd.to_numeric(df[x_col], errors="coerce")
            _y = pd.to_numeric(df[y_col], errors="coerce")
            _mask = _x.notna() & _y.notna()
            if _mask.sum() == 0:
                warnings.warn(f"No numeric data to plot for X='{x_col}', Y='{y_col}'.")
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
    ax.set_title(main_title, fontweight=fontweight, fontsize=title_fontsize)
    # --- 🔢 Journal-style stats annotation ---
    if show_stats:
        text = format_stat_annotation(t_in, f_in, p_in, note_in)

        if text:
            x_pos = 0.02 if "left" in position else 0.98
            y_pos = 0.98 if "top" in position else 0.02
            ha = "left" if "left" in position else "right"
            va = "top" if "top" in position else "bottom"

            ax.text(
                x_pos, y_pos,
                text,
                transform=ax.transAxes,
                fontsize=stat_font_size,
                color=stat_color,
                ha=ha,
                va=va
            )

    if plot_type.startswith("Scatter") and 'x_min' in locals():
        ax.set_xlim(x_min, x_max)
        xticks = safe_ticks(x_min, x_max, x_step)
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
                line_y_values.append(None)
                continue
            tallest = max(t1, t2)
            # ✅ 階梯式上移，防止重疊
            y_line = tallest + y_range_user * (sig_line_lift + extra_for_top_n + idx * sig_stack_gap)
            line_y_values.append((c1, c2, y_line))
            needed_top = max(needed_top, y_line + y_range_user * (sig_star_extra_offset + 0.03))




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
        warnings.warn(f"Draw y=0 baseline skipped: {e}")

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
        warnings.warn(f"Axis alignment adjustment skipped: {e}")

    if plot_type.startswith("Bar") and xpos is not None:
        n_cats = len(xpos)
        ax.set_xlim(-0.5, n_cats - 0.5)  # 原來用 inch margin，簡化到類別邊界視覺更穩定

    ax.set_xlabel(
        xlabel,
        labelpad=xlabel_pad,
        fontweight=fontweight,
        fontsize=xlabel_fontsize
    )
    ax.set_ylabel(
        ylabel,
        fontweight=fontweight,
        fontsize=ylabel_fontsize
    )
    # --- 🔧 Final X tick style enforcement ---
    for tick in ax.get_xticklabels():
        tick.set_fontsize(x_tick_fontsize)
        tick.set_rotation(x_tick_rotation)
    # Draw pairwise lines (Bar)
    if enable_sig and pair_count > 0 and plot_type.startswith("Bar") and (len(sig_pairs) > 0):
        for idx, comp in enumerate(sig_pairs):
            entry = None if idx >= len(line_y_values) else line_y_values[idx]
            if entry is None: continue
            c1, c2, y_line = entry
            pval = comp.get("p", "")
            stars = significance_label(pval, thr_1, thr_2, thr_3, hide_ns)
            if not stars:
                continue
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
        x_levels = custom_order or list(pd.Index(df[x_col].dropna().astype(str).unique()))
        x_index = {lvl: i for i, lvl in enumerate(x_levels)}
        y_rng_full = ax.get_ylim()[1] - ax.get_ylim()[0]
        pos_pairs, bases, stars, colors = [], [], [], []
        for comp in sig_pairs:
            x1 = str(comp.get("x1")); x2 = str(comp.get("x2"))
            if (x1 not in x_index) or (x2 not in x_index): continue
            label = significance_label(comp.get("p", ""), thr_1, thr_2, thr_3, hide_ns)
            if not label:
                continue
            pos_pairs.append((float(x_index[x1]), float(x_index[x2])))
            t1 = df.loc[df[x_col].astype(str) == x1, y_col].max()
            t2 = df.loc[df[x_col].astype(str) == x2, y_col].max()
            tallest = max(t1, t2) if (pd.notna(t1) and pd.notna(t2)) else ax.get_ylim()[1]
            bases.append(float(tallest) + y_rng_full * sig_line_lift)
            stars.append(label)
            colors.append(comp.get("color", sig_line_color_default) or sig_line_color_default)
        if pos_pairs:
            layers = _assign_layers_for_pairs_box(pos_pairs)
            step = y_rng_full * sig_stack_gap
            highest = max(base + layer * step for base, layer in zip(bases, layers))
            y_max_plot = max(y_max_plot, highest + y_rng_full * (sig_star_extra_offset + 0.06))
            ax.set_ylim(y_min_plot, y_max_plot)
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
        yticks = safe_ticks(y_min_plot, y_max_plot, y_step)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.{int(y_dec)}f}" for y in yticks], fontsize=y_tick_fontsize)
    except ValueError:
        raise

    ax.grid(axis='x', which='both', alpha=0.3 if grid_x else 0.0, linewidth=grid_linewidth)
    ax.grid(axis='y', which='both', alpha=0.3 if grid_y else 0.0, linewidth=grid_linewidth)

    if plot_type.startswith('Box') and group_col not in (None, 'None'):
        handles, labels = ax.get_legend_handles_labels()
        labels = [group_labels.get(str(label), str(label)) for label in labels]
        order_map = {label: i for i, label in enumerate(legend_custom_order)}
        order = sorted(range(len(labels)), key=lambda i: order_map.get(labels[i], 10**9))
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  title=str(group_col), fontsize=legend_fontsize,
                  title_fontsize=legend_title_fontsize)

    # Enforce the same legend placement for all plot families.
    legend = ax.get_legend()
    if legend is not None:
        if legend_loc == "none":
            legend.remove()
        else:
            legend.set_loc(legend_loc)
            apply_legend_style(ax, legend_bg_transparent, legend_show_border,
                               legend_border_color, legend_border_width)
    fig.subplots_adjust(bottom=min(0.45, max(fig.subplotpars.bottom, 0.22)))
    return fig, summary_table

def select_uploaded_data():
    st.session_state['use_example'] = st.session_state.get('data_upload') is None


def main():
    st.set_page_config(page_title='J-type Web Viz · '+APP_VERSION, layout='wide')
    ui = Controls()
    title_area, mode_area = st.columns([3, 1])
    with title_area:
        st.header('J-type Web Viz')
        st.caption('Publication-ready figures · Data → Plot → Appearance → Statistics → Export')
    with mode_area:
        mode = st.radio('Controls', ['Basic', 'Advanced'], horizontal=True, key='view_mode', help='Basic collapses fine adjustments; Advanced opens them. Neither resets settings.')
    with st.container():
        tabs = st.tabs(['Data', 'Plot', 'Appearance', 'Statistics', 'Export'])
        panels = [tab.container(height=300, border=True) for tab in tabs]
    with panels[0]:
        st.subheader('Data source')
        up = st.file_uploader('Upload CSV / XLSX / XLS', type=['csv', 'xlsx', 'xls'], key='data_upload', on_change=select_uploaded_data)
        use_example = st.checkbox('Use example data', value=up is None, key='use_example')
        st.caption('JMP: export your table as CSV or Excel first. Source files are never overwritten.')
        with st.expander('Load saved template'):
            st.file_uploader('Template JSON', type=['json'], key='tpl_uploader')
            st.button('Apply template', on_click=queue_template, disabled=st.session_state.get('tpl_uploader') is None)
            if '_template_notice' in st.session_state:
                st.info(st.session_state['_template_notice'])
        if use_example:
            df = example_dataframe()
            df['Z'] = np.random.default_rng(43).normal(20, 3, len(df))
            source_bytes = df.to_csv(index=False).encode()
            source_name = 'built-in example (synthetic)'
        elif up is not None:
            try:
                source_bytes = up.getvalue()
                source_name = up.name
                df = read_uploaded_file(up)
            except Exception as exc:
                st.error(f'Could not load data: {exc}'); st.stop()
        else:
            st.info('Upload a table or enable example data.'); st.stop()
        df.columns = df.columns.map(str)
        if df.empty or len(df.columns) < 2 or not df.columns.is_unique:
            st.error('Use a nonempty table with at least two uniquely named columns.'); st.stop()
        source_info = {'name': source_name, 'sha256': hashlib.sha256(source_bytes).hexdigest(), 'rows': len(df), 'columns': list(df.columns)}
        st.caption(f'{len(df):,} rows · {len(df.columns)} columns · '+source_name)
        with st.expander('Inspect data', expanded=False):
            st.dataframe(df.head(100), hide_index=True)
    with panels[0]:
        rename_map: Dict[str, str] = {}
        with st.expander('Rename columns', expanded=mode == 'Advanced'):
            for col in df.columns:
                newname = ui.text_input(f'Rename {col}', value=col, key=f'rename_{col}')
                rename_map[col] = newname
        if rename_map:
            if len(set(rename_map.values())) != len(rename_map):
                st.error('Column names must be unique.'); st.stop()
            df = df.rename(columns=rename_map)
        numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
        cat_cols = [c for c in df.columns if not is_numeric_series(df[c])]
        if len(df.columns) < 2:
            st.error('Dataset must have at least two columns.')
            st.stop()
        st.markdown('---')

    with panels[1]:
        plot_type = ui.radio('Plot type', ['Bar (mean ± SE)', 'Scatter', 'Box (show samples)'], horizontal=True, key='plot_type')
        group_col = ui.selectbox('Group (optional)', [None] + df.columns.tolist(), key='group_col')
        if plot_type.startswith('Bar'):
            x_col = ui.selectbox('X (categorical)', cat_cols or df.columns.tolist(), key='bar_x')
            y_col = ui.selectbox('Y (numeric)', numeric_cols or df.columns.tolist(), key='bar_y')
            if y_col not in df.columns or not is_numeric_series(df[y_col]):
                st.error('Selected Y must be numeric.')
                st.stop()
            show_se = ui.checkbox('Show SE bars', True, key='show_se')
            with st.expander('⚙️ Bar & Layout Settings (inch)', expanded=mode == 'Advanced'):
                bar_width_inch = ui.number_input('Bar width (inch)', value=0.25, min_value=0.05, max_value=2.0, step=0.05, key='bar_width_inch')
                bar_spacing_inch = ui.number_input('Between-bar spacing (inch)', value=0.15, min_value=0.0, max_value=2.0, step=0.05, key='bar_spacing_inch')
                margin_left_inch = ui.number_input('Left margin (inch)', value=0.5, min_value=0.0, max_value=3.0, step=0.1, key='margin_left_inch')
                margin_right_inch = ui.number_input('Right margin (inch)', value=0.5, min_value=0.0, max_value=3.0, step=0.1, key='margin_right_inch')
            with st.expander('🧱 Bar Edge & Errorbar Style', expanded=mode == 'Advanced'):
                bar_edgecolor = ui.color_picker('Bar edge color', '#000000', key='bar_edgecolor')
                bar_linewidth = ui.slider('Bar edge linewidth', 0.0, 4.0, 0.6, key='bar_linewidth')
                err_capsize = ui.slider('Errorbar capsize (pt)', 0.0, 20.0, 5.0, key='err_capsize')
                err_linewidth = ui.slider('Errorbar linewidth', 0.2, 4.0, 1.2, key='err_linewidth')
            with st.expander('🟡 Overlay sample points', expanded=mode == 'Advanced'):
                bar_show_points = ui.checkbox('Show samples on bars', False, key='bar_show_points')
                bar_point_size = ui.slider('Point size', 5, 200, 60, key='bar_point_size')
                bar_point_alpha = ui.slider('Point alpha', 0.1, 1.0, 0.6, key='bar_point_alpha')
                bar_point_jitter = ui.slider('Horizontal jitter (axes fraction)', 0.0, 0.2, 0.06, key='bar_point_jitter')
            with st.expander('🔢 (n=xx) Labels – Bar', expanded=mode == 'Advanced'):
                show_n_labels = ui.checkbox('Enable (n=xx) for bars', True, key='show_n_labels')
                if show_n_labels:
                    coln1, coln2 = st.columns(2)
                    with coln1:
                        show_n_top_bar = ui.checkbox('Show top (n=xx)', True, key='show_n_top_bar')
                        n_top_color_bar = ui.color_picker('Top (n) color', '#222222', key='n_top_color_bar')
                        n_top_alpha_bar = ui.slider('Top (n) alpha', 0.1, 1.0, 0.95, key='n_top_alpha_bar')
                        n_top_offset_bar = ui.slider('Top (n) offset (rel y-range)', 0.0, 0.5, 0.03, key='n_top_offset_bar')
                    with coln2:
                        show_n_bottom_bar = ui.checkbox('Show bottom (n=xx)', True, key='show_n_bottom_bar')
                        n_bottom_color_bar = ui.color_picker('Bottom (n) color', '#000000', key='n_bottom_color_bar')
                        n_bottom_alpha_bar = ui.slider('Bottom (n) alpha', 0.1, 1.0, 0.9, key='n_bottom_alpha_bar')
                        n_bottom_offset_bar = ui.slider('Bottom (n) offset (rel y-range)', 0.0, 0.5, 0.08, key='n_bottom_offset_bar')
                    n_fontsize = ui.slider('Font size for (n=xx)', 6, 20, 10, key='n_fontsize')
                    n_bold = ui.checkbox('Bold (n=xx)', False, key='n_bold')
                    show_n_inside = ui.checkbox('Also show inside-bar (if bar>0)', False, key='show_n_inside')
                    n_inside_offset = ui.slider('Inside-bar (n) offset (fraction of bar height)', 0.0, 0.5, 0.15, key='n_inside_offset')
                    n_inside_bg = ui.checkbox('Inside-bar (n) white background', False, key='n_inside_bg')
            with st.expander('🧮 Bottom-center n summary', expanded=mode == 'Advanced'):
                bottom_n_summary = ui.selectbox('Show bottom-center n summary', ['Off', 'By X-level', 'By Group'], index=0, key='bottom_n_summary')
                bottom_n_font = ui.slider('Bottom summary font size', 6, 20, 10, key='bottom_n_font')
                bottom_n_offset = ui.slider('Bottom summary offset (relative y-range)', 0.0, 0.5, 0.08, key='bottom_n_offset')
            with st.expander('🔢 Value Labels on Bars', expanded=mode == 'Advanced'):
                show_value_labels = ui.checkbox('Show mean value on bar tops', False, key='show_value_labels')
                value_decimals = ui.number_input('Decimals', value=2, step=1, min_value=0, max_value=6, key='value_decimals')
                value_fontsize = ui.slider('Value label fontsize', 6, 24, 10, key='value_fontsize')
                value_bold = ui.checkbox('Value label bold', False, key='value_bold')
        elif plot_type.startswith('Box'):
            x_col = ui.selectbox('X (categorical)', cat_cols or df.columns.tolist(), key='box_x')
            y_col = ui.selectbox('Y (numeric)', numeric_cols or df.columns.tolist(), key='box_y')
            show_points = ui.checkbox('Show individual sample points', True, key='show_points')
            with st.expander('Box and sample appearance', expanded=(mode == 'Advanced')):
                point_size = ui.slider('Point size', 5, 100, 30, key='box_point_size')
                point_alpha = ui.slider('Point alpha', 0.2, 1.0, 0.6, key='box_point_alpha')
                jitter_width = ui.slider('Jitter width', 0.0, 0.5, 0.2, key='jitter_width')
                box_width = ui.slider('Box width', 0.1, 0.9, 0.6, key='box_width')
                box_edge_lw = ui.slider('Box edge linewidth', 0.5, 5.0, 1.5, 0.1, key='box_edge_lw')
                box_edge_color = ui.color_picker('Box edge color', '#333333', key='box_edge_color')
                box_fill_color = ui.color_picker('Box fill color', '#4C78A8', key='box_fill_color')
                box_alpha = ui.slider('Box fill alpha', 0.1, 1.0, 0.9, 0.05, key='box_alpha')
            with st.expander('🔢 (n=xx) Labels – Box', expanded=mode == 'Advanced'):
                box_show_n = ui.checkbox('Show (n=xx) per category', True, key='box_show_n')
                if box_show_n:
                    colbn1, colbn2 = st.columns(2)
                    with colbn1:
                        box_show_top = ui.checkbox('Show top (n=xx) – Box', True, key='box_show_top')
                        box_top_color = ui.color_picker('Top (n) color – Box', '#222222', key='box_top_color')
                        box_top_alpha = ui.slider('Top (n) alpha – Box', 0.1, 1.0, 0.95, key='box_top_alpha')
                        box_top_offset_rel = ui.slider('Top (n) offset (rel y-range) – Box', 0.0, 0.5, 0.03, key='box_top_offset_rel')
                    with colbn2:
                        box_show_bottom = ui.checkbox('Show bottom (n=xx) – Box', True, key='box_show_bottom')
                        box_bottom_color = ui.color_picker('Bottom (n) color – Box', '#000000', key='box_bottom_color')
                        box_bottom_alpha = ui.slider('Bottom (n) alpha – Box', 0.1, 1.0, 0.9, key='box_bottom_alpha')
                        box_bottom_offset_rel = ui.slider('Bottom (n) offset (rel y-range) – Box', 0.0, 0.5, 0.1, key='box_bottom_offset_rel')
                    box_n_fontsize = ui.slider('n label fontsize (Box)', 6, 24, 10, key='box_n_fontsize')
                box_xtick_offset_rel = ui.slider('X axis category label vertical offset (rel y-range)', 0.0, 0.3, 0.05, 0.01, key='box_xtick_offset_rel')
        else:
            if len(numeric_cols) < 2:
                st.error('Scatter requires at least two numeric columns.')
                st.stop()
            x_col = ui.selectbox('X (numeric)', numeric_cols, key='scat_x')
            y_col = ui.selectbox('Y (numeric)', [c for c in numeric_cols if c != x_col], key='scat_y')
            add_reg = ui.checkbox('Show regression line', True, key='add_reg')
            show_r2 = ui.checkbox('Legend: show R²', True, key='show_r2')
            show_eq = ui.checkbox('Legend: show equation', False, key='show_eq')
            with st.expander('Scatter point appearance', expanded=(mode == 'Advanced')):
                point_size = ui.slider('Point size', 10, 200, 50, key='scatter_point_size')
                point_alpha = ui.slider('Point alpha', 0.1, 1.0, 0.7, key='scatter_point_alpha')

        if not is_numeric_series(df[y_col]):
            st.error('Y must be numeric.'); st.stop()
        if group_col in (x_col, y_col):
            st.error('Choose a grouping column different from X and Y.'); st.stop()
        if df[y_col].replace([np.inf, -np.inf], np.nan).dropna().empty:
            st.error('Y has no finite observations.'); st.stop()
    # Data-dependent defaults follow the selected columns. Explicit template
    # values are honored on the application run; style settings remain durable.
    axes_signature = (x_col, y_col, group_col, source_info['sha256'])
    old_axes = st.session_state.get('_axes_signature')
    template_applied = st.session_state.pop('_template_applied', False)
    if old_axes and old_axes != axes_signature and not template_applied:
        reset_keys = []
        data_changed = old_axes[3] != axes_signature[3]
        if old_axes[0] != x_col or data_changed:
            reset_keys += ['xlabel', 'x_min', 'x_max', 'x_step', 'custom_x_order']
        if old_axes[1] != y_col or data_changed:
            reset_keys += ['ylabel', 'y_min', 'y_max', 'y_step']
        if old_axes[:2] != axes_signature[:2] or data_changed:
            reset_keys += ['main_title']
        if old_axes[2] != group_col:
            reset_keys += ['legend_order_input']
        for key in reset_keys:
            st.session_state.pop(key, None)
            ui.settings.pop(key, None)
    st.session_state['_axes_signature'] = axes_signature
    with panels[2]:
        style_preset = ui.selectbox('Style preset', ['Default', 'Publication-ready', 'Presentation'], index=1, key='style_preset')
        st.markdown('### 📏 Figure size')
        fig_width = ui.slider('Width (inch)', 4.0, 15.0, 8.0, key='fig_width')
        fig_height = ui.slider('Height (inch)', 3.0, 10.0, 5.0, key='fig_height')
        main_title = ui.text_input('Main title', value=f'{y_col} vs {x_col}', key='main_title')
        xlabel = ui.text_input('X-axis label', value=x_col, key='xlabel')
        ylabel = ui.text_input('Y-axis label', value=y_col, key='ylabel')
        with st.expander('Typography, ticks and axis lines', expanded=mode == 'Advanced'):
            font_size = ui.slider('Base font size', 8, 24, 12, key='font_size')
            title_fontsize = ui.slider('Title fontsize', 6, 40, 14, key='title_fontsize')
            xlabel_fontsize = ui.slider('X label fontsize', 6, 40, 12, key='xlabel_fontsize')
            ylabel_fontsize = ui.slider('Y label fontsize', 6, 40, 12, key='ylabel_fontsize')
            bold = ui.checkbox('Bold axis labels', False, key='bold')
            x_tick_rotation = ui.slider('X tick rotation (deg)', 0, 90, 0, key='x_tick_rotation')
            x_tick_fontsize = ui.slider('X tick fontsize', 6, 20, 10, key='x_tick_fontsize')
            y_tick_fontsize = ui.slider('Y tick fontsize', 6, 20, 10, key='y_tick_fontsize')
            xlabel_pad = ui.slider('X-axis labelpad (distance)', 0, 120, 50, key='xlabel_pad')
            y0_line = ui.checkbox('Draw horizontal line at y=0', False, key='y0_line')
            y0_color = ui.color_picker('y=0 line color', '#000000', key='y0_color')
            x_axis_lw = ui.slider('X axis width', 0.2, 5.0, 1.0, 0.1, key='x_axis_lw')
            x_axis_color = ui.color_picker('X axis color', '#000000', key='x_axis_color')
            y_axis_lw = ui.slider('Y axis width', 0.2, 5.0, 1.0, 0.1, key='y_axis_lw')
            y_axis_color = ui.color_picker('Y axis color', '#000000', key='y_axis_color')
            tick_color = ui.color_picker('Tick/Title color', '#000000', key='tick_color')
        label_map: Dict[str, str] = {}
        custom_order = []
        if plot_type.startswith('Bar') or plot_type.startswith('Box'):
            x_levels_default = list(pd.Index(df[x_col].dropna().astype(str).unique()))
            with st.expander('Edit X-category labels & order', expanded=False if mode == 'Basic' else True):
                for xv in x_levels_default:
                    label_map[xv] = ui.text_input(f'Label for {xv}', value=xv, key=f'xlabel_{xv}')
                st.caption('Enter desired X order (comma-separated):')
                order_input = ui.text_input('Custom X order', value=', '.join(x_levels_default), key='custom_x_order')
                custom_order = list(dict.fromkeys((x.strip() for x in order_input.split(',') if x.strip() in x_levels_default)))
                custom_order += [x for x in x_levels_default if x not in custom_order]
        else:
            with st.expander('📈 X-axis (numeric)', expanded=mode == 'Advanced'):
                x_min = ui.number_input('X min', value=float(df[x_col].min()), key='x_min')
                x_max = ui.number_input('X max', value=float(df[x_col].max()), key='x_max')
                x_step = ui.number_input('X tick step', value=max(float(df[x_col].max() - df[x_col].min()) / 6, 0.01), key='x_step')
                x_dec = ui.number_input('X decimals', value=1, step=1, key='x_dec')
        y_min = ui.number_input('Y min', value=float(min(0, df[y_col].min()) if plot_type.startswith('Bar') else df[y_col].min()), key='y_min')
        y_max = ui.number_input('Y max', value=float(df[y_col].max()), key='y_max')
        y_step = ui.number_input('Y tick step', value=max(float(df[y_col].max() - df[y_col].min()) / 6, 0.01), key='y_step')
        y_dec = ui.number_input('Y decimals', value=1, step=1, key='y_dec')
        lock_nice_ticks = ui.checkbox('Lock nice ticks (align to step)', False, key='lock_nice_ticks')
        with st.expander('🧭 Grid', expanded=mode == 'Advanced'):
            grid_x = ui.checkbox('Show x-grid', False, key='grid_x')
            grid_y = ui.checkbox('Show y-grid', True, key='grid_y')
            grid_linewidth = ui.slider('Grid linewidth', 0.2, 2.5, 0.6, key='grid_linewidth')
        with st.expander('📍 Axis intersection control', expanded=mode == 'Advanced'):
            fix_y_intercept = ui.checkbox('Fix Y-axis intersection at specific value', False, key='fix_y_intercept')
            if fix_y_intercept:
                y_intercept_value = ui.number_input('Y value to align with X-axis', value=0.0, key='y_intercept_value')
            else:
                y_intercept_value = None
        with st.expander('🧱 Spines', expanded=mode == 'Advanced'):
            show_spine_left = ui.checkbox('Show left spine', True, key='sp_left')
            show_spine_bottom = ui.checkbox('Show bottom spine', True, key='sp_bottom')
            show_spine_right = ui.checkbox('Show right spine', False, key='sp_right')
            show_spine_top = ui.checkbox('Show top spine', False, key='sp_top')
        legend_bg_transparent, legend_show_border, legend_border_color, legend_border_width = True, False, '#000000', 1.0
        font_options = ['Arial', 'Helvetica', 'Times New Roman', 'Calibri', 'Liberation Sans', 'DejaVu Sans']
        font_family = ui.selectbox('Global font family', font_options, index=0, key='font_family')
        with st.container(border=True):
            st.markdown('**Colors, shapes and legend labels**')
            group_colors: Dict[Any, str] = {}
            group_shapes: Dict[Any, str] = {}
            group_labels: Dict[Any, str] = {}
            legend_custom_order: List[str] = []
            legend_fontsize = 10
            legend_title_fontsize = 10
            if group_col not in (None, 'None'):
                g_levels = list(pd.Index(df[group_col].dropna().astype(str).unique()))
                per_row = 2
                for row_start in range(0, len(g_levels), per_row):
                    row_groups = g_levels[row_start:row_start + per_row]
                    cols = st.columns(len(row_groups))
                    for col_i, g in enumerate(row_groups):
                        with cols[col_i]:
                            st.markdown(f'**{g}**')
                            group_colors[g] = ui.color_picker('Color', value=PALETTE[g_levels.index(g) % len(PALETTE)], key=f'gcol_{g}')
                            group_shapes[g] = ui.selectbox('Shape', {'o': '○ Circle', 's': '□ Square', '^': '△ Triangle', 'D': '◇ Diamond'}, key=f'shape_{g}')
                            group_labels[g] = ui.text_input('Legend label', value=g, key=f'glabel_{g}')
                            st.markdown('---')
                with st.expander('Legend order', expanded=False if mode == 'Basic' else True):
                    st.caption('自訂圖例順序（以逗號分隔；使用上方右列『Label for ...』文字）')
                    default_order = ', '.join([group_labels[g] for g in g_levels])
                    order_input = ui.text_input('Custom legend order', value=default_order, key='legend_order_input')
                    legend_custom_order = [x.strip() for x in order_input.split(',') if x.strip()]
                    st.markdown('---')
                    st.markdown('**Legend font settings**')
                    row1 = st.columns(1)
                    with row1[0]:
                        legend_fontsize = ui.slider('Legend font size', 6, 24, 10, key='legend_fontsize')
                        legend_title_fontsize = ui.slider('Legend title font size', 6, 28, 10, key='legend_title_fontsize')
                    st.markdown('---')
                    st.markdown('**Legend appearance**')
                    row1 = st.columns(1)
                    with row1[0]:
                        legend_bg_transparent = ui.checkbox('Transparent bg', True, key='legend_bg_transparent')
                        legend_show_border = ui.checkbox('Show border', False, key='legend_show_border')
                        legend_border_color = ui.color_picker('Border color', '#000000', disabled=not legend_show_border, key='legend_border_color')
                        legend_border_width = ui.slider('Border width', 0.2, 4.0, 1.0, 0.1, disabled=not legend_show_border, key='legend_border_width')
            if plot_type.startswith('Bar'):
                with st.expander('🎨 X-level Colors', expanded=mode == 'Advanced'):
                    x_levels = list(pd.Index(df[x_col].dropna().astype(str).unique()))
                    x_colors: Dict[str, str] = {}
                    for xv in x_levels:
                        x_colors[xv] = ui.color_picker(f'Color for {xv}', value=PALETTE[x_levels.index(xv) % len(PALETTE)], key=f'xcolor_{xv}')
                    override_by_x = ui.checkbox('Override group colors by X-level', False, key='override_by_x')
            else:
                x_colors = {}
        legend_loc = ui.selectbox('Legend position (if grouping)', options=['best', 'upper right', 'upper left', 'lower right', 'lower left', 'upper center', 'lower center', 'center left', 'center right', 'center', 'none'], index=0, key='legend_loc')
    with panels[3]:
        st.caption('Annotations only: enter results from your statistical analysis. No t-test, ANOVA or pairwise test is computed here.')
        with st.expander('Manual t / F / p annotation', expanded=True):
            show_stats = ui.checkbox('Show stats on plot', True, key='show_stats')
            position = ui.selectbox('Annotation position', ['top-left', 'top-right', 'bottom-left', 'bottom-right'], key='position')
            stat_font_size = ui.slider('Stats font size', 8, 24, 12, key='stat_font_size')
            stat_color = ui.color_picker('Stats text color', value='#000000', key='stat_color')
            t_in = ui.text_input('t value', key='t_in')
            f_in = ui.text_input('F value', key='f_in')
            p_in = ui.text_input('p value', key='p_in')
            note_in = ui.text_input('note', key='note_in')
        enable_sig = ui.checkbox('Enable pairwise significance', False, key='enable_sig')
        with st.expander('Thresholds and comparison appearance', expanded=(mode == 'Advanced')):
            hide_ns = ui.checkbox('Hide non-significant (p>=thr)', True, key='hide_ns')
            thr_1 = ui.number_input('p < threshold (*)', value=0.05, format='%.5f', key='thr_1')
            thr_2 = ui.number_input('p < threshold (**)', value=0.01, format='%.5f', key='thr_2')
            thr_3 = ui.number_input('p < threshold (***)', value=0.001, format='%.5f', key='thr_3')
            sig_line_width = ui.slider('Pairwise line width', 0.5, 5.0, 1.5, key='sig_line_width')
            sig_line_color_default = ui.color_picker('Pairwise default line color', '#000000', key='sig_line_color_default')
            sig_stack_gap = ui.slider('Stacking gap per comparison (relative y-range)', 0.01, 0.25, 0.05, key='sig_stack_gap')
            sig_line_lift = ui.slider('Lift above tallest bar (relative y-range)', 0.0, 0.4, 0.06, key='sig_line_lift')
            sig_tick_length = ui.slider('End tick length (relative y-range)', 0.002, 0.05, 0.01, key='sig_tick_length')
            sig_star_font = ui.slider('Pairwise text size', 8, 32, 12, key='sig_star_font')
            sig_star_bold = ui.checkbox('Pairwise text bold', True, key='sig_star_bold')
            sig_star_extra_offset = ui.slider('Text extra offset (relative y-range)', 0.0, 0.15, 0.02, key='sig_star_extra_offset')
        max_pairs = 12
        pair_count = ui.number_input('Number of comparisons', min_value=0, max_value=max_pairs, value=0, step=1, key='pair_count')
        sig_pairs: List[Dict[str, Any]] = []
        if enable_sig and pair_count > 0 and plot_type.startswith('Bar'):
            st.caption('Select targets to compare. For grouped bars, pick both X and Group for each side.')
            for i in range(int(pair_count)):
                with st.expander(f'Comparison #{i + 1}', expanded=False):
                    if group_col in (None, 'None'):
                        x_lvls = list(pd.Index(df[x_col].dropna().astype(str).unique()))
                        c1 = ui.selectbox('X1', x_lvls, key=f'sig_x1_{i}')
                        c2 = ui.selectbox('X2', [x for x in x_lvls if x != st.session_state.get(f'sig_x1_{i}', x_lvls[0])], key=f'sig_x2_{i}')
                        pval = ui.text_input("p-value (number or 'ns')", key=f'sig_p_{i}')
                        line_color = ui.color_picker('Line color', sig_line_color_default, key=f'sig_color_{i}')
                        sig_pairs.append({'x1': c1, 'g1': None, 'x2': c2, 'g2': None, 'p': pval, 'color': line_color})
                    else:
                        x_lvls = list(pd.Index(df[x_col].dropna().astype(str).unique()))
                        g_lvls = list(pd.Index(df[group_col].dropna().astype(str).unique()))
                        row1c = st.columns(2)
                        with row1c[0]:
                            x1 = ui.selectbox('X1', x_lvls, key=f'sig_x1_{i}')
                        with row1c[1]:
                            g1 = ui.selectbox('Group1', g_lvls, key=f'sig_g1_{i}')
                        row2c = st.columns(2)
                        with row2c[0]:
                            x2 = ui.selectbox('X2', x_lvls, key=f'sig_x2_{i}')
                        with row2c[1]:
                            g2 = ui.selectbox('Group2', g_lvls, key=f'sig_g2_{i}')
                        pval = ui.text_input("p-value (number or 'ns')", key=f'sig_p_{i}')
                        line_color = ui.color_picker('Line color', sig_line_color_default, key=f'sig_color_{i}')
                        sig_pairs.append({'x1': x1, 'g1': g1, 'x2': x2, 'g2': g2, 'p': pval, 'color': line_color})
        elif enable_sig and pair_count > 0 and plot_type.startswith('Box'):
            st.caption('Box comparisons span X categories (pooled over groups), not individual hue boxes.')
            x_lvls = list(pd.Index(df[x_col].dropna().astype(str).unique()))
            for i in range(int(pair_count)):
                with st.expander(f'Comparison (Box) #{i + 1}', expanded=False):
                    c1 = ui.selectbox('X1', x_lvls, key=f'sig_box_x1_{i}')
                    c2 = ui.selectbox('X2', [x for x in x_lvls if x != st.session_state.get(f'sig_box_x1_{i}', x_lvls[0])], key=f'sig_box_x2_{i}')
                    pval = ui.text_input("p-value (number or 'ns')", key=f'sig_box_p_{i}')
                    line_color = sig_line_color_default
                    sig_pairs.append({'x1': c1, 'g1': None, 'x2': c2, 'g2': None, 'p': pval, 'color': line_color})
    effective_keys = ['add_reg', 'bar_edgecolor', 'bar_linewidth', 'bar_point_alpha', 'bar_point_jitter', 'bar_point_size', 'bar_show_points', 'bar_spacing_inch', 'bar_width_inch', 'bold', 'bottom_n_font', 'bottom_n_offset', 'bottom_n_summary', 'box_alpha', 'box_bottom_alpha', 'box_bottom_color', 'box_bottom_offset_rel', 'box_edge_color', 'box_edge_lw', 'box_fill_color', 'box_n_fontsize', 'box_show_bottom', 'box_show_n', 'box_show_top', 'box_top_alpha', 'box_top_color', 'box_top_offset_rel', 'box_width', 'box_xtick_offset_rel', 'custom_order', 'enable_sig', 'err_capsize', 'err_linewidth', 'f_in', 'fig_height', 'fig_width', 'fix_y_intercept', 'font_family', 'font_size', 'grid_linewidth', 'grid_x', 'grid_y', 'group_col', 'group_colors', 'group_labels', 'group_shapes', 'hide_ns', 'jitter_width', 'label_map', 'legend_bg_transparent', 'legend_border_color', 'legend_border_width', 'legend_custom_order', 'legend_fontsize', 'legend_loc', 'legend_show_border', 'legend_title_fontsize', 'lock_nice_ticks', 'main_title', 'margin_left_inch', 'margin_right_inch', 'n_bold', 'n_bottom_alpha_bar', 'n_bottom_color_bar', 'n_bottom_offset_bar', 'n_fontsize', 'n_inside_bg', 'n_inside_offset', 'n_top_alpha_bar', 'n_top_color_bar', 'n_top_offset_bar', 'note_in', 'override_by_x', 'p_in', 'pair_count', 'plot_type', 'point_alpha', 'point_size', 'position', 'rename_map', 'show_eq', 'show_n_bottom_bar', 'show_n_inside', 'show_n_labels', 'show_n_top_bar', 'show_points', 'show_r2', 'show_se', 'show_spine_bottom', 'show_spine_left', 'show_spine_right', 'show_spine_top', 'show_stats', 'show_value_labels', 'sig_line_color_default', 'sig_line_lift', 'sig_line_width', 'sig_pairs', 'sig_stack_gap', 'sig_star_bold', 'sig_star_extra_offset', 'sig_star_font', 'sig_tick_length', 'stat_color', 'stat_font_size', 'style_preset', 't_in', 'thr_1', 'thr_2', 'thr_3', 'tick_color', 'title_fontsize', 'value_bold', 'value_decimals', 'value_fontsize', 'x_axis_color', 'x_axis_lw', 'x_col', 'x_colors', 'x_dec', 'x_max', 'x_min', 'x_step', 'x_tick_fontsize', 'x_tick_rotation', 'xlabel', 'xlabel_fontsize', 'xlabel_pad', 'y0_color', 'y0_line', 'y_axis_color', 'y_axis_lw', 'y_col', 'y_dec', 'y_intercept_value', 'y_max', 'y_min', 'y_step', 'y_tick_fontsize', 'ylabel', 'ylabel_fontsize']
    settings = {k: v for k, v in locals().items() if k in effective_keys}
    try:
        safe_ticks(y_min, y_max, y_step)
        if plot_type == 'Scatter':
            safe_ticks(x_min, x_max, x_step)
        if not 0 < thr_3 < thr_2 < thr_1 <= 1:
            raise ValueError('Thresholds must satisfy 0 < *** < ** < * ≤ 1.')
        for pair in sig_pairs:
            significance_label(pair['p'], thr_1, thr_2, thr_3, hide_ns)
        required = list(dict.fromkeys([x_col, y_col] + ([group_col] if group_col else [])))
        plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        if plot_df.empty:
            raise ValueError('No complete finite observations for the selected columns.')
        if group_col:
            plot_df = plot_df.copy()
            plot_df[group_col] = plot_df[group_col].astype(str)
        source_info['plotted_rows'] = len(plot_df)
        source_info['excluded_rows'] = len(df) - len(plot_df)
        if source_info['excluded_rows']:
            st.info(f"Excluded {source_info['excluded_rows']} rows with missing/nonfinite values in selected columns; source data are unchanged.")
        fig, summary_table = render_plot(plot_df, settings)
    except Exception as exc:
        st.error(f'Check plot settings: {exc}')
        st.stop()
    try:
        st.subheader('Plot preview')
        st.caption(f"{plot_type} · {y_col} vs {x_col} · Settings above do not reduce the preview width")
        st.image(figure_bytes(fig, 'png', 110), width=850)
        with st.expander('Summary table'):
            if summary_table is not None:
                st.dataframe(summary_table, hide_index=True)
            else:
                st.caption('Scatter has no group summary table.')
        with panels[4]:
            export_panel(fig, summary_table, settings, ui, source_info)
    finally:
        plt.close(fig)


if __name__ == '__main__':
    main()
