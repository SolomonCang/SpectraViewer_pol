import base64
import io
import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------------------------------------------------------
def wl2v(wl, wl0):
    """波长→速度 (km/s)"""
    c_kms = 299792.458
    return (wl - wl0) / wl0 * c_kms


def v2dwl(v, wl0):
    # convert velocity to differnce of wavelength
    c_kms = 299792.458
    return wl0 * v / c_kms


def identify_portions(x):
    """按波长或速度突变切段"""
    labels, cur = [], 0
    prev = x.iloc[0]
    for xi in x:
        if xi < prev:
            cur += 1
        labels.append(cur)
        prev = xi
    return labels


def parse_text_to_df(text):
    """
    尝试智能解析为 DataFrame：
    - 空白分隔
    - 先自动探测应跳过的前导非数据行（注释/元信息）
    - 若自动探测失败，回退为 skiprows=2
    - 忽略以 * 开头的注释行
    """
    buf = io.StringIO(text)
    lines = buf.getvalue().splitlines()

    # 过滤掉以 * 开头的行（注释）
    non_comment = [ln for ln in lines if not ln.lstrip().startswith("*")]

    # 从头找到首个“更像数据”的行：至少3个数值，避免把“213542 2”当作数据
    start_idx = None
    for i, ln in enumerate(non_comment):
        parts = ln.strip().split()
        nums = 0
        for p in parts:
            try:
                float(p)
                nums += 1
            except Exception:
                pass
        if nums >= 3:
            start_idx = i
            break

    try:
        if start_idx is not None:
            data_text = "\n".join(non_comment[start_idx:])
            df = pd.read_csv(io.StringIO(data_text),
                             sep=r"\s+",
                             header=None,
                             engine="python")
            return df
        else:
            # 回退方案：沿用原逻辑
            df = pd.read_csv(io.StringIO(text),
                             sep=r"\s+",
                             comment='*',
                             skiprows=2,
                             header=None,
                             engine="python")
            return df
    except Exception:
        return None


def _assign_columns_by_type(df, file_type_hint):
    """
    按给定类型重命名列并返回 (df, x_col)；若不匹配则返回 (None, 错误信息)
    """
    ncol = df.shape[1]
    if file_type_hint == "spec_pol":
        if ncol != 6:
            return None, f"Spec (pol) 期望6列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["Wav", "Int", "Pol", "Null1", "Null2", "sigma_int"]
        x_col = "Wav"
    elif file_type_hint == "spec_i":
        if ncol != 3:
            return None, f"Spec (I) 期望3列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["Wav", "Int", "sigma_int"]
        x_col = "Wav"
    elif file_type_hint == "lsd_pol":
        if ncol != 7:
            return None, f"LSD (pol) 期望7列，但有 {ncol} 列"
        df = df.copy()
        df.columns = [
            "RV", "Int", "sigma_int", "Pol", "sigma_pol", "Null1",
            "sigma_null1"
        ]
        x_col = "RV"
    elif file_type_hint == "lsd_i":
        if ncol != 3:
            return None, f"LSD (I) 期望3列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["RV", "Int", "sigma_int"]
        x_col = "RV"
    else:
        return None, f"未知文件类型: {file_type_hint}"
    return (df, x_col), None


def _heuristic_guess(df):
    """
    启发式自动判别数据类型：
    - 先用列数硬规则
    - 3 列时根据第一列数值范围判断是 Wav(nm) 还是 RV(km/s)
    """
    if df is None or df.empty:
        return None, None

    df = df.dropna(axis=1, how="all")
    ncol = df.shape[1]
    if ncol not in (3, 6, 7):
        return None, None

    if ncol == 6:
        return "spec_pol", _assign_columns_by_type(df, "spec_pol")[0]
    if ncol == 7:
        return "lsd_pol", _assign_columns_by_type(df, "lsd_pol")[0]

    # 3 列：区分 spec_i vs lsd_i
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if x.isna().all():
        return None, None

    xmin, xmax = x.min(), x.max()
    is_wav = (xmin >= 200) and (xmax <= 5000)
    is_rv = (xmin < 0) or (abs(xmin) <= 10000 and abs(xmax) <= 10000
                           and xmax < 200)

    if is_wav and not is_rv:
        return "spec_i", _assign_columns_by_type(df, "spec_i")[0]
    if is_rv and not is_wav:
        return "lsd_i", _assign_columns_by_type(df, "lsd_i")[0]

    for ft in ["spec_i", "lsd_i"]:
        res, err = _assign_columns_by_type(df, ft)
        if not err:
            return ft, res

    return None, None


def detect_and_assign_columns(df, file_type_hint):
    """
    根据列数和用户选择的 file_type 提示，给 DataFrame 设定标准列名。
    - file_type_hint == "auto" 时自动探测
    - 其他值时严格按照指定类型验证
    返回 ((df_named, x_col, resolved_type), err_msg)
    """
    if file_type_hint == "auto":
        guessed_type, res = _heuristic_guess(df)
        if guessed_type is None or res is None:
            return None, "无法自动判别数据类型，请手动选择文件类型"
        (df_named, x_col) = res
        return (df_named, x_col, guessed_type), None
    else:
        res, err = _assign_columns_by_type(df, file_type_hint)
        if err:
            return None, err
        df_named, x_col = res
        return (df_named, x_col, file_type_hint), None


# -----------------------------------------------------------------------------
# 谱线按钮配置：标题 -> 波长(nm)
LINES = {
    # Hydrogen Balmer series
    "Hα 6563": 656.28,
    "Hβ 4861": 486.13,
    "Hγ 4340": 434.05,
    "Hδ 4102": 410.17,
    "Hε 3970": 397.01,
    "Hζ 3889": 388.9064,
    # Helium
    "He I D3 5876": 587.56,
    "He I 4472": 447.15,
    "He I 4026": 402.62,
    "He I 3889": 388.86,
    "He II 4686": 468.57038,
    # Sodium
    "Na II D1 5890": 588.995,
    "Na II D2 5896": 589.592,
    # Calcium
    "Ca II K 3934": 393.37,
    "Ca II H 3969": 396.85,
    "Ca II IRT-1 8498": 849.80,
    "Ca II IRT-2 8542": 854.21,
    "Ca II IRT-3 8662": 866.21,
}


def make_line_buttons():
    """
    生成谱线按钮（分组为若干行，便于手机端换行）
    """
    buttons = []
    for label in LINES.keys():
        btn_id = f"btn-{label}"
        buttons.append(
            html.Button(label,
                        id=btn_id,
                        n_clicks=0,
                        style={
                            "margin": "4px",
                            "padding": "4px 8px"
                        }))
    # 使用一个可换行的容器
    return html.Div(
        id="line-buttons",
        children=buttons,
        style={
            "marginBottom": "20px",
            "display": "none",  # 初始隐藏，按类型显示
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "6px"
        })


# -----------------------------------------------------------------------------
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H2("Spectrum Viewer"),

        # 上传
        html.Div([
            html.Label("拖拽/选择上传文件:"),
            dcc.Upload(id="upload-data",
                       children=html.Div(["将文件拖拽到此处，或 ",
                                          html.A("点击选择文件")]),
                       style={
                           "width": "100%",
                           "height": "60px",
                           "lineHeight": "60px",
                           "borderWidth": "1px",
                           "borderStyle": "dashed",
                           "borderRadius": "5px",
                           "textAlign": "center",
                           "marginBottom": "15px",
                       },
                       multiple=False),
        ],
                 style={"marginBottom": "20px"}),

        # 文件类型选择（增加 auto）
        html.Div([
            html.Label("文件类型:"),
            dcc.RadioItems(id="file-type",
                           options=[
                               {
                                   "label": "自动探测",
                                   "value": "auto"
                               },
                               {
                                   "label": "Spec (pol)",
                                   "value": "spec_pol"
                               },
                               {
                                   "label": "Spec (I)",
                                   "value": "spec_i"
                               },
                               {
                                   "label": "LSD (pol)",
                                   "value": "lsd_pol"
                               },
                               {
                                   "label": "LSD (I)",
                                   "value": "lsd_i"
                               },
                           ],
                           value="auto",
                           inline=True)
        ],
                 style={"marginBottom": "10px"}),

        # 探测结果显示
        html.Div([
            html.Span("探测/采用的数据类型: "),
            html.Strong(id="resolved-type", children="未加载")
        ],
                 style={
                     "marginBottom": "10px",
                     "color": "#444"
                 }),

        # wl0 与窗口宽度
        html.Div([
            html.Label("波长零点 wl0 (nm, 用于SPEC跳转到LSD视图):"),
            dcc.Input(id="wl0-input", type="number", value=656.28, step=0.01)
        ],
                 style={"marginBottom": "10px"}),
        html.Div([
            html.Label("速度范围 ± (km/s):"),
            dcc.Input(id="vel-range",
                      type="number",
                      value=20,
                      style={'width': '120px'})
        ],
                 style={'marginBottom': '20px'}),

        # 谱线按钮区域（自动生成）
        make_line_buttons(),

        # 列选择
        html.Div([
            html.Label("选择要显示的列:"),
            dcc.Checklist(
                id="col-checklist", options=[], value=[], inline=True)
        ],
                 style={"marginBottom": "20px"}),

        # 主图
        dcc.Graph(id="spectrum-figure"),

        # 存数据
        dcc.Store(id="spectrum-data"),
        dcc.Store(id="spectrum-xcol"),
        dcc.Store(id="spectrum-type"),
        dcc.Store(id="clicked-line-label"),  # 存最后点击的谱线标签
    ],
    style={
        "width": "90%",
        "maxWidth": "1200px",
        "margin": "auto"
    })


# 回调0：根据文件类型显示/隐藏谱线按钮
@app.callback(Output("line-buttons", "style"), Input("file-type", "value"))
def toggle_line_buttons(file_type):
    if file_type in ("spec_pol", "spec_i", "auto"):
        return {
            "marginBottom": "20px",
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "6px"
        }
    else:
        return {"marginBottom": "20px", "display": "none"}


# 回调1：接收上传文件并解析，填充 Checklist，并记录解析出来的类型
@app.callback(
    Output("spectrum-data", "data"),
    Output("spectrum-xcol", "data"),
    Output("spectrum-type", "data"),
    Output("col-checklist", "options"),
    Output("col-checklist", "value"),
    Output("resolved-type", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("file-type", "value"),
)
def load_spectrum(contents, filename, file_type):
    if contents is None:
        return None, None, None, [], [], "未加载"

    # 解码上传内容
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            text = decoded.decode('utf-8')
        except UnicodeDecodeError:
            text = decoded.decode('latin-1')
    except Exception as e:
        msg = f"上传解析失败: {e}"
        print(msg)
        return None, None, None, [], [], msg

    df_raw = parse_text_to_df(text)
    if df_raw is None:
        msg = "文件内容解析为表格失败"
        print(msg)
        return None, None, None, [], [], msg

    # 自动/指定类型 -> 设定列名
    result, err = detect_and_assign_columns(df_raw, file_type)
    if err:
        print(err)
        return None, None, None, [], [], err

    (df, x_col, resolved_type) = result

    # 切段（基于x轴：Wav或RV）
    df["order_id"] = identify_portions(df[x_col])

    # 生成 Checklist 选项，排除x轴列和order_id
    options = [{
        "label": c,
        "value": c
    } for c in df.columns if c not in (x_col, "order_id")]
    # 默认勾选 Int、Pol、Null1 若存在
    default = [c for c in ["Int", "Pol", "Null1"] if c in df.columns]

    resolved_text = {
        "spec_pol": "Spec (pol)",
        "spec_i": "Spec (I)",
        "lsd_pol": "LSD (pol)",
        "lsd_i": "LSD (I)",
    }.get(resolved_type, resolved_type)

    return (df.to_json(date_format="iso", orient="split"), x_col,
            resolved_type, options, default, resolved_text)


# 动态生成所有谱线按钮的 Input 列表
def line_button_inputs():
    return [Input(f"btn-{label}", "n_clicks") for label in LINES.keys()]


# 回调：绘制谱线并在按钮点击时同步调整 I/V/N 三个面板的视窗
@app.callback(
    Output("spectrum-figure", "figure"),
    Output("clicked-line-label", "data"),
    # 动态谱线按钮 Inputs
    *line_button_inputs(),
    Input("col-checklist", "value"),
    Input("file-type", "value"),
    Input("wl0-input", "value"),
    Input("vel-range", "value"),
    State("spectrum-data", "data"),
    State("spectrum-xcol", "data"),
    State("spectrum-type", "data"),
    State("clicked-line-label", "data"),
)
def update_figure(*args):
    num_btns = len(LINES)
    btn_nclicks = args[:num_btns]
    selected_cols, file_type, wl0, vel_range, data_json, x_col, resolved_type, last_clicked_label = args[
        num_btns:]

    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=[1, 1, 3],
                        subplot_titles=("", "", ""))

    # 未加载
    if data_json is None or x_col is None:
        fig.update_layout(title="请上传数据文件")
        ftype = resolved_type or file_type or ""
        xlabel = "Wavelength (nm)" if str(ftype).startswith(
            "spec") else "Radial Velocity (km/s)"
        fig.update_xaxes(title_text=xlabel, row=3, col=1)
        fig.update_yaxes(title_text="Flux", row=3, col=1)
        return fig, last_clicked_label

    df = pd.read_json(data_json, orient="split")
    x = df[x_col]

    # 面板默认列集合（存在即用）
    base_panel_cols = {
        "N": [c for c in ["Null1", "Null2", "Null"] if c in df.columns],
        "V": [c for c in ["Pol", "V"] if c in df.columns],
        "I": [c for c in ["Int"] if c in df.columns],
    }

    # 应用于绘图的列：优先使用用户勾选，若无勾选则使用默认列
    def effective_cols(panel_key):
        if selected_cols:
            chosen = [
                c for c in base_panel_cols[panel_key] if c in selected_cols
            ]
            if chosen:
                return chosen
        return base_panel_cols[panel_key]

    panel_cols = {k: effective_cols(k) for k in base_panel_cols.keys()}

    # 颜色
    color_cycle = {
        "Int": "#1f77b4",
        "Pol": "#d62728",
        "V": "#d62728",
        "Null": "#2ca02c",
        "Null1": "#2ca02c",
        "Null2": "#17becf",
        "sigma_int": "#9467bd"
    }

    # 绘制
    def add_traces_for_columns(cols, row_idx):
        if not cols:
            return
        for col in cols:
            for pid, grp in df.groupby("order_id"):
                fig.add_trace(go.Scatter(
                    x=x.loc[grp.index],
                    y=grp[col],
                    mode="lines",
                    name=f"{col}-seg{pid}",
                    line=dict(width=1, color=color_cycle.get(col, None)),
                    opacity=0.9,
                    showlegend=True if pid == 0 else False),
                              row=row_idx,
                              col=1)

    add_traces_for_columns(panel_cols["N"], row_idx=1)
    add_traces_for_columns(panel_cols["V"], row_idx=2)
    add_traces_for_columns(panel_cols["I"], row_idx=3)

    # 坐标轴标签
    use_type = resolved_type or file_type or ""
    xlabel = "Wavelength (nm)" if str(use_type).startswith(
        "spec") else "Radial Velocity (km/s)"
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    fig.update_yaxes(title_text="N", row=1, col=1)
    fig.update_yaxes(title_text="V", row=2, col=1)
    fig.update_yaxes(title_text="I", row=3, col=1)

    fig.update_layout(hovermode="x unified",
                      legend_orientation="h",
                      legend_y=1.02)

    # 按钮点击后同时调整 I/V/N 三个面板
    trig_id = ctx.triggered_id
    clicked_label = last_clicked_label
    if str(use_type).startswith("spec") and trig_id and trig_id.startswith(
            "btn-"):
        clicked_label = trig_id.replace("btn-", "", 1)
        wl_c = LINES.get(clicked_label)
        if wl_c is not None:
            # 计算 x 窗口
            left, right = wl_c - 1, wl_c + 1
            if vel_range is not None:
                dwl = v2dwl(vel_range, wl0 if wl0 is not None else wl_c)
                left, right = wl_c - dwl, wl_c + dwl
            # 同步 x 范围
            fig.update_xaxes(range=[left, right])

            # 对三面板各自自适应 y
            def adapt_y_for_panel(cols, row_idx):
                if not cols:
                    return
                mask = (x >= left) & (x <= right)
                if not mask.any():
                    return
                ymin, ymax = None, None
                for col in cols:
                    ys = df.loc[mask, col]
                    if ys.empty:
                        continue
                    cmin, cmax = ys.min(skipna=True), ys.max(skipna=True)
                    ymin = cmin if ymin is None else min(ymin, cmin)
                    ymax = cmax if ymax is None else max(ymax, cmax)
                if ymin is None or ymax is None:
                    return
                pad = (abs(ymax - ymin) *
                       0.05) if ymax != ymin else (abs(ymin) *
                                                   0.05 if ymin != 0 else 1e-3)
                fig.update_yaxes(range=[ymin - pad, ymax + pad],
                                 row=row_idx,
                                 col=1)

            adapt_y_for_panel(panel_cols["N"], row_idx=1)
            adapt_y_for_panel(panel_cols["V"], row_idx=2)
            adapt_y_for_panel(panel_cols["I"], row_idx=3)

    return fig, clicked_label


if __name__ == "__main__":
    app.run_server(debug=True)
