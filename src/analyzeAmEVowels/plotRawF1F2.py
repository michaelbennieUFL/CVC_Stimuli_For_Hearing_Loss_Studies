import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

# Frequency conversion helpers
def hz_to_erb(f):   return 214 * np.log10(0.00437 * f + 1)
def hz_to_mel(f):   return 1127.01048 * np.log(1 + f / 700)
def hz_to_bark(f):  return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7000) ** 2)
def bark_to_hz(b):  return 1960 * (b + 0.53) / (26.28 - b)

# Precompute ellipse points (parametric form)
def ellipse_points(cx, cy, rx, ry, n=100):
    theta = np.linspace(0, 2 * np.pi, n)
    x = cx + rx * np.cos(theta)
    y = cy + ry * np.sin(theta)
    return x, y

def read_vowel_stats(file_path, gender='m', max_sd=5000):
    data_dict = {}
    measures = ['Duration', 'F0', 'F1', 'F2', 'F3', 'F4']
    with open(file_path, 'r') as f:
        for line in f:
            tok = line.strip().split()
            if len(tok) < 8:
                continue
            vowel, group, mean, sd, *_, measure = tok
            if measure not in measures:
                continue
            key = (vowel, group)
            data_dict.setdefault(key, {})[f"{measure.lower()}_mean"] = float(mean)
            data_dict[key][f"{measure.lower()}_sd"]   = float(sd)
    df = (pd.DataFrame.from_dict(data_dict, orient='index')
          .rename_axis(['vowel', 'group'])
          .reset_index())
    sub = (df[df['group'] == gender]
           .query('f1_sd < @max_sd and f2_sd < @max_sd'))
    return sub

def make_figure(sub, f1_bark, f2_bark, scales=('hz', 'bark', 'mel', 'erb'), scale=1, radius_bark=1.0):
    scale_funcs = {
        'hz':   lambda x: x,
        'bark': hz_to_bark,
        'mel':  hz_to_mel,
        'erb':  hz_to_erb
    }

    fig = make_subplots(rows=2, cols=2, subplot_titles=[s.upper() for s in scales],
                        shared_xaxes=False, shared_yaxes=False)
    subplot_axes = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, scale_name in enumerate(scales):
        convert = scale_funcs[scale_name]
        f1_vals, f2_vals = [], []

        # -- Vowel ellipses & labels
        for _, r in sub.iterrows():
            f2m, f1m = convert(r.f2_mean), convert(r.f1_mean)
            f2sd = abs(convert(r.f2_mean + r.f2_sd) - f2m) * scale * 2
            f1sd = abs(convert(r.f1_mean + r.f1_sd) - f1m) * scale * 2
            ex, ey = ellipse_points(f2m, f1m, f2sd / 2, f1sd / 2, n=100)
            ell = go.Scatter(x=ex, y=ey, mode='lines', line=dict(width=2), showlegend=False)
            txt = go.Scatter(x=[f2m], y=[f1m], mode='text', text=[r.vowel], showlegend=False)
            fig.add_trace(ell, row=subplot_axes[idx][0], col=subplot_axes[idx][1])
            fig.add_trace(txt, row=subplot_axes[idx][0], col=subplot_axes[idx][1])
            f1_vals.append(f1m)
            f2_vals.append(f2m)

        # -- Boundary (circle) and center
        n_circ = 180
        angles = np.linspace(0, 2 * np.pi, n_circ)
        f1_circle_b = f1_bark + radius_bark * np.cos(angles)
        f2_circle_b = f2_bark + radius_bark * np.sin(angles)
        if scale_name == 'hz':
            f1_boundary = bark_to_hz(f1_circle_b)
            f2_boundary = bark_to_hz(f2_circle_b)
            center_x = [bark_to_hz(f2_bark)]
            center_y = [bark_to_hz(f1_bark)]
        else:
            f1_boundary = convert(bark_to_hz(f1_circle_b))
            f2_boundary = convert(bark_to_hz(f2_circle_b))
            center_x = [convert(bark_to_hz(f2_bark))]
            center_y = [convert(bark_to_hz(f1_bark))]
        boundary = go.Scatter(
            x=f2_boundary, y=f1_boundary,
            mode='lines', line=dict(color='red'), name='±1 Bark', showlegend=False
        )
        center = go.Scatter(
            x=center_x, y=center_y, mode='markers', marker=dict(size=10, color='black'),
            name='Center', showlegend=False
        )
        fig.add_trace(boundary, row=subplot_axes[idx][0], col=subplot_axes[idx][1])
        fig.add_trace(center, row=subplot_axes[idx][0], col=subplot_axes[idx][1])

        # Reverse axes for "vowel-space" orientation
        fig.update_yaxes(autorange='reversed', row=subplot_axes[idx][0], col=subplot_axes[idx][1])
        fig.update_xaxes(autorange='reversed', row=subplot_axes[idx][0], col=subplot_axes[idx][1])

    fig.update_layout(height=900, width=1200,
                      margin=dict(l=30, r=10, t=40, b=20))
    return fig

# -------------- DASH APP --------------

file_path = '../../input_data/vowel_stats.txt'  # update path as needed
gender = 'PBm'  # or 'cbm', etc.
scales = ('hz', 'bark', 'mel', 'erb')
scale = 1.8

sub = read_vowel_stats(file_path, gender=gender)

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Vowel Space Interactive (Dash + Plotly)"),
    html.Div([
        html.Label("F1 (Bark):"),
        dcc.Slider(
            id='f1-slider', min=2, max=20, step=0.1, value=6.0,
            marks={i: f"{i}" for i in range(2, 21, 2)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Label("F2 (Bark):"),
        dcc.Slider(
            id='f2-slider', min=2, max=20, step=0.1, value=12.0,
            marks={i: f"{i}" for i in range(2, 21, 2)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'width': '70%', 'margin': 'auto'}),
    dcc.Graph(id='vowel-plot', style={'height': '900px', 'width': '1200px', 'margin': 'auto'}),
])

@app.callback(
    Output('vowel-plot', 'figure'),
    [Input('f1-slider', 'value'),
     Input('f2-slider', 'value')]
)
def update_figure(f1_bark, f2_bark):
    return make_figure(sub, f1_bark, f2_bark, scales=scales, scale=scale)

if __name__ == "__main__":
    app.run(debug=True)
