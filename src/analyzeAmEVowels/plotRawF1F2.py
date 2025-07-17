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

# --- IMPORTS ---
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Your frequency conversion helpers remain unchanged...

# Gaussian posterior computation
def compute_vowel_posterior(sub, f1_range, f2_range, grid_res=200):
    vowels = sub['vowel'].unique()
    F2, F1 = np.meshgrid(f2_range, f1_range)
    probs = np.zeros((grid_res, grid_res, len(vowels)))

    for idx, vowel in enumerate(vowels):
        row = sub[sub['vowel'] == vowel].iloc[0]
        # Convert mean and SD from Hz to Bark
        f1_mean_bark = hz_to_bark(row.f1_mean)
        f2_mean_bark = hz_to_bark(row.f2_mean)

        # For SD, we assume local linearity in the Bark scale and approximate it:
        f1_sd_bark = abs(hz_to_bark(row.f1_mean + row.f1_sd) - f1_mean_bark)
        f2_sd_bark = abs(hz_to_bark(row.f2_mean + row.f2_sd) - f2_mean_bark)

        f1_term = np.exp(-0.5 * ((hz_to_bark(F1) - f1_mean_bark) / f1_sd_bark*0.7) ** 2) / f1_sd_bark
        f2_term = np.exp(-0.5 * ((hz_to_bark(F2) - f2_mean_bark) / f2_sd_bark*0.7) ** 2) / f2_sd_bark
        probs[:, :, idx] = f1_term * f2_term

    posterior = probs / probs.sum(axis=2, keepdims=True)

    # RGB mapping: #1 red, #2 blue, #3 green
    post_sorted = np.sort(posterior, axis=2)
    img_rgb = np.stack([
        post_sorted[:, :, -1],  # red: highest probability
        post_sorted[:, :, -3] if post_sorted.shape[2] > 2 else np.zeros_like(post_sorted[:, :, -1]),  # green: 3rd highest
        post_sorted[:, :, -2] if post_sorted.shape[2] > 1 else np.zeros_like(post_sorted[:, :, -1])   # blue: 2nd highest
    ], axis=2)

    img_rgb /= img_rgb.max()
    return img_rgb, f1_range, f2_range


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

# --- IMPORTS ---
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def bark_contour(center_f1_b, center_f2_b, radius_bark, n=60, seed=None):
    """
    Return two 1-D arrays (f2_bark_pts, f1_bark_pts) of length n+1.
    Points lie *within* radius_bark (not all on the rim) and are
    ordered so the polygon does not self-intersect.
    """
    rng     = np.random.default_rng(seed)
    angles  = np.linspace(0, 2*np.pi, n, endpoint=False)

    # sample radii ∈ [0.4 r, r]  → ‘ragged’ contour inside the max radius
    radii   = rng.uniform(0.4*radius_bark, radius_bark, size=n)

    f1_pts  = center_f1_b + radii * np.cos(angles)
    f2_pts  = center_f2_b + radii * np.sin(angles)

    # close the polygon by appending the first point again
    f1_pts  = np.append(f1_pts, f1_pts[0])
    f2_pts  = np.append(f2_pts, f2_pts[0])

    return f2_pts, f1_pts

# Gaussian posterior computation
def compute_vowel_posterior(sub, f1_range, f2_range, grid_res=200):
    vowels = sub['vowel'].unique()
    F2, F1 = np.meshgrid(f2_range, f1_range)
    probs = np.zeros((grid_res, grid_res, len(vowels)))

    for idx, vowel in enumerate(vowels):
        row = sub[sub['vowel'] == vowel].iloc[0]
        # Convert mean and SD from Hz to Bark
        f1_mean_bark = hz_to_bark(row.f1_mean)
        f2_mean_bark = hz_to_bark(row.f2_mean)

        # For SD, we assume local linearity in the Bark scale and approximate it:
        f1_sd_bark = abs(hz_to_bark(row.f1_mean + row.f1_sd) - f1_mean_bark)
        f2_sd_bark = abs(hz_to_bark(row.f2_mean + row.f2_sd) - f2_mean_bark)

        f1_term = np.exp(-0.5 * ((hz_to_bark(F1) - f1_mean_bark) / f1_sd_bark*0.7) ** 2) / f1_sd_bark
        f2_term = np.exp(-0.5 * ((hz_to_bark(F2) - f2_mean_bark) / f2_sd_bark*0.7) ** 2) / f2_sd_bark
        probs[:, :, idx] = f1_term * f2_term

    posterior = probs / probs.sum(axis=2, keepdims=True)

    # RGB mapping: #1 red, #2 blue, #3 green
    post_sorted = np.sort(posterior, axis=2)
    img_rgb = np.stack([
        post_sorted[:, :, -1],  # red: highest probability
        post_sorted[:, :, -3] if post_sorted.shape[2] > 2 else np.zeros_like(post_sorted[:, :, -1]),  # green: 3rd highest
        post_sorted[:, :, -2] if post_sorted.shape[2] > 1 else np.zeros_like(post_sorted[:, :, -1])   # blue: 2nd highest
    ], axis=2)

    img_rgb /= img_rgb.max()
    return img_rgb, f1_range, f2_range


def make_figure(sub, f1_bark, f2_bark, scales=('hz', 'bark', 'mel', 'erb'), scale=1, radius_bark=1.1):
    scale_funcs = {'hz': lambda x: x, 'bark': hz_to_bark, 'mel': hz_to_mel, 'erb': hz_to_erb}
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s.upper() for s in scales])

    subplot_axes = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, scale_name in enumerate(scales):
        convert = scale_funcs[scale_name]
        f1_vals = convert(sub.f1_mean)
        f2_vals = convert(sub.f2_mean)

        # Define grid for probabilities
        f1_min, f1_max = f1_vals.min() - 100, f1_vals.max() + 100
        f2_min, f2_max = f2_vals.min() - 100, f2_vals.max() + 100
        grid_res = 250
        f1_range = np.linspace(f1_min, f1_max, grid_res)
        f2_range = np.linspace(f2_min, f2_max, grid_res)

        img_rgb, f1_grid, f2_grid = compute_vowel_posterior(sub.assign(f1_mean=f1_vals, f2_mean=f2_vals),
                                                            f1_range, f2_range, grid_res)

        # Background RGB probability image
        fig.add_trace(
            go.Image(z=(img_rgb * 255).astype(np.uint8),
                     x0=f2_grid.min(), dx=(f2_grid.max()-f2_grid.min())/grid_res,
                     y0=f1_grid.min(), dy=(f1_grid.max()-f1_grid.min())/grid_res,
                     opacity=0.6),
            row=subplot_axes[idx][0], col=subplot_axes[idx][1])

        # Ellipses & Labels
        for _, r in sub.iterrows():
            f2m, f1m = convert(r.f2_mean), convert(r.f1_mean)
            f2sd = abs(convert(r.f2_mean + r.f2_sd) - f2m) * scale * 2
            f1sd = abs(convert(r.f1_mean + r.f1_sd) - f1m) * scale * 2
            ex, ey = ellipse_points(f2m, f1m, f2sd/2, f1sd/2, n=100)

            fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(width=2), showlegend=False),
                          row=subplot_axes[idx][0], col=subplot_axes[idx][1])
            fig.add_trace(go.Scatter(x=[f2m], y=[f1m], mode='text', text=[r.vowel], textfont=dict(size=12),
                                     showlegend=False),
                          row=subplot_axes[idx][0], col=subplot_axes[idx][1])

        f2_cont_b, f1_cont_b = bark_contour(f1_bark, f2_bark, radius_bark,
                                            n=150, seed=42)  # n points, reproducible
        if scale_name == 'hz':
            f1_boundary = bark_to_hz(f1_cont_b)
            f2_boundary = bark_to_hz(f2_cont_b)
            center_x, center_y = bark_to_hz(f2_bark), bark_to_hz(f1_bark)
        else:
            f1_boundary = convert(bark_to_hz(f1_cont_b))
            f2_boundary = convert(bark_to_hz(f2_cont_b))
            center_x, center_y = convert(bark_to_hz(f2_bark)), convert(bark_to_hz(f1_bark))

        fig.add_trace(
            go.Scatter(x=f2_boundary,
                       y=f1_boundary,
                       mode='lines',
                       line=dict(color='red'),
                       showlegend=False),
            row=subplot_axes[idx][0],
            col=subplot_axes[idx][1]
        )

        fig.add_trace(go.Scatter(x=[center_x], y=[center_y], mode='markers', marker=dict(color='black', size=10),
                                 showlegend=False),
                      row=subplot_axes[idx][0], col=subplot_axes[idx][1])


        fig.update_yaxes(autorange='reversed', row=subplot_axes[idx][0], col=subplot_axes[idx][1])
        fig.update_xaxes(autorange='reversed', row=subplot_axes[idx][0], col=subplot_axes[idx][1])

    fig.update_layout(height=900, width=1200, margin=dict(l=30, r=10, t=40, b=20))
    return fig

# -------------- DASH APP --------------

file_path = '../../input_data/vowel_stats.txt'  # update path as needed
gender = 'PBm'  # or 'cbm', etc.
scales = ('hz', 'bark', 'mel', 'erb')
scale = 2

sub = read_vowel_stats(file_path, gender=gender)

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Vowel Space Interactive (Dash + Plotly)"),
    html.Div([
        html.Label("F1 (Bark):"),
        dcc.Slider(
            id='f1-slider', min=2, max=20, step=0.05, value=6.0,
            marks={i: f"{i}" for i in range(2, 21, 2)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Label("F2 (Bark):"),
        dcc.Slider(
            id='f2-slider', min=2, max=20, step=0.05, value=12.0,
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
