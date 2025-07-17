import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEFAULT_CSV_PATH = "vectors.csv"

st.set_page_config(
    page_title="Stream Cross Section Visualiser",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Stream Cross Section Visualiser")
st.subheader("Survey Data")

# ── IMPORT / LOAD CSV ──────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

uploaded = st.sidebar.file_uploader("Import Vectors CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_csv(DEFAULT_CSV_PATH)

# ── ROTATION STATE ─────────────────────────────────────────────────────────────
if "rot_angle" not in st.session_state:
    st.session_state.rot_angle = 0

# ── SIDEBAR CONTROLS ───────────────────────────────────────────────────────────
# PCA Cross‐Section
st.sidebar.header("PCA Cross-Section")
show_fit   = st.sidebar.checkbox("Show PCA fit", value=True)
all_pts    = sorted(set(df['vector point 1']) | set(df['vector point 2']))
default_fit = [
    'C','F','G','H','I','J','K','L',
    'P','Q','R','T','U','V','W','X','Y',
    'a','b','d','f','g','h','i'
]
initial    = [p for p in default_fit if p in all_pts]
fit_pts    = st.sidebar.multiselect("Points to include", all_pts, default=initial)
flip_cross = st.sidebar.checkbox("Flip cross‐section horizontally")

# Rotate diagram
st.sidebar.header("Rotate diagram")
pivot_pt = st.sidebar.selectbox("Pivot point", all_pts)

# ±1° buttons
col1, col2, col3 = st.sidebar.columns([1,2,1])
if col1.button("−1°"):
    st.session_state.rot_angle = (
        st.session_state.rot_angle - 1 if st.session_state.rot_angle > -180 else 180
    )
if col3.button("+1°"):
    st.session_state.rot_angle = (
        st.session_state.rot_angle + 1 if st.session_state.rot_angle <  180 else -180
    )

# single slider bound –180→180, driving the same key
st.sidebar.number_input(
    "Rotation (°)",
    min_value=-180, max_value=180, step=1,
    key="rot_angle"
)

# Map View Controls
st.sidebar.header("Map View Controls")
map_token = st.sidebar.text_input(
    "Mapbox token",
    type="password",
    value="pk.eyJ1IjoiY2RvbW90b3IiLCJhIjoiY21kNWlwbDVkMDAwdjJ4cTg0dDhyajg1diJ9.Z0yrFi2lmJpLDyJ8Gb9qaQ"
)
ref_pt = st.sidebar.selectbox("Reference point", all_pts, index=0)
lat0   = st.sidebar.number_input("Reference latitude", format="%.6f", value=-25.275758)
lon0   = st.sidebar.number_input("Reference longitude", format="%.6f", value=152.508823)
zoom   = st.sidebar.slider("Map zoom", 1, 20, 18)

# ── APPLY ROTATION TO DATAFRAME ────────────────────────────────────────────────
# Copy original, apply rot_angle to the direction column
df2 = df.copy()
if st.session_state.rot_angle != 0:
    df2["vector degrees at 3 point angle"] = (
        df2["vector degrees at 3 point angle"] + st.session_state.rot_angle
    ) % 360

# ── EDITABLE TABLE ─────────────────────────────────────────────────────────────
edited = st.data_editor(
    df2,
    num_rows="dynamic",
    use_container_width=True,
    height=300,
)

# Save / Download edited CSV
if not uploaded and st.sidebar.button("Save edits to default CSV"):
    edited.to_csv(DEFAULT_CSV_PATH, index=False)
    st.sidebar.success(f"Saved edits to `{DEFAULT_CSV_PATH}`")
if uploaded:
    st.sidebar.download_button(
        "Download edited CSV",
        edited.to_csv(index=False).encode(),
        "edited_vectors.csv",
        "text/csv"
    )

# ── COMPUTE LOCAL COORDS ───────────────────────────────────────────────────────
def compute_coords(df):
    coords = { df.iloc[0]['vector point 1']: (0.0, 0.0) }
    rem = df.set_index('Vector index').copy()
    while True:
        progressed = False
        for idx, r in rem.iterrows():
            p1, p2 = r['vector point 1'], r['vector point 2']
            θ, L    = math.radians(r['vector degrees at 3 point angle']), r['vector length']
            vx, vy  = L*math.cos(θ), L*math.sin(θ)
            if p1 in coords and p2 not in coords:
                x, y = coords[p1]
                coords[p2] = (x + vx, y + vy)
                rem = rem.drop(idx); progressed = True
            elif p2 in coords and p1 not in coords:
                x, y = coords[p2]
                coords[p1] = (x - vx, y - vy)
                rem = rem.drop(idx); progressed = True
        if not progressed:
            break
    if not rem.empty:
        raise RuntimeError("Unresolved vectors")
    return coords

coords = compute_coords(edited)

# ── PCA & CROSS-SECTION DATAFRAME ──────────────────────────────────────────────
cross_df = pd.DataFrame()
if show_fit and len(fit_pts) >= 2:
    XY       = np.array([coords[p] for p in fit_pts])
    pca      = PCA(n_components=1).fit(XY)
    center, dir_vec = pca.mean_, pca.components_[0]
    dists    = (XY - center).dot(dir_vec)

    RL_map = {
        r['vector point 2']: r['RL']
        for _, r in edited.iterrows()
        if pd.notna(r.get('RL'))
    }
    RLs = [RL_map.get(p, np.nan) for p in fit_pts]

    cross_df = pd.DataFrame({
        "xs_point_index":       np.arange(1, len(fit_pts)+1),
        "xs_survey_point_name": fit_pts,
        "xs_distance":          dists,
        "RL":                   RLs
    })
    if flip_cross:
        cross_df["xs_distance"] *= -1
    cross_df = cross_df.sort_values("xs_distance").reset_index(drop=True)

    st.sidebar.download_button(
        "Download cross_section.csv",
        data=cross_df.to_csv(index=False).encode(),
        file_name="cross_section.csv",
        mime="text/csv"
    )

# ── PLAN VIEW MAP ─────────────────────────────────────────────────────────────
st.subheader("Plan view (satellite)")
if not map_token:
    st.warning("Enter your Mapbox token above to render the satellite map.")
else:
    # helper to go from local coords → lat/lon
    x_ref, y_ref = coords[ref_pt]
    mlat = 111000
    mlon = 111000 * math.cos(math.radians(lat0))
    def to_latlon(pt):
        x, y = coords[pt]
        return lat0 + (y - y_ref)/mlat, lon0 + (x - x_ref)/mlon

    # auto‐center on mean of all points
    lats = [to_latlon(pt)[0] for pt in coords]
    lons = [to_latlon(pt)[1] for pt in coords]
    mid_lat, mid_lon = np.mean(lats), np.mean(lons)

    fig = go.Figure()
    fig.update_layout(
        mapbox=dict(
            accesstoken=map_token,
            style="satellite",
            center={"lat": mid_lat, "lon": mid_lon},
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="closest",
        height=700
    )

    # draw vectors + hover‐markers
    for idx, r in edited.iterrows():
        p1, p2 = r['vector point 1'], r['vector point 2']
        lat1, lon1 = to_latlon(p1)
        lat2, lon2 = to_latlon(p2)
        info = (
            f"Vector {idx}: {p1}→{p2}<br>"
            f"Len: {r['vector length']:.2f}<br>"
            f"Ang: {r['vector degrees at 3 point angle']}°"
        )
        fig.add_trace(go.Scattermapbox(
            lat=[lat1, lat2], lon=[lon1, lon2],
            mode="lines", line=dict(color=r['vector colour'], width=2),
            hoverinfo="none", name=f"{p1}→{p2}"
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[(lat1+lat2)/2], lon=[(lon1+lon2)/2],
            mode="markers", marker=dict(size=8, opacity=0),
            hoverinfo="text", hovertext=info, showlegend=False
        ))

    # draw points + hover
    for pt in coords:
        lat, lon = to_latlon(pt)
        info = f"Point {pt}<br>Lat {lat:.6f}<br>Lon {lon:.6f}"
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode="markers+text", marker=dict(size=6, color="white"),
            text=[pt], textposition="top center",
            hoverinfo="text", hovertext=info,
            name=f"Pt {pt}"
        ))

    # PCA‐fit line on map
    if show_fit and not cross_df.empty:
        dmin, dmax = cross_df.xs_distance.min(), cross_df.xs_distance.max()
        pad = (dmax - dmin)*0.05
        t   = np.linspace(dmin-pad, dmax+pad, 200)
        line_xy = center + np.outer(t, dir_vec)

        # constant bearing
        bearing = (math.degrees(math.atan2(dir_vec[0], dir_vec[1])) + 360) % 360

        lat_line, lon_line = [], []
        for x, y in line_xy:
            lat_line.append(lat0 + (y-y_ref)/mlat)
            lon_line.append(lon0 + (x-x_ref)/mlon)

        cum_dist   = np.abs(t)
        customdata = np.vstack([cum_dist, np.full_like(cum_dist, bearing)]).T

        fig.add_trace(go.Scattermapbox(
            lat=lat_line, lon=lon_line,
            mode="lines+markers",
            line=dict(color="black", width=4),
            marker=dict(size=8, symbol="diamond"),
            customdata=customdata,
            hovertemplate=(
                "Along‐line: %{customdata[0]:.1f} m<br>"
                "Bearing: %{customdata[1]:.1f}°"
                "<extra></extra>"
            ),
            name="PCA fit"
        ))

    st.plotly_chart(fig, use_container_width=True)

# ── 2D PLAN VIEW (no background) ───────────────────────────────────────────────
st.subheader("Plan view (2D vectors)")
fig2d = go.Figure()

# draw vectors + hidden hover‐markers
for idx, r in edited.iterrows():
    p1, p2 = r['vector point 1'], r['vector point 2']
    x1, y1 = coords[p1]; x2, y2 = coords[p2]
    info = (
        f"Vector {idx}: {p1}→{p2}<br>"
        f"Len: {r['vector length']:.2f}<br>"
        f"Ang: {r['vector degrees at 3 point angle']}°"
    )
    fig2d.add_trace(go.Scatter(
        x=[x1, x2], y=[y1, y2],
        mode="lines", line=dict(color=r['vector colour'], width=2),
        hoverinfo="none", name=f"{p1}→{p2}"
    ))
    fig2d.add_trace(go.Scatter(
        x=[(x1+x2)/2], y=[(y1+y2)/2],
        mode="markers", marker=dict(size=8, opacity=0),
        hoverinfo="text", hovertext=info, showlegend=False
    ))

# draw points + hover
for pt, (x, y) in coords.items():
    info = f"Point {pt}<br>X: {x:.2f}<br>Y: {y:.2f}"
    fig2d.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers+text", marker=dict(size=6),
        text=[pt], textposition="top center",
        hoverinfo="text", hovertext=info,
        showlegend=False
    ))

# PCA fit in 2D
if show_fit and not cross_df.empty:
    pad        = (cross_df.xs_distance.max() - cross_df.xs_distance.min())*0.05
    t          = np.linspace(cross_df.xs_distance.min()-pad,
                              cross_df.xs_distance.max()+pad, 200)
    line_xy    = center + np.outer(t, dir_vec)
    fig2d.add_trace(go.Scatter(
        x=line_xy[:,0], y=line_xy[:,1],
        mode="lines", line=dict(color="black", dash="dash", width=3),
        name="PCA fit"
    ))

fig2d.update_layout(
    xaxis=dict(title="X", constrain="domain"),
    yaxis=dict(title="Y", scaleanchor="x"),
    margin=dict(l=20, r=20, t=20, b=20),
    height=600
)
st.plotly_chart(fig2d, use_container_width=True)

# ── CROSS-SECTION & PROFILE ───────────────────────────────────────────────────
if not cross_df.empty:
    st.subheader("Cross-Section Data")
    st.dataframe(cross_df)

    st.subheader("Cross-Section Profile (RL vs Distance)")
    figXs = go.Figure()
    figXs.add_trace(go.Scatter(
        x=cross_df.xs_distance, y=cross_df.RL,
        mode="lines+markers", line=dict(color="black"),
        hoverinfo="x+y", name="Profile"
    ))
    for _, row in cross_df.iterrows():
        hover = (
            f"{row.xs_survey_point_name}<br>"
            f"Dist: {row.xs_distance:.2f}<br>"
            f"RL: {row.RL:.2f}"
        )
        figXs.add_trace(go.Scatter(
            x=[row.xs_distance], y=[row.RL],
            mode="markers+text", marker=dict(size=6),
            text=[row.xs_survey_point_name], textposition="top center",
            hoverinfo="text", hovertext=hover,
            name=row.xs_survey_point_name
        ))
    figXs.update_layout(
        hovermode="x unified",
        xaxis_title="xs_distance",
        yaxis_title="RL",
        margin=dict(l=20, r=20, t=20, b=20),
        height=400
    )
    st.plotly_chart(figXs, use_container_width=True)
