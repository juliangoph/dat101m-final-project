import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import json
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Load preprocessed data (you'll need to adjust paths)
df = pd.read_csv(
    "data/processed_philippine_cities_monthly.csv",
)
gdf = gpd.read_file("data/phl_adm_simple_maps.gpkg")
geojson_data = json.loads(gdf.to_json())

# identify quantitative columns for aggregation
quantitative_columns = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "HLI",
]

MODEBAR_REMOVE = [
    "zoom",
    "zoomIn",
    "zoomOut",
    "autoScale",
    "resetScale",
    "pan",
    "select",
    "lasso",
    "reset",
    "toimage",
]


def process_spatial_aggregation(df, group_cols, gdf, crs, adm_col="adm1"):
    # Aggregate by specified grouping columns
    df_agg = df.groupby(group_cols).mean().reset_index()

    # Convert to GeoDataFrame and reproject
    df_agg["geometry"] = gpd.points_from_xy(df_agg["longitude"], df_agg["latitude"])
    gdf_agg = gpd.GeoDataFrame(df_agg, geometry="geometry", crs=crs).to_crs(crs)

    # Perform spatial join with administrative boundaries
    merged_gdf = gpd.sjoin(
        gdf_agg, gdf[["name", "geometry"]], how="left", predicate="intersects"
    )
    merged_gdf = merged_gdf.rename(columns={"name": adm_col})

    df_final = (
        merged_gdf.groupby(
            [adm_col] + [col for col in group_cols if col != "city_name"]
        )[quantitative_columns]
        .mean()
        .reset_index()
    )

    return df_final


# Process monthly decadal aggregation
gdf_month_decadal_adm1 = process_spatial_aggregation(
    df, ["decade", "month", "city_name"], gdf, gdf.crs
)
gdf_month_decadal_adm1["decade"] = gdf_month_decadal_adm1["decade"].astype(
    int, errors="ignore"
)

# Process decadal aggregation
df["decade"] = df["year"] // 10 * 10  # Ensure decade column is created
gdf_decadal_adm1 = process_spatial_aggregation(
    df, ["decade", "city_name"], gdf, gdf.crs
)
gdf_decadal_adm1["decade"] = gdf_decadal_adm1["decade"].astype(int, errors="ignore")


# create dash app
app = dash.Dash(external_stylesheets=[dbc.themes.YETI, dbc.icons.BOOTSTRAP])
server = app.server  # Required for deployment with Gunicorn

steps = [
    html.Div(
        [
            html.P("Use the decade slider below the map to select a time period."),
            html.Img(
                src="/assets/step1.png", style={"width": "100%", "borderRadius": "12px"}
            ),
        ]
    ),
    html.Div(
        [
            html.P(
                'Click on a region to view localized statistics. Use the "All Regions" button to reset the map.'
            ),
            html.Img(
                src="/assets/step2.png", style={"width": "100%", "borderRadius": "12px"}
            ),
        ]
    ),
    html.Div(
        [
            html.P("Click the play button to animate changes across decades."),
            html.Img(
                src="/assets/step3.png", style={"width": "100%", "borderRadius": "12px"}
            ),
        ]
    ),
    html.Div(
        [
            html.P(
                "Hover over charts for detailed values. Use the filter panel on the right to explore data further."
            ),
            html.Img(
                src="/assets/step4.png", style={"width": "100%", "borderRadius": "12px"}
            ),
        ]
    ),
    html.Div(
        html.P("You're all set! Click Finish to start exploring the dashboard."),
    ),
]

app.layout = html.Div(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.NavbarBrand(
                                        "DAT101M - Final Project", className="ms-2"
                                    )
                                ),
                            ],
                        ),
                        className="navbar-brand",
                    ),
                    dbc.Nav(
                        [
                            dbc.Button(
                                "Launch Onboarding",
                                id="open",
                                color="light",
                                outline=True,
                                size="sm",
                                className="mb-2 mb-md-0 me-0 me-md-2",
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    html.I(className="bi bi-github"),
                                    href="https://github.com/juliangoph/dat101m-final-project",
                                    className="d-flex align-items-center",
                                )
                            ),
                        ],
                        className=(
                            "d-flex flex-column flex-md-row align-items-center "
                            "justify-content-center justify-content-md-end ms-md-auto"
                        ),
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            sticky="top",
        ),
        # Main content
        dbc.Container(
            [
                html.Div(
                    [
                        dbc.Container(
                            [
                                html.H1(
                                    "Urban Heat Index",
                                    className="display-3",
                                ),
                                html.P(
                                    "Head Index across the Philippines between 1950-2025",
                                    className="lead",
                                ),
                            ],
                            fluid=True,
                            className="py-3",
                        ),
                    ],
                    className="p-3",
                ),
                # Main content (Map + Charts)
                dbc.Row(
                    [
                        # Left Column: Map
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dcc.Slider(
                                            id="year-slider",
                                            min=int(gdf_decadal_adm1["decade"].min()),
                                            max=int(gdf_decadal_adm1["decade"].max()),
                                            value=int(gdf_decadal_adm1["decade"].min()),
                                            marks={
                                                str(int(decade)): str(int(decade))
                                                for decade in gdf_decadal_adm1[
                                                    "decade"
                                                ].unique()
                                            },
                                            step=None,
                                        ),
                                    ],
                                    className="slider-container",
                                ),
                                dcc.Interval(
                                    id="play-interval",
                                    interval=1000,
                                    n_intervals=0,
                                    disabled=True,
                                ),
                                dcc.Store(id="play-state", data={"playing": False}),
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Graph(
                                                id="choropleth-map", responsive=True
                                            ),
                                            className="map-frame",
                                        ),
                                        dbc.Button(
                                            "All Regions",
                                            color="secondary",
                                            id="reset-button",
                                            n_clicks=0,
                                            className="map-controls me-1",
                                        ),
                                    ],
                                    className="map-container",
                                ),
                            ],
                            width=6,
                            xs=12,
                            sm=12,
                            md=6,
                            lg=6,
                            className="choropleth-container",
                        ),
                        # Right Column: Charts
                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(id="line-chart", responsive=True),
                                    className="chart-container",
                                ),
                                html.Div(
                                    dcc.Graph(id="bar-chart", responsive=True),
                                    className="chart-container",
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="line-chart-hli-monthly", responsive=True
                                    ),
                                    className="chart-container",
                                ),
                            ],
                            width=6,
                            xs=12,
                            sm=12,
                            md=6,
                            lg=6,
                        ),
                    ]
                ),
            ]
        ),
        # Floating play button
        dbc.Button(
            html.I(className="bi bi-play-fill fs-4"),
            id="play-button",
            color="primary",
            className="rounded-circle border-0 d-flex align-items-center justify-content-center",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Onboarding")),
                dbc.ModalBody(
                    id="onboarding-body", children=steps[0], className="fade-text"
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("Back", id="back", n_clicks=0, disabled=True),
                        dbc.Button("Next", id="next", n_clicks=0),
                    ]
                ),
            ],
            id="modal",
            is_open=False,
            backdrop="static",  # prevent clicking outside to close
            keyboard=False,  # prevent ESC to close
        ),
        # Hidden store to keep track of step index
        html.Div(id="step-store", children="0", style={"display": "none"}),
    ],
)


def get_selected_region(clickData):
    if not clickData or "points" not in clickData or len(clickData["points"]) == 0:
        return "All Regions"
    return clickData["points"][0]["location"]


# Helper function for layout
def apply_chart_layout(
    fig, title, x_label="Year", y_label="Value", df=None, x_col=None, y_col=None
):
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        autosize=True,  # Ensures the figure resizes dynamically
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.5,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=80),  # Adjust margin for legend placement
        xaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,  # Keeps grid lines visible
            zeroline=True,  # Keeps the zero line visible
        ),
        dragmode=False,  # Disables zoom & pan
        modebar_remove=MODEBAR_REMOVE,
        clickmode="none",
    )

    if df is not None and x_col is not None:
        if x_col in df.columns:
            fig.update_layout(
                xaxis=dict(
                    type=(
                        "category" if df[x_col].dtype == "O" else "linear"
                    ),  # Categorical vs numeric handling
                    tickmode="array",
                    tickvals=df[x_col]
                    .unique()
                    .tolist(),  # Ensure all x-axis values are shown
                )
            )

    if df is not None and y_col is not None:
        if y_col in df.columns:
            fig.update_layout(
                yaxis=dict(
                    autorange=True,
                    tickmode="array",
                    tickvals=(
                        df[y_col].unique().tolist() if df[y_col].dtype == "O" else None
                    ),  # Only force ticks for categorical values
                )
            )


# calculate the aggregate on all regions
all_regions_decadal_avg = (
    gdf_decadal_adm1.groupby(["decade"])[quantitative_columns].mean().reset_index()
)
all_regions_monthly_avg = (
    gdf_month_decadal_adm1.groupby(["month", "decade"])[quantitative_columns]
    .mean()
    .reset_index()
)

# ✅ Precompute datasets per year (instead of slicing on every slider change)
preloaded_yearly_data = {
    year: gdf_decadal_adm1[gdf_decadal_adm1["decade"] == year].copy()
    for year in gdf_decadal_adm1["decade"].unique()
}


def create_choropleth_layer(
    df,
    color_scale,
    border_color="gainsboro",
    show_colorbar=False,
    geojson_data=None,
    coloraxis="coloraxis",
):
    global_hli_min = gdf_decadal_adm1["HLI"].min()
    global_hli_max = gdf_decadal_adm1["HLI"].max()

    choropleth = px.choropleth(
        df,
        geojson=geojson_data,
        locations="adm1",
        featureidkey="properties.name",
        color="HLI",
        color_continuous_scale=color_scale,
        range_color=[global_hli_min, global_hli_max],
    )
    choropleth.update_traces(
        marker_line_color=border_color,
        marker_line_width=1,
        showscale=show_colorbar,
        coloraxis=coloraxis,
    )
    return choropleth


# Callbacks
@app.callback(
    Output("choropleth-map", "figure"),
    [
        Input("year-slider", "value"),
        Input("choropleth-map", "clickData"),
        Input("reset-button", "n_clicks"),  # Reset trigger
    ],
)
def update_choropleth(selected_year, clickData, n_clicks):
    triggered_id = ctx.triggered_id

    # Filter data for the selected year
    filtered_df = preloaded_yearly_data.get(selected_year, gdf_decadal_adm1.copy())

    # Determine the selected region, reset if the reset button is clicked
    if triggered_id == "reset-button":
        selected_region = "All Regions"  # Force reset to ALL REGIONS
    else:
        selected_region = get_selected_region(
            clickData
        )  # Only run if reset wasn't clicked

    # Simulated low-opacity color scale (pale)
    highlight_colorscale = [(0.0, "#0000FF"), (0.5, "#FFFF00"), (1.0, "#FF0000")]

    # Highlight color scale (vivid)
    dimmed_colorscale = [
        (0.0, "#ccccff"),
        (0.5, "#ffffcc"),
        (1.0, "#ffcccc"),
    ]

    # Create Choropleth Map
    fig = create_choropleth_layer(
        filtered_df,
        highlight_colorscale,
        border_color="gainsboro",
        geojson_data=geojson_data,
    )

    # Highlight selected region
    if selected_region != "All Regions":
        non_highlight_df = filtered_df[filtered_df["adm1"] != selected_region]

        non_highlight_df = create_choropleth_layer(
            non_highlight_df,
            dimmed_colorscale,
            border_color="gainsboro",
            geojson_data=geojson_data,
            coloraxis="coloraxis2",
        )

        for trace in non_highlight_df.data:
            fig.add_trace(trace)

        # Highlighted region
        highlight_df = filtered_df[filtered_df["adm1"] == selected_region]
        highlight_fig = create_choropleth_layer(
            highlight_df,
            highlight_colorscale,
            border_color="black",
            geojson_data=geojson_data,
        )
        for trace in highlight_fig.data:
            fig.add_trace(trace)

    # Remove background map and maximize plot area
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
    )

    # Maximize the size of the map in the canvas
    fig.update_layout(
        coloraxis_colorbar=dict(title="HLI"),
        coloraxis2=dict(
            colorscale=dimmed_colorscale,
            showscale=False,  # Hide secondary color
        ),
        uirevision=str(n_clicks),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        dragmode=False,
        modebar_remove=MODEBAR_REMOVE,
    )

    return fig


@app.callback(
    [
        Output("line-chart", "figure"),
        Output("bar-chart", "figure"),
        Output("line-chart-hli-monthly", "figure"),
    ],
    [
        Input("year-slider", "value"),
        Input("choropleth-map", "clickData"),
        Input("reset-button", "n_clicks"),
    ],
)
def update_charts(selected_year, clickData, _):
    triggered_id = ctx.triggered_id

    # Force reset if reset-button is clicked
    if triggered_id == "reset-button":
        selected_region = "All Regions"
    else:
        selected_region = get_selected_region(clickData)

    # Use precomputed averages instead of redundant groupby calculations
    adm1_df = (
        all_regions_decadal_avg
        if selected_region == "All Regions"
        else gdf_decadal_adm1[gdf_decadal_adm1["adm1"] == selected_region].copy()
    )
    adm1_month_df = (
        all_regions_monthly_avg
        if selected_region == "All Regions"
        else gdf_month_decadal_adm1[
            gdf_month_decadal_adm1["adm1"] == selected_region
        ].copy()
    )

    # LINE CHART: HLI Trends + Horizontal Line Averages
    fig_line = go.Figure()

    # Original HLI Trend
    fig_line.add_trace(
        go.Scatter(
            x=adm1_df["decade"],
            y=adm1_df["HLI"],
            mode="lines+markers",
            name="HLI",
            line=dict(width=2),
        )
    )

    # Add Vertical Line at Selected Year
    fig_line.add_shape(
        dict(
            type="line",
            x0=selected_year,  # Position on x-axis
            x1=selected_year,  # Same x position to form a vertical line
            y0=adm1_df["HLI"].min(),  # Start from the lowest value in HLI
            y1=adm1_df["HLI"].max(),  # End at the highest value in HLI
            line=dict(width=0.5, dash="dot"),  # Customize color and style
        )
    )

    # Define x range for horizontal lines (full decade span)
    x_range = [adm1_df["decade"].min(), adm1_df["decade"].max()]

    three_decade_avg = adm1_df["HLI"].tail(3).mean()
    five_decade_avg = adm1_df["HLI"].tail(5).mean()
    seven_decade_avg = adm1_df["HLI"].mean()

    # Add 3-decade average line (bright magenta)
    fig_line.add_trace(
        go.Scatter(
            x=x_range,
            y=[three_decade_avg, three_decade_avg],
            mode="lines",
            line=dict(dash="dash", width=2, color="#D62728"),
            showlegend=False,
        )
    )
    fig_line.add_annotation(
        x=x_range[1],  # Right end of the line
        y=three_decade_avg,
        text="3-Decade Avg",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=12, color="#D62728"),
        xshift=5,  # Offset to the right
    )

    # Add 5-decade average line (deep purple)
    fig_line.add_trace(
        go.Scatter(
            x=x_range,
            y=[five_decade_avg, five_decade_avg],
            mode="lines",
            line=dict(dash="dash", width=2, color="#9467BD"),
            showlegend=False,
        )
    )
    fig_line.add_annotation(
        x=x_range[1],
        y=five_decade_avg,
        text="5-Decade Avg",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=12, color="#9467BD"),
        xshift=5,
    )

    # Add 7-decade average line (dark green)
    fig_line.add_trace(
        go.Scatter(
            x=x_range,
            y=[seven_decade_avg, seven_decade_avg],
            mode="lines",
            line=dict(dash="dash", width=2, color="#2CA02C"),
            showlegend=False,
        )
    )
    fig_line.add_annotation(
        x=x_range[1],
        y=seven_decade_avg,
        text="7-Decade Avg",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=12, color="#2CA02C"),
        xshift=5,
    )

    # Layout settings
    apply_chart_layout(
        fig_line,
        f"HLI Trends ({selected_region})",
        "Year",
        "Heat Load Index (HLI)",
        x_col="decade",
        df=adm1_df,
    )

    fig_line.update_layout(showlegend=False)

    # BAR CHART: Temperature and Wind Speed for Selected Year & Region
    # Create figure with secondary y-axis
    fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

    # Set custom data for hover tooltips
    customdata_temp_minmax = adm1_df[
        ["temperature_2m_min", "temperature_2m_max"]
    ].values
    customdata_apparent_minmax = adm1_df[
        ["apparent_temperature_min", "apparent_temperature_max"]
    ].values

    # Fake trace for first group
    fig_bar.add_trace(
        go.Scatter(
            x=[None],
            y=[None],  # No data, just for legend
            mode="lines",
            name="<b>Temperature Metrics</b>",  # Bold section title
            line=dict(color="rgba(0,0,0,0)"),  # Invisible
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Fake trace for second group
    fig_bar.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="<b>Environmental Factors</b>",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig_bar.add_trace(
        go.Bar(
            x=adm1_df["decade"],
            y=adm1_df["temperature_2m_mean"],
            name="Mean Temp",
            marker=dict(color="#4DAF4A"),
            customdata=customdata_temp_minmax,
            hovertemplate=(
                "<b>Decade: %{x}</b><br>"
                "Mean Temp: %{y:.2f}°C<br>"
                "Min Temp: %{customdata[0]:.2f}°C<br>"
                "Max Temp: %{customdata[1]:.2f}°C<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Add Scatter Line (Secondary Y-Axis: Wind Speed)
    fig_bar.add_trace(
        go.Scatter(
            x=adm1_df["decade"],
            y=adm1_df["wind_speed_10m_max"],
            mode="markers+lines",
            name="Wind Speed",
            line=dict(dash="dot", width=2, color="#1F78B4"),
        ),
        secondary_y=True,  # Assign this trace to the secondary y-axis
    )

    fig_bar.add_trace(
        go.Bar(
            x=adm1_df["decade"],
            y=adm1_df["apparent_temperature_mean"],
            name="Apparent Temp",
            marker=dict(color="#984EA3"),
            customdata=customdata_apparent_minmax,
            hovertemplate=(
                "<b>Decade: %{x}</b><br>"
                "Apparent Temp: %{y:.2f}°C<br>"
                "Min Apparent Temp: %{customdata[0]:.2f}°C<br>"
                "Max Apparent Temp: %{customdata[1]:.2f}°C<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Add Scatter Line (Secondary Y-Axis: Shortwave Radiation)
    fig_bar.add_trace(
        go.Scatter(
            x=adm1_df["decade"],
            y=adm1_df["shortwave_radiation_sum"],
            mode="markers+lines",
            name="Shortwave Radiation",
            line=dict(dash="dot", width=2, color="#FF7F0E"),
        ),
        secondary_y=True,  # Assign this trace to the secondary y-axis
    )

    # Add vertical shade at Selected Year in Bar Chart
    fig_bar.add_shape(
        dict(
            type="rect",
            x0=selected_year - 5,  # Adjusting to center around the decade
            x1=selected_year + 5,  # Ensuring the highlight covers the decade properly
            y0=0,
            y1=max(
                adm1_df["temperature_2m_mean"].max(),
                adm1_df["apparent_temperature_mean"].max(),
            ),
            fillcolor="rgba(200, 200, 200, 1)",  # Light gray with 30% opacity
            layer="below",  # Ensure it is behind all other elements
            line=dict(width=0),  # No border
        )
    )

    # Update layout for dual Y-Axis
    apply_chart_layout(
        fig_bar,
        f"Temperature & Wind Speed ({selected_region})",
        "Decade",
        "Temperature (°C)",
        x_col="decade",
        df=adm1_df,
    )

    # Dynamically calculate the lowest and highest values among both bar traces
    min_y = (
        min(
            adm1_df["temperature_2m_mean"].min(),
            adm1_df["apparent_temperature_mean"].min(),
        )
        - 1
    )
    max_y = (
        max(
            adm1_df["temperature_2m_mean"].max(),
            adm1_df["apparent_temperature_mean"].max(),
        )
        + 1
    )

    # Remove Y-axis grid lines and add dynamic range
    fig_bar.update_layout(
        yaxis=dict(range=[min_y, max_y]),  # Update y-axis to not start from zero
        yaxis2=dict(
            showgrid=False,  # Removes secondary Y-axis grid
            zeroline=False,
        ),
    )

    # LINE CHART: HLI by Month and Decade
    fig_monthly_hli = px.line(
        adm1_month_df,
        x="month",
        y="HLI",
        color="decade",
        labels={"month": "Month", "HLI": "Heat Load Index (HLI)", "decade": "Decade"},
        markers=True,  # Adds markers
    )

    # Set all lines to gray with 50% opacity by default
    for trace in fig_monthly_hli.data:
        trace.line.color = "lightgray"
        trace.opacity = 0.5

    # Highlight the selected decade in blue with full opacity
    for trace in fig_monthly_hli.data:
        if str(trace.name) == str(selected_year):  # Match decade with slider
            trace.line.color = "#1f77b4"  # Plotly's default blue or use any hex
            trace.opacity = 1.0

    # Move the selected trace to the end so it renders on top
    for i, trace in enumerate(fig_monthly_hli.data):
        if str(trace.name) == str(selected_year):
            selected_trace = fig_monthly_hli.data[i]
            fig_monthly_hli.data = tuple(
                t for j, t in enumerate(fig_monthly_hli.data) if j != i
            ) + (selected_trace,)
            break

    # Add a title
    apply_chart_layout(
        fig_monthly_hli,
        f"Monthly HLI Trends - {selected_year} vs Other Decades ({selected_region})",
        "Month",
        "HLI",
        x_col="month",
        df=adm1_month_df,
    )

    fig_monthly_hli.update_layout(showlegend=False)

    return fig_line, fig_bar, fig_monthly_hli


@app.callback(
    [
        Output("play-interval", "disabled"),
        Output("play-button", "children"),
        Output("play-state", "data"),
    ],
    Input("play-button", "n_clicks"),
    State("play-state", "data"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, play_state):
    playing = not play_state["playing"]
    icon = (
        html.I(className="bi bi-pause-fill fs-4")
        if playing
        else html.I(className="bi bi-play-fill fs-4")
    )
    return not playing, icon, {"playing": playing}


@app.callback(
    Output("year-slider", "value"),
    Input("play-interval", "n_intervals"),
    State("year-slider", "value"),
    prevent_initial_call=True,
)
def step_year_slider(n, current_year):
    decades = sorted(gdf_decadal_adm1["decade"].unique())
    if current_year not in decades:
        return decades[0]

    current_idx = decades.index(current_year)
    next_idx = (current_idx + 1) % len(decades)  # Loop back to start
    return decades[next_idx]


# Modal toggle
@app.callback(
    Output("modal", "is_open"), Input("open", "n_clicks"), prevent_initial_call=True
)
def open_modal(n):
    return True


# Step navigation logic
@app.callback(
    Output("onboarding-body", "children"),
    Output("step-store", "children"),
    Output("back", "disabled"),
    Output("next", "children"),  # Change to "Finish" on last step
    Input("next", "n_clicks"),
    Input("back", "n_clicks"),
    State("step-store", "children"),
    prevent_initial_call=True,
)
def navigate_steps(n_next, n_back, step_str):
    ctx = dash.callback_context
    step = int(step_str)

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "next":
        step = min(len(steps) - 1, step + 1)
    elif button_id == "back":
        step = max(0, step - 1)

    next_label = "Finish" if step == len(steps) - 1 else "Next"
    return steps[step], str(step), step == 0, next_label


# Optionally: Close modal on "Finish"
@app.callback(
    Output("modal", "is_open", allow_duplicate=True),
    Input("next", "n_clicks"),
    State("step-store", "children"),
    prevent_initial_call="initial_duplicate",
)
def close_on_finish(n, step):
    if int(step) == len(steps) - 1:
        return False
    raise dash.exceptions.PreventUpdate


# Run app
if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
