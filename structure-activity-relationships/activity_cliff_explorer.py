"""
Interactive 3D Activity-Cliff Explorer
======================================

Usage in Google Colab
---------------------
1. Install dependencies once:
   %pip -q install dash plotly

2. Upload this file to Colab, then:
   from activity_cliff_explorer import launch_activity_cliff_explorer

3. Run:
   launch_activity_cliff_explorer(
       df,
       pair_df,
       SIMILARITY_THRESHOLD,
       ACTIVITY_DELTA_THRESHOLD
   )

Expected dataframe columns
--------------------------
df:
    pIC50
    mol_obj

pair_df:
    idx_A
    idx_B
    similarity
    delta_pIC50
"""

import base64
from io import BytesIO

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from rdkit.Chem import Draw


def _mol_to_data_uri(mol, legend=""):
    """Convert an RDKit molecule drawing to an embeddable PNG data URI."""
    image = Draw.MolToImage(
        mol,
        size=(350, 250),
        legend=legend
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


def _prepare_pair_data(
    df,
    pair_df,
    similarity_threshold,
    activity_delta_threshold
):
    """Prepare normal-pair and activity-cliff tables."""
    plot_pairs = pair_df.copy()

    plot_pairs["pIC50_A"] = plot_pairs["idx_A"].map(df["pIC50"])
    plot_pairs["pIC50_B"] = plot_pairs["idx_B"].map(df["pIC50"])

    plot_pairs["mean_pIC50"] = (
        plot_pairs["pIC50_A"] +
        plot_pairs["pIC50_B"]
    ) / 2.0

    is_cliff = (
        (plot_pairs["similarity"] >= similarity_threshold) &
        (plot_pairs["delta_pIC50"] >= activity_delta_threshold)
    )

    normal_pairs = (
        plot_pairs.loc[~is_cliff]
        .copy()
        .reset_index(drop=True)
    )

    cliff_pairs = (
        plot_pairs.loc[is_cliff]
        .copy()
        .reset_index(drop=True)
    )

    return normal_pairs, cliff_pairs


def _build_figure(normal_pairs, cliff_pairs):
    """Create the interactive 3D Plotly similarity–activity landscape."""
    fig = go.Figure()

    # Other molecular pairs
    fig.add_trace(
        go.Scatter3d(
            x=normal_pairs["similarity"],
            y=normal_pairs["delta_pIC50"],
            z=normal_pairs["mean_pIC50"],
            mode="markers",
            marker=dict(
                size=2.5,
                opacity=0.18,
                color="gray"
            ),
            name="Other molecular pairs",
            customdata=normal_pairs[
                ["idx_A", "idx_B", "pIC50_A", "pIC50_B"]
            ].to_numpy(),
            hovertemplate=(
                "<b>Molecular Pair</b><br>"
                "Molecule A: %{customdata[0]:.0f}<br>"
                "Molecule B: %{customdata[1]:.0f}<br>"
                "Similarity: %{x:.3f}<br>"
                "|Δ pIC50|: %{y:.2f}<br>"
                "pIC50 A: %{customdata[2]:.2f}<br>"
                "pIC50 B: %{customdata[3]:.2f}<br>"
                "Mean pIC50: %{z:.2f}"
                "<extra></extra>"
            )
        )
    )

    # Activity cliffs
    fig.add_trace(
        go.Scatter3d(
            x=cliff_pairs["similarity"],
            y=cliff_pairs["delta_pIC50"],
            z=cliff_pairs["mean_pIC50"],
            mode="markers",
            marker=dict(
                size=7,
                opacity=0.95,
                color="red",
                symbol="circle",
                line=dict(
                    width=1,
                    color="darkred"
                )
            ),
            name="Activity cliffs",
            customdata=cliff_pairs[
                ["idx_A", "idx_B", "pIC50_A", "pIC50_B"]
            ].to_numpy(),
            hovertemplate=(
                "<b>ACTIVITY CLIFF</b><br>"
                "Molecule A: %{customdata[0]:.0f}<br>"
                "Molecule B: %{customdata[1]:.0f}<br>"
                "Similarity: %{x:.3f}<br>"
                "|Δ pIC50|: %{y:.2f}<br>"
                "pIC50 A: %{customdata[2]:.2f}<br>"
                "pIC50 B: %{customdata[3]:.2f}<br>"
                "Mean pIC50: %{z:.2f}<br><br>"
                "<b>CLICK TO SHOW STRUCTURES</b>"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=dict(
            text="3D Similarity–Activity Landscape",
            x=0.5,
            font=dict(size=20)
        ),
        height=640,
        margin=dict(l=0, r=0, b=0, t=55),
        scene=dict(
            xaxis=dict(
                title="Morgan Tanimoto Similarity",
                backgroundcolor="white",
                gridcolor="lightgray"
            ),
            yaxis=dict(
                title="|Δ pIC50|",
                backgroundcolor="white",
                gridcolor="lightgray"
            ),
            zaxis=dict(
                title="Mean pIC50",
                backgroundcolor="white",
                gridcolor="lightgray"
            ),
            camera=dict(
                eye=dict(x=1.45, y=1.45, z=1.15)
            )
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.75)"
        ),
        paper_bgcolor="white"
    )

    return fig


def launch_activity_cliff_explorer(
    df,
    pair_df,
    similarity_threshold=0.75,
    activity_delta_threshold=1.50,
    app_height=720
):
    """
    Launch an interactive 3D activity-cliff explorer inside Google Colab.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - pIC50
        - mol_obj

    pair_df : pandas.DataFrame
        Must contain:
        - idx_A
        - idx_B
        - similarity
        - delta_pIC50

    similarity_threshold : float, default=0.75
        Minimum Tanimoto similarity used to define an activity cliff.

    activity_delta_threshold : float, default=1.50
        Minimum absolute pIC50 difference used to define an activity cliff.

    app_height : int, default=720
        Height in pixels of the embedded Dash application.
    """

    normal_pairs, cliff_pairs = _prepare_pair_data(
        df,
        pair_df,
        similarity_threshold,
        activity_delta_threshold
    )

    print(f"Other molecular pairs: {len(normal_pairs):,}")
    print(f"Activity cliffs:       {len(cliff_pairs):,}")

    fig = _build_figure(
        normal_pairs,
        cliff_pairs
    )

    app = Dash(__name__)

    initial_panel = html.Div(
        [
            html.H3(
                "Selected Activity Cliff",
                style={
                    "textAlign": "center",
                    "marginTop": "8px",
                    "marginBottom": "8px"
                }
            ),
            html.Div(
                "Click a red point to compare its two molecules.",
                style={
                    "textAlign": "center",
                    "fontSize": "15px",
                    "color": "#555",
                    "padding": "25px 15px"
                }
            )
        ]
    )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H3(
                        "Interactive Activity-Cliff Explorer",
                        style={
                            "textAlign": "center",
                            "marginBottom": "4px"
                        }
                    ),
                    html.P(
                        [
                            "Rotate and zoom the 3D plot. Hover for pair data. ",
                            html.B("Click a red activity-cliff point"),
                            " to display its structures."
                        ],
                        style={
                            "textAlign": "center",
                            "fontSize": "14px",
                            "margin": "0 auto",
                            "maxWidth": "620px",
                            "lineHeight": "1.35"
                        }
                    )
                ],
                style={"marginBottom": "8px"}
            ),

            html.Div(
                [
                    # LEFT: 3D plot
                    html.Div(
                        [
                            dcc.Graph(
                                id="activity-cliff-3d",
                                figure=fig,
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "responsive": True
                                },
                                style={
                                    "width": "100%",
                                    "height": "640px"
                                }
                            )
                        ],
                        style={
                            "flex": "1.55",
                            "minWidth": "560px",
                            "paddingRight": "6px"
                        }
                    ),

                    # RIGHT: structures
                    html.Div(
                        id="structure-display",
                        children=initial_panel,
                        style={
                            "flex": "1",
                            "minWidth": "340px",
                            "border": "1px solid #cccccc",
                            "borderRadius": "8px",
                            "padding": "10px",
                            "backgroundColor": "#fafafa",
                            "alignSelf": "stretch"
                        }
                    )
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "stretch",
                    "gap": "10px",
                    "flexWrap": "wrap"
                }
            )
        ],
        style={
            "fontFamily": "Arial, sans-serif",
            "maxWidth": "1250px",
            "margin": "auto"
        }
    )

    @app.callback(
        Output(
            "structure-display",
            "children"
        ),
        Input(
            "activity-cliff-3d",
            "clickData"
        )
    )
    def display_cliff_structures(click_data):
        if click_data is None:
            return initial_panel

        point = click_data["points"][0]

        # Trace 0 = gray molecular pairs
        # Trace 1 = red activity cliffs
        if point["curveNumber"] != 1:
            return html.Div(
                [
                    html.H3(
                        "Selected Molecular Pair",
                        style={"textAlign": "center"}
                    ),
                    html.P(
                        "This gray point is not classified as an activity cliff.",
                        style={"textAlign": "center"}
                    ),
                    html.P(
                        "Select one of the red points.",
                        style={
                            "textAlign": "center",
                            "fontWeight": "bold"
                        }
                    )
                ]
            )

        custom = point["customdata"]

        idx_a = int(custom[0])
        idx_b = int(custom[1])

        mol_a = df.loc[idx_a, "mol_obj"]
        mol_b = df.loc[idx_b, "mol_obj"]

        p_a = float(df.loc[idx_a, "pIC50"])
        p_b = float(df.loc[idx_b, "pIC50"])

        similarity = float(point["x"])
        delta = float(point["y"])
        mean_activity = float(point["z"])

        fold_difference = 10 ** delta

        if p_a > p_b:
            more_active = "Molecule A"
        elif p_b > p_a:
            more_active = "Molecule B"
        else:
            more_active = "Equal activity"

        img_a = _mol_to_data_uri(
            mol_a,
            legend=(
                f"Molecule A — index {idx_a}\n"
                f"pIC50 = {p_a:.2f}"
            )
        )

        img_b = _mol_to_data_uri(
            mol_b,
            legend=(
                f"Molecule B — index {idx_b}\n"
                f"pIC50 = {p_b:.2f}"
            )
        )

        return html.Div(
            [
                html.H3(
                    "Selected Activity Cliff",
                    style={
                        "textAlign": "center",
                        "marginTop": "3px",
                        "marginBottom": "4px"
                    }
                ),

                html.P(
                    f"Tanimoto = {similarity:.3f} | ΔpIC50 = {delta:.2f}",
                    style={
                        "textAlign": "center",
                        "fontSize": "14px",
                        "marginTop": "0px",
                        "marginBottom": "8px"
                    }
                ),

                # Molecular structures side-by-side
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=img_a,
                                    style={
                                        "width": "100%",
                                        "maxWidth": "230px"
                                    }
                                ),
                                html.Div(
                                    [
                                        html.B("Molecule A"),
                                        html.Br(),
                                        f"pIC50 = {p_a:.2f}"
                                    ],
                                    style={"textAlign": "center"}
                                )
                            ],
                            style={
                                "width": "49%",
                                "textAlign": "center"
                            }
                        ),

                        html.Div(
                            [
                                html.Img(
                                    src=img_b,
                                    style={
                                        "width": "100%",
                                        "maxWidth": "230px"
                                    }
                                ),
                                html.Div(
                                    [
                                        html.B("Molecule B"),
                                        html.Br(),
                                        f"pIC50 = {p_b:.2f}"
                                    ],
                                    style={"textAlign": "center"}
                                )
                            ],
                            style={
                                "width": "49%",
                                "textAlign": "center"
                            }
                        )
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "flex-start"
                    }
                ),

                # Compact interpretation
                html.Div(
                    [
                        html.H4(
                            "Activity-Cliff Interpretation",
                            style={"marginBottom": "7px"}
                        ),
                        html.P(
                            [html.B("Similarity: "), f"{similarity:.3f}"],
                            style={"margin": "3px"}
                        ),
                        html.P(
                            [html.B("|Δ pIC50|: "), f"{delta:.2f}"],
                            style={"margin": "3px"}
                        ),
                        html.P(
                            [html.B("Mean pIC50: "), f"{mean_activity:.2f}"],
                            style={"margin": "3px"}
                        ),
                        html.P(
                            [
                                html.B("Approx. IC50 difference: "),
                                f"{fold_difference:,.1f}-fold"
                            ],
                            style={"margin": "3px"}
                        ),
                        html.P(
                            [html.B("Higher potency: "), more_active],
                            style={"margin": "3px"}
                        )
                    ],
                    style={
                        "textAlign": "center",
                        "fontSize": "13px",
                        "marginTop": "8px",
                        "padding": "8px",
                        "backgroundColor": "white",
                        "borderRadius": "6px"
                    }
                ),

                html.Div(
                    [
                        html.B("SAR question:"),
                        html.Br(),
                        "What structural change might explain the large difference in activity?"
                    ],
                    style={
                        "textAlign": "center",
                        "fontSize": "13px",
                        "marginTop": "8px",
                        "padding": "8px",
                        "border": "1px solid #dddddd",
                        "borderRadius": "6px",
                        "backgroundColor": "white"
                    }
                )
            ]
        )

    app.run(
        jupyter_mode="inline",
        jupyter_height=app_height,
        debug=False
    )
