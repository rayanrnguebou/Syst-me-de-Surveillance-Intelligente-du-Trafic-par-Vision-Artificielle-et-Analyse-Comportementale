"""
================================================================
SYSTÈME DE SURVEILLANCE TRAFIC — DOUALA
dashboard_dash.py : Dashboard Dash temps réel + Authentification
================================================================
Lancement : python dashboard_dash.py
Accès     : http://localhost:8050
================================================================
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import json
import time
from datetime import datetime, timedelta
from collections import deque
import threading

from Database import DatabaseManager

# ── Connexion DB ─────────────────────────────────────────────
db = DatabaseManager(
    uri="mongodb://localhost:27017/",
    db_name="traffic_douala",
    use_tls=False,
)

# ── Application Dash ─────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE,
                           dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="🚦 Trafic Douala — Surveillance IA",
)
server = app.server

# ── Palette de couleurs ───────────────────────────────────────
COLORS = {
    "bg"        : "#0A1931",
    "card"      : "#0D2136",
    "border"    : "#1F3A5A",
    "gold"      : "#FFD700",
    "teal"      : "#00C8C8",
    "green"     : "#00C864",
    "orange"    : "#FFA500",
    "red"       : "#FF4444",
    "text"      : "#B0C4DE",
    "text_dim"  : "#4A6A8A",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="#0A1931",
    font=dict(color=COLORS["text"], family="Courier New"),
    margin=dict(l=40, r=20, t=40, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"]),
    xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
)


# ================================================================
# LAYOUT — PAGE DE CONNEXION
# ================================================================

login_layout = html.Div(style={
    "minHeight"      : "100vh",
    "backgroundColor": COLORS["bg"],
    "display"        : "flex",
    "alignItems"     : "center",
    "justifyContent" : "center",
    "fontFamily"     : "Courier New, monospace",
}, children=[
    html.Div(style={
        "background"  : COLORS["card"],
        "border"      : f"1px solid {COLORS['border']}",
        "borderRadius": "12px",
        "padding"     : "48px 40px",
        "width"       : "420px",
        "boxShadow"   : "0 20px 60px rgba(0,0,0,0.6)",
    }, children=[

        # Logo & Titre
        html.Div(style={"textAlign": "center", "marginBottom": "32px"},
        children=[
            html.Div("🚦", style={"fontSize": "48px"}),
            html.H2("Surveillance Trafic",
                    style={"color": COLORS["gold"],
                           "margin": "8px 0 4px", "fontSize": "1.6rem"}),
            html.P("Douala, Cameroun — Intelligence Artificielle",
                   style={"color": COLORS["text_dim"],
                          "fontSize": "0.85rem", "margin": 0}),
        ]),

        # Séparateur
        html.Hr(style={"borderColor": COLORS["border"], "margin": "0 0 28px"}),

        # Formulaire
        html.Div([
            html.Label("Identifiant",
                       style={"color": COLORS["teal"],
                              "fontSize": "0.82rem",
                              "fontWeight": "bold",
                              "letterSpacing": "1px"}),
            dbc.Input(
                id="login-username",
                type="text",
                placeholder="nom d'utilisateur",
                style={"background": "#0A1931",
                       "border"    : f"1px solid {COLORS['border']}",
                       "color"     : "white",
                       "marginTop" : "6px",
                       "marginBottom": "18px"},
                n_submit=0,
                debounce=False,
            ),

            html.Label("Mot de passe",
                       style={"color": COLORS["teal"],
                              "fontSize": "0.82rem",
                              "fontWeight": "bold",
                              "letterSpacing": "1px"}),
            dbc.Input(
                id="login-password",
                type="password",
                placeholder="••••••••",
                style={"background": "#0A1931",
                       "border"    : f"1px solid {COLORS['border']}",
                       "color"     : "white",
                       "marginTop" : "6px",
                       "marginBottom": "24px"},
                n_submit=0,
            ),

            # Message d'erreur
            html.Div(id="login-error",
                     style={"color": COLORS["red"],
                            "fontSize": "0.85rem",
                            "marginBottom": "16px",
                            "minHeight" : "20px"}),

            # Bouton connexion
            dbc.Button(
                [html.I(className="fas fa-sign-in-alt me-2"),
                 "Se connecter"],
                id="login-btn",
                color="primary",
                style={"width"     : "100%",
                       "background": f"linear-gradient(135deg, #065A82, #1C7293)",
                       "border"    : "none",
                       "padding"   : "12px",
                       "fontWeight": "bold",
                       "letterSpacing": "1px"},
                n_clicks=0,
            ),
        ]),

        html.Hr(style={"borderColor": COLORS["border"],
                       "margin": "28px 0 16px"}),
        html.P("Système réservé aux administrateurs autorisés.",
               style={"color": COLORS["text_dim"],
                      "fontSize": "0.75rem",
                      "textAlign": "center", "margin": 0}),
    ])
])


# ================================================================
# LAYOUT — DASHBOARD PRINCIPAL
# ================================================================

def make_kpi_card(icon, title, value_id, unit="", color=None):
    """Crée une carte KPI réutilisable."""
    return dbc.Card(style={
        "background"  : COLORS["card"],
        "border"      : f"1px solid {color or COLORS['border']}",
        "borderRadius": "8px",
        "padding"     : "16px",
        "textAlign"   : "center",
    }, children=[
        html.Div(icon, style={"fontSize": "24px", "marginBottom": "4px"}),
        html.P(title, style={"color"      : COLORS["text_dim"],
                             "fontSize"   : "0.75rem",
                             "margin"     : "0 0 4px",
                             "letterSpacing": "0.5px"}),
        html.H3(id=value_id,
                style={"color"    : color or COLORS["gold"],
                       "margin"   : 0,
                       "fontSize" : "1.8rem",
                       "fontFamily": "Courier New"}),
        html.Small(unit, style={"color": COLORS["text_dim"]}),
    ])


def make_alert_badge(alert):
    """Crée un badge d'alerte coloré."""
    colors = {
        "DANGER" : (COLORS["red"],    "#2D0A0A"),
        "WARNING": (COLORS["orange"], "#2D1A0A"),
        "INFO"   : (COLORS["green"],  "#0A2D1A"),
    }
    border_c, bg_c = colors.get(
        alert.get("niveau", "INFO"), (COLORS["teal"], COLORS["card"]))

    ts = alert.get("timestamp", "")
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%H:%M:%S")
    elif isinstance(ts, str) and "T" in ts:
        ts = ts.split("T")[-1][:8]

    return html.Div(style={
        "background"  : bg_c,
        "border"      : f"1px solid {border_c}",
        "borderLeft"  : f"4px solid {border_c}",
        "borderRadius": "6px",
        "padding"     : "8px 12px",
        "marginBottom": "6px",
    }, children=[
        html.Div(style={"display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center"}, children=[
            html.Span(f"⚠ {alert.get('categorie', '')}",
                      style={"color": border_c,
                             "fontWeight": "bold",
                             "fontSize": "0.82rem"}),
            html.Span(ts, style={"color": COLORS["text_dim"],
                                 "fontSize": "0.75rem"}),
        ]),
        html.P(alert.get("message", ""),
               style={"color"    : COLORS["text"],
                      "fontSize" : "0.78rem",
                      "margin"   : "4px 0 0"}),
    ])


dashboard_layout = html.Div(style={
    "backgroundColor": COLORS["bg"],
    "minHeight"      : "100vh",
    "fontFamily"     : "Courier New, monospace",
    "color"          : COLORS["text"],
}, children=[

    # ── Navbar ────────────────────────────────────────────────
    html.Div(style={
        "background"  : COLORS["card"],
        "borderBottom": f"2px solid {COLORS['gold']}",
        "padding"     : "12px 24px",
        "display"     : "flex",
        "alignItems"  : "center",
        "justifyContent": "space-between",
        "position"    : "sticky",
        "top"         : 0,
        "zIndex"      : 1000,
    }, children=[
        html.Div([
            html.Span("🚦 ", style={"fontSize": "22px"}),
            html.Span("SURVEILLANCE TRAFIC",
                      style={"color"      : COLORS["gold"],
                             "fontWeight" : "bold",
                             "fontSize"   : "1.1rem",
                             "letterSpacing": "2px"}),
            html.Span(" — DOUALA, CAMEROUN",
                      style={"color"    : COLORS["text_dim"],
                             "fontSize" : "0.85rem"}),
        ]),
        html.Div(style={"display": "flex", "alignItems": "center",
                        "gap": "20px"}, children=[
            html.Div(id="nav-user",
                     style={"color"   : COLORS["teal"],
                            "fontSize": "0.85rem"}),
            html.Div(id="nav-status",
                     style={"fontSize": "0.8rem"}),
            dbc.Button(
                [html.I(className="fas fa-sign-out-alt me-1"), "Déconnexion"],
                id="logout-btn", size="sm",
                style={"background": "transparent",
                       "border"    : f"1px solid {COLORS['border']}",
                       "color"     : COLORS["text_dim"],
                       "fontSize"  : "0.78rem"},
                n_clicks=0,
            ),
        ]),
    ]),

    html.Div(id="db-status-banner",
             style={
                 "marginTop"  : "10px",
                 "marginBottom": "12px",
                 "padding"    : "10px 16px",
                 "borderRadius": "10px",
                 "border"     : f"1px solid {COLORS['border']}",
                 "background" : "#132040",
                 "color"      : COLORS['text'],
                 "fontSize"   : "0.9rem",
             }),

    html.Div(style={"padding": "20px 24px"}, children=[

        # Intervalle de rafraîchissement
        dcc.Interval(id="interval-fast",   interval=2000,  n_intervals=0),
        dcc.Interval(id="interval-medium", interval=10000, n_intervals=0),
        dcc.Interval(id="interval-slow",   interval=30000, n_intervals=0),

        # ── Rangée KPIs ──────────────────────────────────────
        dbc.Row(style={"marginBottom": "20px"}, children=[
            dbc.Col(make_kpi_card("🚗", "Débit",
                                  "kpi-flow",   "véh/h",
                                  COLORS["teal"]),  width=2),
            dbc.Col(make_kpi_card("📊", "Densité",
                                  "kpi-density","véh/km",
                                  COLORS["gold"]),  width=2),
            dbc.Col(make_kpi_card("⚡", "Vitesse moy.",
                                  "kpi-speed",  "km/h",
                                  COLORS["green"]), width=2),
            dbc.Col(make_kpi_card("🔢", "Total véhicules",
                                  "kpi-total",  "",
                                  COLORS["text"]),  width=2),
            dbc.Col(make_kpi_card("🚨", "Alertes actives",
                                  "kpi-alerts", "",
                                  COLORS["red"]),   width=2),
            dbc.Col(make_kpi_card("👁", "EAR Fatigue",
                                  "kpi-ear",    "",
                                  COLORS["orange"]),width=2),
        ]),

        # ── Rangée Graphiques + Alertes ───────────────────────
        dbc.Row(style={"marginBottom": "20px"}, children=[

            # Graphiques temps réel
            dbc.Col(width=8, children=[
                dbc.Card(style={
                    "background"  : COLORS["card"],
                    "border"      : f"1px solid {COLORS['border']}",
                    "borderRadius": "10px",
                    "padding"     : "16px",
                }, children=[
                    html.H5("📈 Métriques Temps Réel",
                            style={"color": COLORS["gold"],
                                   "marginBottom": "12px"}),
                    dcc.Graph(id="graph-metrics",
                              config={"displayModeBar": False},
                              style={"height": "380px"}),
                ]),
            ]),

            # Panneau alertes
            dbc.Col(width=4, children=[
                dbc.Card(style={
                    "background"  : COLORS["card"],
                    "border"      : f"1px solid {COLORS['red']}",
                    "borderRadius": "10px",
                    "padding"     : "16px",
                    "height"      : "460px",
                }, children=[
                    html.Div(style={
                        "display"       : "flex",
                        "justifyContent": "space-between",
                        "alignItems"    : "center",
                        "marginBottom"  : "12px",
                    }, children=[
                        html.H5("🚨 Alertes",
                                style={"color": COLORS["red"],
                                       "margin": 0}),
                        dbc.Badge(id="alert-count",
                                  color="danger", pill=True),
                    ]),
                    html.Div(id="alerts-panel",
                             style={"overflowY" : "auto",
                                    "maxHeight" : "380px"}),
                ]),
            ]),
        ]),

        # ── Rangée Heatmap + Stats ────────────────────────────
        dbc.Row(style={"marginBottom": "20px"}, children=[

            # Heatmap
            dbc.Col(width=5, children=[
                dbc.Card(style={
                    "background"  : COLORS["card"],
                    "border"      : f"1px solid {COLORS['border']}",
                    "borderRadius": "10px",
                    "padding"     : "16px",
                }, children=[
                    html.H5("🗺️ Heatmap Densité Véhiculaire",
                            style={"color": COLORS["gold"],
                                   "marginBottom": "12px"}),
                    dcc.Graph(id="graph-heatmap",
                              config={"displayModeBar": False},
                              style={"height": "300px"}),
                ]),
            ]),

            # Stats par classe
            dbc.Col(width=4, children=[
                dbc.Card(style={
                    "background"  : COLORS["card"],
                    "border"      : f"1px solid {COLORS['border']}",
                    "borderRadius": "10px",
                    "padding"     : "16px",
                }, children=[
                    html.H5("🚘 Répartition par Classe",
                            style={"color": COLORS["gold"],
                                   "marginBottom": "12px"}),
                    dcc.Graph(id="graph-classes",
                              config={"displayModeBar": False},
                              style={"height": "300px"}),
                ]),
            ]),

            # Stats résumé 24h
            dbc.Col(width=3, children=[
                dbc.Card(style={
                    "background"  : COLORS["card"],
                    "border"      : f"1px solid {COLORS['border']}",
                    "borderRadius": "10px",
                    "padding"     : "16px",
                    "height"      : "360px",
                    "overflowY"   : "auto",
                }, children=[
                    html.H5("📋 Résumé 24h",
                            style={"color": COLORS["gold"],
                                   "marginBottom": "12px"}),
                    html.Div(id="stats-summary"),
                ]),
            ]),
        ]),

        # ── Onglets avancés (Admin seulement) ─────────────────
        html.Div(id="admin-section"),

    ]),

    # Footer
    html.Div(style={
        "background"  : COLORS["card"],
        "borderTop"   : f"1px solid {COLORS['border']}",
        "padding"     : "10px 24px",
        "textAlign"   : "center",
        "fontSize"    : "0.75rem",
        "color"       : COLORS["text_dim"],
    }, children=[
        "Système de Surveillance Intelligente du Trafic — Douala, Cameroun  |  "
        "Vision Artificielle & Analyse Comportementale  |  "
        "Projet Tutoré 4ème année IA/Info  |  © 2025"
    ]),
])


# LAYOUT RACINE
# ================================================================

app.layout = html.Div([
    dcc.Store(id="auth-token",    storage_type="session"),
    dcc.Store(id="auth-user",     storage_type="session"),
    dcc.Store(id="auth-role",     storage_type="session"),
    dcc.Store(id="live-metrics",  storage_type="memory"),
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
])


# ================================================================
# CALLBACKS — ROUTING
# ================================================================

@app.callback(
    Output("page-content", "children"),
    Input("auth-token", "data"),
    Input("url", "pathname"),
)
def display_page(token, pathname):
    """Affiche login ou dashboard selon l'état d'authentification."""
    if token:
        payload = db.security.verify_token(token)
        if payload:
            return dashboard_layout
    return login_layout


# ================================================================
# CALLBACKS — AUTHENTIFICATION
# ================================================================

@app.callback(
    Output("auth-token",  "data"),
    Output("auth-user",   "data"),
    Output("auth-role",   "data"),
    Output("login-error", "children"),
    Input("login-btn",      "n_clicks"),
    Input("login-password", "n_submit"),
    State("login-username", "value"),
    State("login-password", "value"),
    prevent_initial_call=True,
)
def handle_login(n_clicks, n_submit, username, password):
    if not username or not password:
        return no_update, no_update, no_update, "⚠️ Remplissez tous les champs."

    result = db.authenticate(username, password)
    if result:
        return (result["token"],
                result["username"],
                result["role"],
                "")
    return (no_update, no_update, no_update,
            "❌ Identifiant ou mot de passe incorrect.")


@app.callback(
    Output("auth-token", "data", allow_duplicate=True),
    Input("logout-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_logout(n_clicks):
    if n_clicks:
        return None
    return no_update


@app.callback(
    Output("nav-user",   "children"),
    Output("nav-status", "children"),
    Input("auth-user", "data"),
    Input("auth-role", "data"),
    Input("interval-fast", "n_intervals"),
)
def update_navbar(user, role, _):
    role_labels = {
        "admin"   : ("🔑 Admin",    COLORS["gold"]),
        "operator": ("🛡 Opérateur", COLORS["teal"]),
        "viewer"  : ("👁 Lecteur",  COLORS["text_dim"]),
    }
    label, color = role_labels.get(role, ("", COLORS["text"]))
    user_str = html.Span(
        f"Connecté : {user or '—'}  [{label}]",
        style={"color": color})
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    status = html.Span(f"🟢 EN LIGNE  |  {now}",
                       style={"color": COLORS["green"]})
    return user_str, status


@app.callback(
    Output("db-status-banner", "children"),
    Input("interval-fast", "n_intervals"),
    State("auth-token", "data"),
)
def update_db_status(_, token):
    if not token:
        return "Connectez-vous pour voir l'état de la base de données."

    if not db.online:
        return html.Span(
            "⚠️ MongoDB indisponible. Le dashboard est en mode JSON local ; "
            "les données ne seront visibles que si une source insère des métriques.",
            style={"color": COLORS["orange"]}
        )

    recent = db.get_metrics_last_n(1)
    if not recent:
        return html.Span(
            "ℹ️ En attente de données : lancez `main.py` pour alimenter la base de données.",
            style={"color": COLORS["text_dim"]}
        )

    ts = recent[0].get("timestamp")
    if isinstance(ts, datetime):
        ts = ts.strftime("%d/%m/%Y %H:%M:%S")
    return html.Span(
        f"✅ Données actives — dernière métrique reçue le {ts}.",
        style={"color": COLORS["green"]}
    )


# ================================================================
# CALLBACKS — MÉTRIQUES TEMPS RÉEL (2 sec)
# ================================================================

@app.callback(
    Output("kpi-flow",    "children"),
    Output("kpi-density", "children"),
    Output("kpi-speed",   "children"),
    Output("kpi-total",   "children"),
    Output("kpi-alerts",  "children"),
    Output("kpi-ear",     "children"),
    Input("interval-fast", "n_intervals"),
    State("auth-token", "data"),
)
def update_kpis(_, token):
    if not token:
        return ["—"] * 6

    metrics = db.get_metrics_last_n(1)
    m = metrics[0] if metrics else {}

    alerts = db.get_alerts(unacknowledged_only=True, limit=100)

    # EAR : lire d'abord depuis le champ direct de la métrique (plus fiable)
    ear_val = "—"
    ear_raw = m.get("ear")
    if ear_raw is not None:
        ear_val = f"{ear_raw:.3f}"
        if m.get("is_fatigued"):
            ear_val += " ⚠️"
    else:
        # Fallback : chercher dans les alertes FATIGUE récentes
        fat_alerts = [a for a in db.get_alerts(level="DANGER", limit=10)
                      if "FATIGUE" in str(a.get("categorie", ""))]
        if fat_alerts:
            msg = fat_alerts[0].get("message", "")
            if "EAR=" in msg:
                try:
                    ear_val = msg.split("EAR=")[1].split(" ")[0]
                except Exception:
                    ear_val = "—"

    # KPI total véhicules : avec fallback si offline
    if db.online:
        try:
            total_veh = str(db.db.vehicules.count_documents({}))
        except Exception:
            total_veh = str(m.get("vehicules_zone", "—"))
    else:
        total_veh = str(m.get("vehicules_zone", "—"))

    return (
        f"{m.get('debit_veh_h', 0):.0f}",
        f"{m.get('densite_veh_km', 0):.1f}",
        f"{m.get('vitesse_moy_kmh', 0):.1f}",
        total_veh,
        str(len(alerts)),
        ear_val,
    )


@app.callback(
    Output("graph-metrics", "figure"),
    Input("interval-medium", "n_intervals"),
    State("auth-token", "data"),
)
def update_metrics_graph(_, token):
    if not token:
        return go.Figure()

    data_flow    = db.get_time_series("debit_veh_h",     60)
    data_density = db.get_time_series("densite_veh_km",  60)
    data_speed   = db.get_time_series("vitesse_moy_kmh", 60)

    def extract(data, key):
        ts = [d.get("timestamp", "") for d in data]
        vs = [d.get(key, 0) for d in data]
        return ts, vs

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["Débit (véh/h)",
                        "Densité (véh/km)",
                        "Vitesse moy. (km/h)"],
        vertical_spacing=0.08,
    )

    t1, v1 = extract(data_flow,    "debit_veh_h")
    t2, v2 = extract(data_density, "densite_veh_km")
    t3, v3 = extract(data_speed,   "vitesse_moy_kmh")

    fig.add_trace(go.Scatter(
        x=t1, y=v1, fill="tozeroy",
        line=dict(color=COLORS["teal"], width=2),
        fillcolor="rgba(0,200,200,0.15)",
        name="Débit"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t2, y=v2, fill="tozeroy",
        line=dict(color=COLORS["gold"], width=2),
        fillcolor="rgba(255,215,0,0.15)",
        name="Densité"), row=2, col=1)

    # Barres vitesse colorées selon niveau
    bar_colors = [COLORS["red"] if v < 20
                  else COLORS["orange"] if v < 50
                  else COLORS["green"]
                  for v in v3]
    fig.add_trace(go.Bar(
        x=t3, y=v3,
        marker_color=bar_colors,
        name="Vitesse"), row=3, col=1)

    # Seuil débit congestion
    if t1:
        fig.add_hline(y=800, line_dash="dash",
                      line_color=COLORS["orange"],
                      annotation_text="Seuil congestion",
                      row=1, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        showlegend=False,
    )
    fig.update_annotations(font=dict(color=COLORS["text_dim"],
                                     size=10))
    return fig


@app.callback(
    Output("alerts-panel", "children"),
    Output("alert-count",  "children"),
    Input("interval-fast", "n_intervals"),
    State("auth-token", "data"),
)
def update_alerts(_, token):
    if not token:
        return [], "0"
    alerts = db.get_alerts(unacknowledged_only=True, limit=15)
    if not alerts:
        return [html.P("✅ Aucune alerte active",
                       style={"color": COLORS["green"],
                              "textAlign": "center",
                              "marginTop": "40px"})], "0"
    badges = [make_alert_badge(a) for a in alerts]
    return badges, str(len(alerts))


@app.callback(
    Output("graph-heatmap", "figure"),
    Input("interval-slow",  "n_intervals"),
    State("auth-token", "data"),
)
def update_heatmap(_, token):
    if not token:
        return go.Figure()

    # Récupère positions véhicules dernières 10 minutes
    since = datetime.utcnow() - timedelta(minutes=10)
    if db.online:
        docs = list(db.db.vehicules.find(
            {"timestamp": {"$gte": since}},
            {"position": 1, "_id": 0}
        ).limit(2000))
    else:
        docs = db.get_metrics_last_n(100)

    if not docs:
        # Heatmap vide
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT,
                          title="En attente de données...",
                          height=300)
        return fig

    xs = [d["position"]["x"] for d in docs if "position" in d]
    ys = [d["position"]["y"] for d in docs if "position" in d]

    if not xs:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        return fig

    fig = go.Figure(go.Histogram2dContour(
        x=xs, y=ys,
        colorscale="Jet",
        reversescale=False,
        contours=dict(showlabels=False),
        line=dict(width=0),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title=dict(text="Zones de Congestion (10 dernières min.)",
                   font=dict(color=COLORS["gold"], size=12)),
    )
    return fig


@app.callback(
    Output("graph-classes", "figure"),
    Input("interval-medium", "n_intervals"),   # rafraîchissement toutes les 10s
    State("auth-token", "data"),
)
def update_classes_chart(_, token):
    if not token:
        return go.Figure()

    # ── Étape 1 : agrégation depuis la collection vehicules (MongoDB) ──
    result = []
    if db.online:
        try:
            pipeline = [
                {"$match": {"classe": {"$exists": True, "$ne": None}}},
                {"$group": {"_id": "$classe", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
            result = list(db.db.vehicules.aggregate(pipeline))
        except Exception:
            result = []

    # ── Étape 2 : fallback → comptage_classes dans les métriques récentes ──
    if not result:
        # Chercher dans les N dernières métriques pour cumuler les comptages
        latest_metrics = db.get_metrics_last_n(20)
        cumul = {}
        for m in latest_metrics:
            cc = m.get("comptage_classes") or {}
            for classe, nb in cc.items():
                if classe and str(classe).strip():
                    cumul[str(classe)] = cumul.get(str(classe), 0) + nb
        if cumul:
            result = [
                {"_id": k, "count": v}
                for k, v in sorted(cumul.items(), key=lambda x: -x[1])
                if v > 0
            ]

    # ── Étape 3 : fallback → fichier JSONL local (mode offline) ──
    if not result:
        import os, json as _json, glob
        # Le fallback écrit en .jsonl (une ligne JSON par doc)
        jsonl_files = glob.glob("outputs/db_fallback/*.jsonl") + \
                      glob.glob("outputs/db_fallback/*.json")
        cumul = {}
        for jf in jsonl_files[:5]:
            try:
                with open(jf, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = _json.loads(line)
                        except Exception:
                            continue
                        # Comptage depuis un doc métrique
                        cc = entry.get("comptage_classes") or {}
                        for classe, nb in cc.items():
                            if classe and str(classe).strip():
                                cumul[str(classe)] = cumul.get(str(classe), 0) + nb
                        # Comptage depuis un doc véhicule
                        cl = entry.get("classe")
                        if cl and str(cl).strip():
                            cumul[str(cl)] = cumul.get(str(cl), 0) + 1
            except Exception:
                continue
        if cumul:
            result = [
                {"_id": k, "count": v}
                for k, v in sorted(cumul.items(), key=lambda x: -x[1])
                if v > 0
            ]

    # ── Étape 4 : aucune donnée réelle → figure vide avec message ──
    if not result:
        fig = go.Figure()
        fig.add_annotation(
            text="⏳ En attente de données de détection…<br>"
                 "<span style='font-size:11px'>Lancez main.py pour alimenter le système</span>",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=COLORS["text_dim"], size=13),
            align="center",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        return fig

    # ── Construction du graphique ──────────────────────────────────────
    labels = [r["_id"] for r in result]
    values = [r["count"] for r in result]
    total  = sum(values)

    PALETTE = [
        COLORS["green"], COLORS["orange"], COLORS["teal"],
        COLORS["red"],   "#AA00FF",        "#00BFFF",
        COLORS["gold"],
    ]
    colors_pie = PALETTE[:len(labels)]

    # Donut avec texte nb + %
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(
            colors=colors_pie,
            line=dict(color=COLORS["bg"], width=2),
        ),
        texttemplate="%{label}<br><b>%{value}</b> (%{percent})",
        textposition="outside",
        textfont=dict(color="white", size=10),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value} véhicules<br>%{percent}<extra></extra>",
    ))

    # Annotation centrale : total
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:10px'>total</span>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=COLORS["gold"], size=15),
        align="center",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        showlegend=True,
        legend=dict(
            font=dict(color=COLORS["text"], size=9),
            bgcolor="rgba(0,0,0,0)",
            orientation="v",
            x=1.02, y=0.5,
        ),
        margin=dict(l=10, r=80, t=30, b=10),
    )
    return fig


@app.callback(
    Output("stats-summary", "children"),
    Input("interval-slow",  "n_intervals"),
    State("auth-token", "data"),
)
def update_summary(_, token):
    if not token:
        return []
    stats = db.get_stats_summary(24)
    if not stats:
        return [html.P("En attente...",
                       style={"color": COLORS["text_dim"]})]

    rows = [
        ("Débit moyen",      f"{stats.get('debit_moy',0):.0f} véh/h"),
        ("Vitesse moyenne",   f"{stats.get('vitesse_moy',0):.1f} km/h"),
        ("Densité moyenne",   f"{stats.get('densite_moy',0):.1f} véh/km"),
        ("Débit max",         f"{stats.get('debit_max',0):.0f} véh/h"),
        ("Vitesse min",       f"{stats.get('vitesse_min',0):.1f} km/h"),
        ("Alertes totales",   str(stats.get("alertes_total", 0))),
        ("Alertes DANGER",    str(stats.get("alertes_danger", 0))),
    ]
    items = []
    for label, val in rows:
        items.append(html.Div(style={
            "display"        : "flex",
            "justifyContent" : "space-between",
            "padding"        : "6px 0",
            "borderBottom"   : f"1px solid {COLORS['border']}",
        }, children=[
            html.Span(label, style={"color": COLORS["text_dim"],
                                    "fontSize": "0.8rem"}),
            html.Span(val,   style={"color": COLORS["teal"],
                                    "fontWeight": "bold",
                                    "fontSize": "0.82rem"}),
        ]))
    return items


# ================================================================
# LANCEMENT
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🚦 DASHBOARD SURVEILLANCE TRAFIC — DOUALA")
    print("="*60)
    print(f"  MongoDB  : {'✅ Connecté' if db.online else '⚠️  Mode JSON local'}")
    print(f"  Accès    : http://localhost:8050")
    print(f"  Login    : admin / Admin@Douala2025")
    print("="*60 + "\n")

    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False,
    )
