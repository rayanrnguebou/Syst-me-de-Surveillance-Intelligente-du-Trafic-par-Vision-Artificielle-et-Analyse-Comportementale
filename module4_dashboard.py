"""
================================================================
SYSTÈME DE SURVEILLANCE INTELLIGENTE DU TRAFIC — DOUALA
Module 4 : Dashboard Streamlit
================================================================
Interface web temps réel pour les gestionnaires du trafic.
Lancement : streamlit run module4_dashboard.py
================================================================
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import json
import csv
from datetime import datetime
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tempfile

# ── Import des modules du système ────────────────────────────
try:
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
    SYSTEM_OK = True
except ImportError:
    SYSTEM_OK = False

try:
    from module2_parametres_trafic import TrafficAnalyzer
    from module3_analyse_comportementale import (
        FatigueDetector, VehicleAnomalyDetector, AlertManager, AlertLevel
    )
    MODULES_OK = True
except ImportError:
    MODULES_OK = False

try:
    from Database import DatabaseManager

# PARTIE MODIFIEE ##############################################

    import Database
    print(Database.__file__)

# FIN

    DB_MODULE_OK = True
except ImportError:
    DB_MODULE_OK = False

import config as cfg

# ================================================================
# CONFIGURATION STREAMLIT
# ================================================================

st.set_page_config(
    page_title="Surveillance Trafic — Douala",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ─────────────────────────────────────────
st.markdown("""
<style>
  /* Fond principal */
  .stApp { background-color: #0A1931; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0D2136 0%, #0A1931 100%);
      border-right: 1px solid #1F3A5A;
  }

  /* Titres */
  h1, h2, h3 { color: #FFD700 !important; font-family: 'Courier New', monospace; }
  p, label, .stMarkdown { color: #B0C4DE !important; }

  /* Métriques */
  [data-testid="metric-container"] {
      background: #0D2136;
      border: 1px solid #1F497D;
      border-radius: 8px;
      padding: 12px;
  }
  [data-testid="metric-container"] label { color: #7FB3D3 !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      color: #FFD700 !important; font-size: 1.8rem !important;
  }

  /* Boutons */
  .stButton > button {
      background: linear-gradient(135deg, #065A82, #1C7293);
      color: white; border: none; border-radius: 6px;
      font-weight: bold; padding: 0.5rem 1.5rem;
  }
  .stButton > button:hover { background: #1C7293; }

  /* Alertes */
  .alert-danger  { background:#2D0A0A; border-left:4px solid #FF4444;
                   padding:8px 12px; border-radius:4px; margin:4px 0; }
  .alert-warning { background:#2D1A0A; border-left:4px solid #FFA500;
                   padding:8px 12px; border-radius:4px; margin:4px 0; }
  .alert-info    { background:#0A2D1A; border-left:4px solid #00C864;
                   padding:8px 12px; border-radius:4px; margin:4px 0; }

  /* Séparateur */
  hr { border-color: #1F3A5A; }

  /* Selectbox / inputs */
  .stSelectbox > div > div { background:#0D2136; color:white; }
  .stSlider > div { color: #7FB3D3; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# ÉTAT DE SESSION
# ================================================================

def init_session():
    """Initialise les variables de session Streamlit."""
    defaults = {
        "running"        : False,
        "frame_count"    : 0,
        "start_time"     : None,
        "alerts_log"     : [],
        "metrics_history": {"flow":[], "density":[], "speed":[], "time":[]},
        "class_counts"   : defaultdict(int),
        "total_vehicles" : 0,
        "current_flow"   : 0.0,
        "current_density": 0.0,
        "current_speed"  : 0.0,
        "current_ear"    : 0.0,
        "is_fatigued"    : False,
        "total_braking"  : 0,
        "total_overtake" : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


def init_database():
    if "db" not in st.session_state:
        st.session_state.db = None
        st.session_state.db_status = "Base de données non initialisée"
        if DB_MODULE_OK:
            try:
                st.session_state.db = DatabaseManager(
                    uri="mongodb://localhost:27017/",
                    db_name="traffic_douala",
                    use_tls=False,
                )
                st.session_state.db_status = (
                    "✅ MongoDB connecté"
                    if st.session_state.db.online
                    else "⚠️ MongoDB indisponible — fallback JSON activé"
                )
            except Exception as e:
                st.session_state.db = None
                st.session_state.db_status = (
                    f"❌ Erreur DB : {e}"
                )
        else:
            st.session_state.db_status = (
                "⚠️ database.py introuvable — écriture DB désactivée"
            )
    return st.session_state.db


db = init_database()


# ================================================================
# SIDEBAR — CONTRÔLES
# ================================================================

with st.sidebar:
    st.markdown("## 🚦 Surveillance Trafic")
    st.markdown("**Douala, Cameroun**")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")

    # Source vidéo
    source_type = st.selectbox(
        "Source vidéo",
        ["Webcam (0)", "Webcam USB (1)", "Fichier vidéo", "URL RTSP"]
    )

    video_source = None
    if source_type in ["Webcam (0)", "Webcam USB (1)"]:
        default_index = 0 if source_type == "Webcam (0)" else 1
        camera_index = st.number_input(
            "Index caméra",
            min_value=0,
            max_value=10,
            value=default_index,
            step=1,
            help="Choisissez l'index de la caméra si 0 ou 1 ne fonctionnent pas."
        )
        video_source = int(camera_index)
    elif source_type == "Fichier vidéo":
        uploaded = st.file_uploader(
            "Uploader une vidéo", type=["mp4","avi","mov","mkv"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            video_source = tmp.name
    elif source_type == "URL RTSP":
        video_source = st.text_input("URL RTSP", "rtsp://...")

    # Mode de détection
    detection_mode = st.selectbox(
        "Mode de détection",
        ["Trafic", "Conducteur"]
    )

    # Modèle
    model_path = st.selectbox(
        "Modèle YOLOv8",
        ["models/yolov8_douala_best.pt", "yolov8n.pt",
         "yolov8s.pt", "yolov8m.pt"]
    )

    if detection_mode == "Conducteur":
        st.info(
            "Mode Conducteur : YOLO détecte la personne puis module 3 analyse fatigue/distraction."
        )

    # Seuils
    st.markdown("### 🎯 Seuils")
    conf_thresh = st.slider("Confiance détection", 0.1, 0.9, 0.35, 0.05)
    ear_thresh  = st.slider("Seuil EAR (fatigue)", 0.15, 0.35, 0.25, 0.01)
    show_heatmap= st.checkbox("Afficher heatmap", value=True)
    show_traj   = st.checkbox("Afficher trajectoires", value=True)

    st.markdown("### 📐 Calibration")
    road_length = st.number_input(
        "Longueur tronçon (m)", 10, 500, 50, 10)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Démarrer", use_container_width=True)
    with col2:
        stop_btn  = st.button("⏹ Arrêter", use_container_width=True)

    if start_btn:
        st.session_state.running   = True
        st.session_state.start_time= time.time()
    if stop_btn:
        st.session_state.running   = False
        if st.session_state.db:
            st.session_state.db.flush_all()

    st.markdown("---")
    st.markdown("### 💾 Export")
    export_csv = st.button("📊 Exporter CSV", use_container_width=True)
    export_rep = st.button("📋 Rapport JSON", use_container_width=True)

    # Bouton de test DB — insère une métrique + une alerte de test
    test_db = st.button("🔍 Tester la DB", use_container_width=True)
    if test_db:
        dbc = st.session_state.get("db")
        if not dbc:
            st.error("❌ Base de données non initialisée — vérifiez la connexion.")
        else:
            try:
                # Insertions de test (metric + alert)
                dbc.insert_metric(
                    flow=0.1, density=0.1, speed=0.1,
                    vehicles_in_zone=0, class_counts={},
                    intersection="test_streamlit",
                )
                dbc.insert_alert(
                    level="INFO",
                    category="TEST",
                    message="Insertion de test depuis Streamlit",
                    track_id=None,
                    intersection="test_streamlit",
                )
                # Force flush pour s'assurer que les données sont écrites
                dbc.flush_all()
                st.success("✅ Test DB effectué — métrique et alerte insérées.")
                st.info("Vérifiez MongoDB ou outputs/db_fallback pour confirmer.")
            except Exception as e:
                st.error(f"❌ Erreur lors du test DB : {e}")

    # Upload image pour test rapide (détection + EAR)
    test_image_file = st.file_uploader(
        "📷 Uploader image (test détection)", type=["jpg", "jpeg", "png"])
    if test_image_file:
        from PIL import Image
        img = Image.open(test_image_file).convert("RGB")
        frame_img = np.array(img)[:, :, ::-1].copy()  # RGB->BGR

        st.markdown("**Résultats test image :**")
        try:
            model_test = YOLO(model_path)
            if detection_mode == "Conducteur":
                yolo_classes = [0]
                class_names = {0: "Personne"}
            else:
                yolo_classes = list(cfg.VEHICLE_CLASSES.keys())
                class_names = cfg.VEHICLE_CLASSES

            res = model_test(frame_img, conf=conf_thresh,
                             classes=yolo_classes, verbose=False)[0]
            annotated = frame_img.copy()
            counts = {}
            for box in res.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cname = class_names.get(cls_id, str(cls_id))
                counts[cname] = counts.get(cname, 0) + 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(annotated, cname, (x1, max(10, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Affiche la liste des classes détectées
            if counts:
                st.write("**Comptage classes :**", counts)
            else:
                st.info("Aucune détection sur l'image (filtre de classes ou confidence).")

            # Si mode conducteur, exécute FatigueDetector sur le recadrage
            if detection_mode == "Conducteur" and res.boxes:
                # prend la première personne détectée
                b = res.boxes[0].xyxy[0]
                x1, y1, x2, y2 = map(int, b)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(annotated.shape[1], x2), min(annotated.shape[0], y2)
                crop = annotated[y1:y2, x1:x2]
                if crop.size > 0:
                    fd = FatigueDetector(ear_threshold=ear_thresh)
                    fres = fd.process(crop)
                    st.write({
                        "EAR": fres.get("ear"),
                        "is_fatigued": fres.get("is_fatigued"),
                        "is_distracted": fres.get("is_distracted"),
                    })
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Recadrage conducteur")
                else:
                    st.info("Recadrage conducteur invalide (taille nulle)")

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
        except Exception as e:
            st.error(f"Erreur lors du test image : {e}")


# ================================================================
# LAYOUT PRINCIPAL
# ================================================================

st.markdown(
    "# 🚦 Système de Surveillance Intelligente du Trafic\n"
    "### Douala, Cameroun — Vision Artificielle & Analyse Comportementale"
)
if st.session_state.get("db_status"):
    st.info(f"**Statut base de données :** {st.session_state.db_status}")
st.markdown("---")

# ── Rangée 1 : KPIs ──────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

kpi_flow_ph    = col1.empty()
kpi_density_ph = col2.empty()
kpi_speed_ph   = col3.empty()
kpi_total_ph   = col4.empty()
kpi_ear_ph     = col5.empty()
kpi_alerts_ph  = col6.empty()


def _render_kpis():
    """Met à jour les KPIs depuis le session_state courant."""
    kpi_flow_ph.metric("🚗 Débit (véh/h)",
                       f"{st.session_state.current_flow:.0f}", "")
    kpi_density_ph.metric("📊 Densité (véh/km)",
                           f"{st.session_state.current_density:.1f}", "")
    kpi_speed_ph.metric("⚡ Vitesse moy. (km/h)",
                         f"{st.session_state.current_speed:.1f}", "")
    kpi_total_ph.metric("🔢 Total compté",
                         f"{st.session_state.total_vehicles}", "")

    _ear_val = st.session_state.current_ear
    if _ear_val > 0.0:
        _ear_display = f"{_ear_val:.3f}"
        _ear_delta   = "⚠️ ALERTE" if st.session_state.is_fatigued else "OK"
    elif MODULES_OK:
        _ear_display = "0.000"
        _ear_delta   = "En attente visage"
    else:
        _ear_display = "N/A"
        _ear_delta   = "MediaPipe absent"

    kpi_ear_ph.metric("👁 EAR (fatigue)", _ear_display, _ear_delta)
    kpi_alerts_ph.metric("🚨 Alertes",
                          len(st.session_state.alerts_log), "")


# Affichage initial
_render_kpis()

st.markdown("---")

# ── Rangée 2 : Vidéo + Alertes ───────────────────────────────
col_video, col_alerts = st.columns([2, 1])

with col_video:
    st.markdown("### 📹 Flux Vidéo en Direct")
    video_placeholder = st.empty()

with col_alerts:
    st.markdown("### 🚨 Alertes Temps Réel")
    alerts_placeholder = st.empty()

st.markdown("---")

# ── Rangée 3 : Graphiques ─────────────────────────────────────
col_g1, col_g2, col_g3 = st.columns([2, 1, 1])

with col_g1:
    st.markdown("### 📈 Débit, Densité & Vitesse")
    chart_placeholder = st.empty()

with col_g2:
    st.markdown("### 🚘 Répartition par Classe")
    classes_chart_placeholder = st.empty()

with col_g3:
    st.markdown("### 🗺️ Heatmap de Densité")
    heatmap_placeholder = st.empty()

# ── Rangée 4 : Stats détaillées ──────────────────────────────
st.markdown("---")
st.markdown("### 📋 Statistiques Détaillées")
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.markdown("**Comptage par classe**")
    classes_placeholder = st.empty()

with col_s2:
    st.markdown("**Anomalies comportementales**")
    anomaly_placeholder = st.empty()

with col_s3:
    st.markdown("**Session**")
    session_placeholder = st.empty()


# ================================================================
# BOUCLE PRINCIPALE DE TRAITEMENT
# ================================================================

def run_detection(video_source, model_path, conf_thresh,
                  ear_thresh, road_length, show_heatmap,
                  detection_mode, classes_chart_ph):
    """
    Boucle principale : capture vidéo → détection → suivi →
    analyse → affichage dashboard.
    """
    if not SYSTEM_OK:
        st.error("❌ ultralytics ou deep-sort-realtime non installés.")
        return

    # Chargement modèle
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle : {e}")
        return

    tracker = DeepSort(
        max_age=cfg.MAX_AGE, n_init=cfg.N_INIT,
        max_cosine_distance=cfg.MAX_COSINE_DISTANCE,
        nn_budget=cfg.NN_BUDGET)

    cap = cv2.VideoCapture(
        int(video_source) if str(video_source).isdigit()
        else video_source)

    if not cap.isOpened() and isinstance(video_source, int):
        # Essaye des indices voisins si le périphérique est mal référencé
        for alt_index in range(0, 4):
            if alt_index == video_source:
                continue
            alt_cap = cv2.VideoCapture(alt_index)
            if alt_cap.isOpened():
                cap.release()
                cap = alt_cap
                st.warning(
                    f"⚠️ Caméra {video_source} indisponible, bascule vers l'index {alt_index}."
                )
                break

    if not cap.isOpened():
        st.error(
            f"❌ Impossible d'ouvrir la source : {video_source}. "
            "Vérifiez l'index de la webcam dans le panneau de gauche."
        )
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialisation des modules
    analyzer  = TrafficAnalyzer(fw, fh, fps, road_length)
    fatigue   = FatigueDetector(ear_threshold=ear_thresh)
    anomaly   = VehicleAnomalyDetector(fps=fps,
                    px_per_meter=fh/road_length)
    alertmgr  = AlertManager()

    db_conn         = st.session_state.db
    db_tick_time    = time.time()
    db_interval_sec = 5.0

    trajectories = defaultdict(lambda: deque(maxlen=40))
    frame_idx    = 0
    fps_history  = deque(maxlen=30)

    while st.session_state.running:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rebobinage
            continue

        frame_idx += 1
        st.session_state.frame_count = frame_idx

        # ① Détection YOLOv8
        if detection_mode == "Conducteur":
            yolo_classes = [0]  # person
            class_names = {0: "Personne"}
        else:
            yolo_classes = list(cfg.VEHICLE_CLASSES.keys())
            class_names = cfg.VEHICLE_CLASSES

        results = model(
            frame, conf=conf_thresh,
            classes=yolo_classes,
            verbose=False)[0]

        det_input = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_input.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))

        # ② Suivi DeepSORT
        tracks = tracker.update_tracks(det_input, frame=frame)
        confirmed = [t for t in tracks if t.is_confirmed()]

        # ③ Module 2 — Métriques trafic
        analyzer.update(confirmed, frame_idx)

        # ④ Module 3 — Analyse comportementale
        driver_crop = None
        if detection_mode == "Conducteur":
            persons = [t for t in confirmed if t.det_class == 0]
            if persons:
                person = persons[0]
                x1, y1, x2, y2 = map(int, person.to_ltrb())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    driver_crop = frame[y1:y2, x1:x2]
        fat_result = fatigue.process(driver_crop if driver_crop is not None else frame)
        anom_alerts= anomaly.update(confirmed, analyzer.vehicle_speeds)

        # Gestion alertes
        if fat_result["alert"]:
            alertmgr.add(fat_result["alert"])
        alertmgr.add_many(anom_alerts)

        # ⑤ Dessin
        # Trajectoires
        if show_traj:
            for track in confirmed:
                tid  = track.track_id
                ltrb = track.to_ltrb()
                cx   = int((ltrb[0]+ltrb[2])/2)
                cy   = int((ltrb[1]+ltrb[3])/2)
                trajectories[tid].append((cx, cy))
                pts  = list(trajectories[tid])
                color= cfg.CLASS_COLORS.get(track.det_class, (200,200,200))
                for k in range(1, len(pts)):
                    cv2.line(frame, pts[k-1], pts[k], color, 2)

        # Boîtes + labels
        for track in confirmed:
            ltrb  = track.to_ltrb()
            x1,y1,x2,y2 = map(int, ltrb)
            cid   = track.det_class
            color = cfg.CLASS_COLORS.get(cid, (200,200,200))
            if detection_mode == "Conducteur" and cid == 0:
                color = (0, 255, 0)
            cname = class_names.get(cid, "Véhicule")
            if detection_mode == "Conducteur" and cid == 0:
                cname = "Conducteur"
            spd   = analyzer.vehicle_speeds.get(track.track_id, 0)
            label = f"{cname} #{track.track_id} {spd:.0f}km/h"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255,255,255), 1)

        # Module 2 overlay
        frame = analyzer.draw(frame, confirmed)

        # Heatmap overlay
        if show_heatmap:
            frame = analyzer.get_heatmap_overlay(frame, alpha=0.35)

        # Alertes overlay
        frame = alertmgr.draw(frame)

        # Anomalies overlay
        frame = anomaly.draw_anomalies(frame, confirmed)

        # FPS
        elapsed = time.time() - t0
        fps_val  = 1.0/elapsed if elapsed > 0 else 0
        fps_history.append(fps_val)
        avg_fps = np.mean(fps_history)
        cv2.putText(frame, f"FPS:{avg_fps:.1f}",
                    (fw-120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,100), 2)

        # ⑥ Mise à jour session state
        summ  = analyzer.get_summary()
        astats= anomaly.get_stats()

        st.session_state.current_flow    = summ["debit_veh_heure"]
        st.session_state.current_density = summ["densite_veh_km"]
        st.session_state.current_speed   = summ["vitesse_moy_kmh"]
        st.session_state.total_vehicles  = summ["total_vehicules"]
        st.session_state.current_ear     = fat_result["ear"]
        st.session_state.is_fatigued     = fat_result["is_fatigued"]
        st.session_state.total_braking   = astats["total_freinages"]
        st.session_state.total_overtake  = astats["total_depassements"]

        # Conversion des clés int → noms lisibles pour l'affichage
        raw_counts = summ.get("comptage_classes", {})
        for k, v in raw_counts.items():
            # k peut être int (id YOLO) ou str selon le module source
            if isinstance(k, int):
                label = cfg.VEHICLE_CLASSES.get(k, f"Classe {k}")
            else:
                try:
                    label = cfg.VEHICLE_CLASSES.get(int(k), str(k))
                except (ValueError, TypeError):
                    label = str(k)
            if label and v > 0:
                # Cumul : on additionne au lieu de remplacer
                current = st.session_state.class_counts
                if isinstance(current, dict):
                    current[label] = current.get(label, 0) + v
                else:
                    st.session_state.class_counts = {label: v}




        # Envoi des données vers MongoDB / fallback JSON
        if db_conn and time.time() - db_tick_time >= db_interval_sec:
            db_tick_time = time.time()
            db_conn.insert_metric(
                flow             = summ["debit_veh_heure"],
                density          = summ["densite_veh_km"],
                speed            = summ["vitesse_moy_kmh"],
                vehicles_in_zone = summ["total_vehicules"],
                class_counts     = {
                    cfg.VEHICLE_CLASSES.get(k, str(k)): v
                    for k, v in summ["comptage_classes"].items()
                },
                ear              = fat_result.get("ear", None),
                is_fatigued      = fat_result.get("is_fatigued", False),
            )
            # Flush immédiat pour que le dashboard Dash lise les données à temps
            db_conn.flush_all()
            for track in confirmed:
                ltrb = track.to_ltrb()
                cx   = int((ltrb[0] + ltrb[2]) / 2)
                cy   = int((ltrb[1] + ltrb[3]) / 2)
                db_conn.insert_vehicle(
                    track_id   = track.track_id,
                    class_name = cfg.VEHICLE_CLASSES.get(
                        track.det_class, "Véhicule"),
                    speed_kmh  = analyzer.vehicle_speeds.get(track.track_id, 0.0),
                    position   = (cx, cy),
                )

        # Envoi des alertes vers MongoDB / fallback JSON
        if db_conn and fat_result["alert"]:
            a = fat_result["alert"]
            db_conn.insert_alert(
                level    = a.level.value,
                category = a.category,
                message  = a.message,
                track_id = a.track_id,
            )
        if db_conn and anom_alerts:
            for a in anom_alerts:
                db_conn.insert_alert(
                    level    = a.level.value,
                    category = a.category,
                    message  = a.message,
                    track_id = a.track_id,
                )

        # Historique métriques
        h = st.session_state.metrics_history
        h["flow"].append(summ["debit_veh_heure"])
        h["density"].append(summ["densite_veh_km"])
        h["speed"].append(summ["vitesse_moy_kmh"])
        h["time"].append(datetime.now().strftime("%H:%M:%S"))
        for key in h:
            if len(h[key]) > 300:
                h[key] = h[key][-300:]

        # Alertes log
        for a in alertmgr.alerts:
            entry = str(a)
            if entry not in st.session_state.alerts_log:
                st.session_state.alerts_log.append(entry)

        # ⑦ Affichage dans Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(
            frame_rgb, channels="RGB", use_container_width=True)

        # KPIs mis à jour à chaque frame
        _render_kpis()

        # Alertes
        _render_alerts(alerts_placeholder, alertmgr)

        # Graphiques (toutes les 30 frames)
        if frame_idx % 30 == 0:
            _render_charts(chart_placeholder)
            _render_classes_chart(classes_chart_ph,
                                  st.session_state.class_counts)
            _render_heatmap(heatmap_placeholder, analyzer)
            _render_stats(classes_placeholder,
                          anomaly_placeholder, session_placeholder,
                          astats)

    if db_conn:
        db_conn.flush_all()
    cap.release()


# ================================================================
# FONCTIONS DE RENDU
# ================================================================

def _render_alerts(placeholder, alertmgr):
    """Affiche les dernières alertes dans le panneau dédié."""
    html = ""
    for a in list(alertmgr.alerts)[-8:]:
        cls = {
            AlertLevel.DANGER : "alert-danger",
            AlertLevel.WARNING: "alert-warning",
            AlertLevel.INFO   : "alert-info",
        }.get(a.level, "alert-info")
        ts = a.timestamp.strftime("%H:%M:%S")
        html += (f'<div class="{cls}">'
                 f'<b>{ts} — {a.category}</b><br>'
                 f'<small>{a.message}</small></div>')
    if not html:
        html = '<p style="color:#555">Aucune alerte pour l\'instant.</p>'
    placeholder.markdown(html, unsafe_allow_html=True)


def _render_charts(placeholder):
    """Graphiques débit + densité + vitesse."""
    h = st.session_state.metrics_history
    if len(h["flow"]) < 2:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 7))
    fig.patch.set_facecolor('#0D2136')

    x = range(len(h["flow"]))
    datasets = [
        (axes[0], h["flow"],    "#00C8C8", "Débit (véh/h)"),
        (axes[1], h["density"], "#FFD700", "Densité (véh/km)"),
        (axes[2], h["speed"],   "#00FF88", "Vitesse moy. (km/h)"),
    ]
    for ax, data, color, ylabel in datasets:
        ax.set_facecolor('#0A1931')
        ax.fill_between(x, data, alpha=0.25, color=color)
        ax.plot(x, data, color=color, linewidth=1.5)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.tick_params(colors='gray', labelsize=8)
        ax.spines[:].set_color('#1F3A5A')

    plt.tight_layout(pad=1.5)
    placeholder.pyplot(fig)
    plt.close()


def _render_heatmap(placeholder, analyzer):
    """Affiche la heatmap de densité."""
    if analyzer.heatmap_accum.max() == 0:
        return
    hm = cv2.GaussianBlur(analyzer.heatmap_accum, (0,0), 25)
    hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
    hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8),
                                  cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    placeholder.image(hm_rgb, caption="Heatmap densité vehiculaire",
                      use_container_width=True)


def _render_classes_chart(placeholder, class_counts: dict):
    """Graphique en barres horizontales pour la répartition par classe de véhicule."""
    if not class_counts:
        placeholder.markdown(
            "<p style='color:#4A6A8A; text-align:center; padding-top:30px;'>"
            "⏳ En attente de détections…</p>",
            unsafe_allow_html=True,
        )
        return

    labels = list(class_counts.keys())
    values = list(class_counts.values())
    total  = sum(values) or 1

    # Palette de couleurs par classe
    PALETTE = [
        "#00C8C8", "#FFD700", "#00FF88",
        "#FFA500", "#FF4444", "#AA00FF", "#00BFFF",
    ]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(4, max(2.5, len(labels) * 0.55)))
    fig.patch.set_facecolor("#0D2136")
    ax.set_facecolor("#0A1931")

    bars = ax.barh(labels, values, color=colors, height=0.55,
                   edgecolor="#1F3A5A", linewidth=0.8)

    # Labels valeur + pourcentage sur chaque barre
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)",
            va="center", ha="left",
            color="white", fontsize=8,
        )

    ax.set_xlabel("Nombre de véhicules", color="#7FB3D3", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_color("#1F3A5A")
    ax.xaxis.label.set_color("#7FB3D3")
    ax.set_xlim(0, max(values) * 1.35)
    plt.tight_layout(pad=1.2)

    placeholder.pyplot(fig)
    plt.close()


def _render_stats(classes_ph, anomaly_ph, session_ph, astats):
    """Affiche les statistiques détaillées."""
    # Comptage par classe — les clés sont déjà des noms lisibles
    cc = st.session_state.class_counts

    if cc:
        total = sum(cc.values()) or 1
        rows = ""
        for label, v in sorted(cc.items(), key=lambda x: -x[1]):
            pct = v / total * 100
            rows += f"| {label} | {v} | {pct:.1f}% |\n"
        classes_ph.markdown(
            f"| Classe | Nb | % |\n|---|---|---|\n{rows}")
    else:
        classes_ph.markdown("_Aucune détection_")



    # Anomalies
    anomaly_ph.markdown(
        f"| Métrique | Valeur |\n|---|---|\n"
        f"| Freinages brusques | {astats['total_freinages']} |\n"
        f"| Dépassements danger. | {astats['total_depassements']} |\n"
        f"| Véhicules impliqués | {astats['vehicules_impliques']} |"
    )

    # Session
    elapsed = 0
    if st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
    session_ph.markdown(
        f"| Métrique | Valeur |\n|---|---|\n"
        f"| Durée session | {elapsed}s |\n"
        f"| Frames traitées | {st.session_state.frame_count} |\n"
        f"| Total alertes | {len(st.session_state.alerts_log)} |"
    )


def _export_csv():
    """Exporte l'historique des métriques en CSV."""
    h = st.session_state.metrics_history
    path = "outputs/rapport_trafic.csv"
    os.makedirs("outputs", exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "debit_veh_h",
                    "densite_veh_km", "vitesse_kmh"])
        for i in range(len(h["time"])):
            w.writerow([h["time"][i], h["flow"][i],
                        h["density"][i], h["speed"][i]])
    st.success(f"✅ CSV exporté : {path}")


def _export_json():
    """Exporte un rapport JSON complet."""
    report = {
        "date"          : datetime.now().isoformat(),
        "session_frames": st.session_state.frame_count,
        "total_vehicles": st.session_state.total_vehicles,
        "debit_moy"     : np.mean(
            st.session_state.metrics_history["flow"] or [0]),
        "vitesse_moy"   : np.mean(
            st.session_state.metrics_history["speed"] or [0]),
        "total_alertes" : len(st.session_state.alerts_log),
        "alertes"       : st.session_state.alerts_log[-50:],
    }
    path = "outputs/rapport_trafic.json"
    os.makedirs("outputs", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=str)
    st.success(f"✅ Rapport JSON exporté : {path}")

if export_csv:
    _export_csv()
if export_rep:
    _export_json()


# ================================================================
# LANCEMENT
# ================================================================

if st.session_state.running:
    run_detection(
        video_source=video_source,
        model_path=model_path,
        conf_thresh=conf_thresh,
        ear_thresh=ear_thresh,
        road_length=road_length,
        show_heatmap=show_heatmap,
        detection_mode=detection_mode,
        classes_chart_ph=classes_chart_placeholder,
    )
else:
    video_placeholder.markdown(
        """
        <div style='text-align:center; padding:80px;
                    background:#0D2136; border-radius:10px;
                    border:1px dashed #1F497D;'>
          <h2 style='color:#7FB3D3;'>⏸ Système en attente</h2>
          <p style='color:#4A6A8A;'>
            Configurez la source vidéo dans le panneau gauche<br>
            puis cliquez sur <b style='color:#FFD700;'>▶ Démarrer</b>
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
