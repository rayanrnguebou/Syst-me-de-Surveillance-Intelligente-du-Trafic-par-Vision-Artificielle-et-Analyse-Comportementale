"""
================================================================
SYSTÈME DE SURVEILLANCE INTELLIGENTE DU TRAFIC — DOUALA
Pipeline Principal : intègre les 4 modules + MongoDB
================================================================
Utilisation :
    python main.py                          (webcam)
    python main.py --source data/video.mp4  (fichier)
    python main.py --heatmap                (avec heatmap)
    python main.py --no-behavior            (sans analyse comport.)

Pour le dashboard web :
    python dashboard_dash.py
    → http://localhost:8050
================================================================
"""

import cv2
import numpy as np
import argparse
import time
import sys
import os
from collections import defaultdict, deque

# ── Imports modules du système ────────────────────────────────
try:
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as e:
    print(f"[ERREUR] Dépendance manquante : {e}")
    print("         pip install ultralytics deep-sort-realtime")
    sys.exit(1)

from module2_parametres_trafic       import TrafficAnalyzer
from module3_analyse_comportementale import (
    FatigueDetector, VehicleAnomalyDetector, AlertManager
)
from database import DatabaseManager   # ← NOUVEAU
import config as cfg


def parse_args():
    p = argparse.ArgumentParser(
        description="Système de Surveillance Trafic Douala — Pipeline complet")
    p.add_argument("--source",       default=None,
                   help="Source vidéo (0=webcam, chemin fichier)")
    p.add_argument("--model",        default=None,
                   help="Modèle YOLOv8 (.pt)")
    p.add_argument("--conf",         type=float, default=None,
                   help="Seuil de confiance détection")
    p.add_argument("--heatmap",      action="store_true",
                   help="Afficher la heatmap de densité")
    p.add_argument("--no-behavior",  action="store_true",
                   help="Désactiver l'analyse comportementale")
    p.add_argument("--road-length",  type=float, default=50.0,
                   help="Longueur tronçon en mètres (défaut: 50m)")
    p.add_argument("--intersection", default="Ndokoti",
                   help="Nom du carrefour surveillé (défaut: Ndokoti)")
    return p.parse_args()


def main():
    args = parse_args()

    # Surcharge config
    source     = args.source or cfg.VIDEO_SOURCE
    model_path = args.model  or cfg.MODEL_PATH
    conf       = args.conf   or cfg.CONFIDENCE_THRESHOLD

    print("\n" + "="*65)
    print("  SYSTÈME DE SURVEILLANCE TRAFIC — DOUALA, CAMEROUN")
    print("  Vision Artificielle + Analyse Comportementale")
    print("="*65)
    print(f"  Source        : {source}")
    print(f"  Modèle        : {model_path}")
    print(f"  Confiance     : {conf}")
    print(f"  Heatmap       : {args.heatmap}")
    print(f"  Comportement  : {not args.no_behavior}")
    print(f"  Tronçon       : {args.road_length} m")
    print(f"  Intersection  : {args.intersection}")
    print("="*65)
    print("\n  Touches : [Q] Quitter  [P] Pause  [H] Toggle Heatmap")
    print("            [S] Screenshot  [G] Générer graphiques\n")

    # ── Connexion MongoDB ─────────────────────────────────────
    db = DatabaseManager(
        uri     = "mongodb://localhost:27017/",
        db_name = "traffic_douala",
        use_tls = False,
    )
    print(f"[DB] MongoDB : {'✅ Connecté' if db.online else '⚠️  Mode JSON local'}\n")

    # ── Chargement modèle YOLOv8 ─────────────────────────────
    print("[INFO] Chargement YOLOv8...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERREUR] {e}")
        print(f"         Essayez : MODEL_PATH = 'yolov8n.pt' dans config.py")
        db.close()
        sys.exit(1)

    # ── Tracker DeepSORT ──────────────────────────────────────
    tracker = DeepSort(
        max_age             = cfg.MAX_AGE,
        n_init              = cfg.N_INIT,
        max_cosine_distance = cfg.MAX_COSINE_DISTANCE,
        nn_budget           = cfg.NN_BUDGET,
    )

    # ── Ouverture source vidéo ────────────────────────────────
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {source}")
        db.close()
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Vidéo : {fw}x{fh} @ {fps:.1f} fps")

    # ── Initialisation des modules ────────────────────────────
    analyzer = TrafficAnalyzer(fw, fh, fps, args.road_length)

    if not args.no_behavior:
        fatigue  = FatigueDetector()
        anomaly  = VehicleAnomalyDetector(
                       fps=fps, px_per_meter=fh/args.road_length)
        alertmgr = AlertManager()
    else:
        fatigue = anomaly = alertmgr = None

    # ── Enregistreur vidéo ────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    writer = None
    if cfg.SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            cfg.OUTPUT_VIDEO_PATH, fourcc, fps, (fw, fh))
        print(f"[INFO] Enregistrement → {cfg.OUTPUT_VIDEO_PATH}")

    # ── Variables boucle ──────────────────────────────────────
    trajectories    = defaultdict(lambda: deque(maxlen=cfg.TRAJECTORY_LENGTH))
    counted_ids     = set()        # IDs déjà comptés (pour insert_vehicle)
    frame_idx       = 0
    fps_history     = deque(maxlen=30)
    paused          = False
    show_heatmap    = args.heatmap
    metric_interval = int(fps * 5) # Insertion métriques toutes les 5 secondes

    # ── Boucle principale ─────────────────────────────────────
    while True:
        t0 = time.time()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin de la vidéo.")
                break

            frame_idx += 1

            # ① Détection YOLOv8
            results = model(
                frame,
                conf    = conf,
                iou     = cfg.IOU_THRESHOLD,
                classes = list(cfg.VEHICLE_CLASSES.keys()),
                verbose = False
            )[0]

            det_input = []
            for box in results.boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_input.append(
                    ([x1, y1, x2-x1, y2-y1], confidence, cls_id))

            # ② Suivi DeepSORT
            tracks    = tracker.update_tracks(det_input, frame=frame)
            confirmed = [t for t in tracks if t.is_confirmed()]

            # ③ Module 2 — Paramètres de trafic
            analyzer.update(confirmed, frame_idx)

            # ── Insertion métriques en BD (toutes les 5s) ────
            if frame_idx % metric_interval == 0:
                db.insert_metric(
                    flow             = analyzer.current_flow,
                    density          = analyzer.current_density,
                    speed            = analyzer.current_speed,
                    vehicles_in_zone = analyzer.vehicles_in_zone,
                    class_counts     = {
                        cfg.VEHICLE_CLASSES.get(k, str(k)): v
                        for k, v in analyzer.class_counts.items()
                    },
                    intersection     = args.intersection,
                )

            # ── Insertion véhicules franchissant la ligne ─────
            for track in confirmed:
                tid  = track.track_id
                ltrb = track.to_ltrb()
                cx   = int((ltrb[0] + ltrb[2]) / 2)
                cy   = int((ltrb[1] + ltrb[3]) / 2)

                if tid not in counted_ids:
                    if tid in analyzer.prev_positions:
                        prev_cy = analyzer.prev_positions[tid][1]
                        if (prev_cy < analyzer.line_y <= cy or
                                cy < analyzer.line_y <= prev_cy):
                            counted_ids.add(tid)
                            db.insert_vehicle(
                                track_id     = tid,
                                class_name   = cfg.VEHICLE_CLASSES.get(
                                                   track.det_class, "Inconnu"),
                                speed_kmh    = analyzer.vehicle_speeds.get(
                                                   tid, 0.0),
                                position     = (cx, cy),
                                intersection = args.intersection,
                            )

            # ④ Module 3 — Analyse comportementale
            if not args.no_behavior:
                fat_result  = fatigue.process(frame)
                anom_alerts = anomaly.update(
                    confirmed, analyzer.vehicle_speeds)

                # Alertes fatigue → alerte manager + BD
                if fat_result["alert"]:
                    alertmgr.add(fat_result["alert"])
                    a = fat_result["alert"]
                    db.insert_alert(
                        level        = a.level.value,
                        category     = a.category,
                        message      = a.message,
                        track_id     = a.track_id,
                        intersection = args.intersection,
                    )

                # Alertes anomalies véhicule → alerte manager + BD
                alertmgr.add_many(anom_alerts)
                for a in anom_alerts:
                    db.insert_alert(
                        level        = a.level.value,
                        category     = a.category,
                        message      = a.message,
                        track_id     = a.track_id,
                        intersection = args.intersection,
                    )

            # ⑤ Dessin — Trajectoires
            if cfg.SHOW_TRAJECTORIES:
                for track in confirmed:
                    tid   = track.track_id
                    ltrb  = track.to_ltrb()
                    cx    = int((ltrb[0] + ltrb[2]) / 2)
                    cy    = int((ltrb[1] + ltrb[3]) / 2)
                    trajectories[tid].append((cx, cy))
                    pts   = list(trajectories[tid])
                    color = cfg.CLASS_COLORS.get(
                        track.det_class, (200, 200, 200))
                    for k in range(1, len(pts)):
                        alpha = k / len(pts)
                        c = tuple(int(v * alpha) for v in color)
                        cv2.line(frame, pts[k-1], pts[k], c, 2)

            # Boîtes englobantes + labels
            for track in confirmed:
                ltrb  = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cid   = track.det_class
                color = cfg.CLASS_COLORS.get(cid, (200, 200, 200))
                cname = cfg.VEHICLE_CLASSES.get(cid, "Véhicule")
                spd   = analyzer.vehicle_speeds.get(track.track_id, 0)
                label = f"{cname} #{track.track_id}  {spd:.0f}km/h"
                lsz, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame,
                              (x1, y1 - lsz[1] - 8),
                              (x1 + lsz[0] + 4, y1),
                              color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)

            # Overlay Module 2 (métriques + ligne de comptage)
            frame = analyzer.draw(frame, confirmed)

            # Heatmap
            if show_heatmap:
                frame = analyzer.get_heatmap_overlay(frame, alpha=0.40)

            # Anomalies Module 3
            if not args.no_behavior and anomaly:
                frame = anomaly.draw_anomalies(frame, confirmed)

            # Alertes Module 3
            if not args.no_behavior and alertmgr:
                frame = alertmgr.draw(frame)

            # FPS
            elapsed = time.time() - t0
            fps_val = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps_val)
            avg_fps = np.mean(fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                        (fw - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 100), 2)

            # Enregistrement vidéo
            if writer:
                writer.write(frame)

            # Affichage
            display = cv2.resize(
                frame, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))
            cv2.imshow(
                "Surveillance Trafic Douala — Pipeline Complet",
                display)

        # ── Gestion clavier ───────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("\n[INFO] Arrêt demandé.")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'PAUSE' if paused else 'REPRISE'}")
        elif key == ord('h'):
            show_heatmap = not show_heatmap
        elif key == ord('s'):
            fname = f"outputs/capture_{frame_idx}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Capture : {fname}")
        elif key == ord('g'):
            analyzer.generate_plots()
            analyzer.save_heatmap()
            print("[INFO] Graphiques générés dans outputs/")

    # ── Nettoyage & fermeture ─────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Sauvegarde finale graphiques
    analyzer.generate_plots()
    analyzer.save_heatmap()

    # Flush final MongoDB — vide tous les buffers restants
    db.flush_all()
    db.close()
    print("[DB] ✅ Toutes les données sauvegardées dans MongoDB")

    # Résumé console
    print("\n" + "="*65)
    print("  RÉSUMÉ FINAL")
    print("="*65)
    summ = analyzer.get_summary()
    for k, v in summ.items():
        print(f"  {k:<25} : {v}")
    if not args.no_behavior and anomaly:
        print()
        astats = anomaly.get_stats()
        for k, v in astats.items():
            print(f"  {k:<25} : {v}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
