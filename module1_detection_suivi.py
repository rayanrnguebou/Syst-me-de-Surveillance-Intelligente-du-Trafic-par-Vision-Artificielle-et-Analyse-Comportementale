"""
================================================================
SYSTÈME DE SURVEILLANCE INTELLIGENTE DU TRAFIC — DOUALA
Module 1 : Détection & Suivi (YOLOv8 + DeepSORT)
================================================================

Auteurs  : BEKOU Adrien, NGUEBOU TEMGOUA Rayan
Projet   : Projet Tutoré 4ème année — IA/Informatique
Encadrants : M. NDJE MAN D.F., M. KAZE Roger (3iL / Labo MIA)

Description :
    Ce module réalise :
    1. Détection multi-classes de véhicules (YOLOv8)
    2. Suivi multi-objets avec ID persistants (DeepSORT)
    3. Comptage par ligne virtuelle
    4. Affichage des trajectoires
    5. Enregistrement de la vidéo annotée

Utilisation :
    python module1_detection_suivi.py
    python module1_detection_suivi.py --source data/ma_video.mp4
    python module1_detection_suivi.py --source 0              (webcam)
    python module1_detection_suivi.py --source 1              (webcam USB)
================================================================
"""

import cv2
import numpy as np
import argparse
import time
import sys
import os
from collections import defaultdict, deque

# ── Vérification des dépendances ────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERREUR] ultralytics non installé.")
    print("         Exécutez : pip install ultralytics")
    sys.exit(1)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("[ERREUR] deep-sort-realtime non installé.")
    print("         Exécutez : pip install deep-sort-realtime")
    sys.exit(1)

# ── Import de la configuration ───────────────────────────────
try:
    import config as cfg
except ImportError:
    print("[ERREUR] Fichier config.py introuvable.")
    sys.exit(1)


# ================================================================
# CLASSE PRINCIPALE : TrafficDetector
# ================================================================

class TrafficDetector:
    """
    Système de détection et suivi de véhicules pour le trafic de Douala.

    Pipeline :
        Frame vidéo → YOLOv8 (détection) → DeepSORT (suivi) → Affichage
    """

    def __init__(self, source=None):
        """
        Initialise le détecteur.

        Args:
            source : source vidéo (chemin fichier, 0 pour webcam, etc.)
                     Si None, utilise cfg.VIDEO_SOURCE
        """
        self.source = source if source is not None else cfg.VIDEO_SOURCE

        # ── Chargement du modèle YOLOv8 ─────────────────────
        print(f"\n[INFO] Chargement du modèle YOLOv8 : {cfg.MODEL_PATH}")
        print("       (Téléchargement automatique si absent — ~25 Mo)\n")
        self.model = YOLO(cfg.MODEL_PATH)

        # ── Initialisation de DeepSORT ───────────────────────
        print("[INFO] Initialisation du tracker DeepSORT...")
        self.tracker = DeepSort(
            max_age=cfg.MAX_AGE,
            n_init=cfg.N_INIT,
            max_cosine_distance=cfg.MAX_COSINE_DISTANCE,
            nn_budget=cfg.NN_BUDGET,
        )

        # ── Structures de données ────────────────────────────
        # Trajectoires : dict {track_id: deque([(x,y), ...])}
        self.trajectories = defaultdict(lambda: deque(maxlen=cfg.TRAJECTORY_LENGTH))

        # Compteurs de véhicules par classe
        self.class_counter = defaultdict(int)

        # Ensemble des IDs déjà comptés (évite double comptage)
        self.counted_ids = set()

        # Historique des vitesses estimées par véhicule
        self.speed_history = defaultdict(list)

        # Positions précédentes pour calcul de vitesse
        self.prev_positions = {}

        # Statistiques générales
        self.total_frames    = 0
        self.fps_history     = deque(maxlen=30)
        self.total_vehicles  = 0

        print("[INFO] Système initialisé avec succès !\n")
        print("=" * 60)
        print(f"  Source vidéo  : {self.source}")
        print(f"  Modèle YOLO   : {cfg.MODEL_PATH}")
        print(f"  Confiance min : {cfg.CONFIDENCE_THRESHOLD}")
        print(f"  Classes       : {list(cfg.VEHICLE_CLASSES.values())}")
        print("=" * 60)
        print("\n  [Q] ou [Echap] pour quitter")
        print("  [S] pour sauvegarder une capture d'écran")
        print("  [P] pour mettre en pause\n")

    # ────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : run()
    # ────────────────────────────────────────────────────────

    def run(self):
        """
        Lance la boucle principale de détection et suivi.
        Lit la source vidéo frame par frame et traite chaque image.
        """
        # Ouverture de la source vidéo
        cap = self._open_source()
        if cap is None:
            return

        # Récupération des dimensions réelles de la vidéo
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        print(f"[INFO] Vidéo ouverte — Résolution : {frame_w}x{frame_h}  |  FPS source : {source_fps:.1f}")

        # Position Y de la ligne de comptage (en pixels absolus)
        self.line_y = int(frame_h * cfg.COUNTING_LINE_Y)

        # Initialisation de l'enregistreur vidéo (si activé)
        writer = self._init_writer(frame_w, frame_h, source_fps)

        paused = False

        # ── Boucle principale ────────────────────────────────
        while True:
            t_start = time.time()

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[INFO] Fin de la vidéo ou source inaccessible.")
                    break

                self.total_frames += 1

                # ① Détection avec YOLOv8
                detections_yolo = self._detect(frame)

                # ② Suivi avec DeepSORT
                tracks = self._track(frame, detections_yolo)

                # ③ Mise à jour des trajectoires, vitesses, compteurs
                self._update_trajectories(tracks, frame_h)

                # ④ Dessin sur la frame
                frame_annotated = self._draw(frame, tracks)

                # ⑤ Calcul et affichage des FPS
                elapsed = time.time() - t_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history)

                if cfg.SHOW_FPS:
                    self._draw_fps(frame_annotated, avg_fps)

                # ⑥ Enregistrement si activé
                if writer is not None:
                    writer.write(frame_annotated)

                # ⑦ Affichage
                display = self._resize_for_display(frame_annotated)
                cv2.imshow("Surveillance Trafic Douala — Module 1", display)

            # ── Gestion clavier ──────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):          # Q ou Echap → quitter
                print("\n[INFO] Arrêt demandé par l'utilisateur.")
                break
            elif key == ord('p'):              # P → pause/reprendre
                paused = not paused
                status = "PAUSE" if paused else "REPRISE"
                print(f"[INFO] {status}")
            elif key == ord('s'):              # S → capture d'écran
                self._save_screenshot(frame_annotated)

        # ── Nettoyage ────────────────────────────────────────
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        self._print_summary()

    # ────────────────────────────────────────────────────────
    # ① DÉTECTION — YOLOv8
    # ────────────────────────────────────────────────────────

    def _detect(self, frame):
        """
        Applique YOLOv8 sur la frame et retourne les détections filtrées.

        Returns:
            List de tuples : ([x1, y1, x2, y2], confidence, class_id)
        """
        results = self.model(
            frame,
            conf=cfg.CONFIDENCE_THRESHOLD,
            iou=cfg.IOU_THRESHOLD,
            classes=list(cfg.VEHICLE_CLASSES.keys()),  # Filtre sur véhicules seulement
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2, y2], confidence, class_id))

        return detections

    # ────────────────────────────────────────────────────────
    # ② SUIVI — DeepSORT
    # ────────────────────────────────────────────────────────

    def _track(self, frame, detections):
        """
        Passe les détections YOLOv8 à DeepSORT pour le suivi.

        DeepSORT attend le format : ([x1,y1,w,h], conf, class_id)
        On convertit donc xyxy → xywh.

        Returns:
            Liste de tracks actifs (objets DeepSORT Track)
        """
        # Conversion xyxy → xywh pour DeepSORT
        ds_input = []
        for (x1, y1, x2, y2), conf, class_id in detections:
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], conf, class_id))

        # Mise à jour du tracker
        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        # Filtre : on ne garde que les pistes confirmées
        confirmed = [t for t in tracks if t.is_confirmed()]
        return confirmed

    # ────────────────────────────────────────────────────────
    # ③ MISE À JOUR — Trajectoires, vitesses, comptage
    # ────────────────────────────────────────────────────────

    def _update_trajectories(self, tracks, frame_h):
        """
        Pour chaque piste confirmée :
        - Enregistre la position dans la trajectoire
        - Estime la vitesse par déplacement pixel
        - Compte les véhicules franchissant la ligne virtuelle
        """
        for track in tracks:
            tid = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            cx = int((ltrb[0] + ltrb[2]) / 2)  # Centre X
            cy = int((ltrb[1] + ltrb[3]) / 2)  # Centre Y

            # Enregistrement trajectoire
            self.trajectories[tid].append((cx, cy))

            # ── Estimation vitesse (déplacement pixel) ───────
            if tid in self.prev_positions:
                px, py = self.prev_positions[tid]
                dist_pixels = np.sqrt((cx - px)**2 + (cy - py)**2)
                # Note : pour une vitesse en km/h réelle, on multiplie
                # par le facteur de calibration f_px (cf. Module 2)
                # Ici on stocke le déplacement brut en pixels/frame
                self.speed_history[tid].append(dist_pixels)

            self.prev_positions[tid] = (cx, cy)

            # ── Comptage par ligne virtuelle ─────────────────
            if tid not in self.counted_ids:
                if len(self.trajectories[tid]) >= 2:
                    prev_cy = self.trajectories[tid][-2][1]
                    curr_cy = cy
                    # Détection de franchissement (sens haut→bas)
                    if prev_cy < self.line_y <= curr_cy:
                        self.counted_ids.add(tid)
                        class_id = track.det_class
                        if class_id in cfg.VEHICLE_CLASSES:
                            class_name = cfg.VEHICLE_CLASSES[class_id]
                            self.class_counter[class_name] += 1
                            self.total_vehicles += 1

    # ────────────────────────────────────────────────────────
    # ④ DESSIN — Annotations sur la frame
    # ────────────────────────────────────────────────────────

    def _draw(self, frame, tracks):
        """
        Dessine sur la frame :
        - Boîtes englobantes colorées par classe
        - ID du véhicule + classe + vitesse estimée
        - Trajectoires
        - Ligne de comptage
        - Tableau de bord (compteurs)
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── Ligne de comptage ────────────────────────────────
        cv2.line(annotated,
                 (0, self.line_y),
                 (w, self.line_y),
                 cfg.COUNTING_LINE_COLOR, 2)
        cv2.putText(annotated, "LIGNE DE COMPTAGE",
                    (10, self.line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    cfg.COUNTING_LINE_COLOR, 2)

        # ── Pistes confirmées ────────────────────────────────
        for track in tracks:
            tid = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = track.det_class

            # Couleur selon la classe
            color = cfg.CLASS_COLORS.get(class_id, (200, 200, 200))
            class_name = cfg.VEHICLE_CLASSES.get(class_id, "Véhicule")

            # Boîte englobante
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Vitesse estimée (pixels/frame)
            speed_str = ""
            if len(self.speed_history[tid]) > 0:
                avg_spd = np.mean(list(self.speed_history[tid])[-5:])
                speed_str = f"  {avg_spd:.1f}px/f"

            # Étiquette : classe + ID + vitesse
            label = f"{class_name} #{tid}{speed_str}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lw, lh = label_size

            # Fond de l'étiquette
            cv2.rectangle(annotated,
                          (x1, y1 - lh - 10),
                          (x1 + lw + 6, y1),
                          color, -1)
            cv2.putText(annotated, label,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # ── Trajectoire ──────────────────────────────────
            if cfg.SHOW_TRAJECTORIES:
                pts = list(self.trajectories[tid])
                for i in range(1, len(pts)):
                    alpha = i / len(pts)         # opacité progressive
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(annotated, pts[i-1], pts[i], c, 2)

        # ── Tableau de bord en haut à gauche ────────────────
        if cfg.SHOW_COUNTER:
            self._draw_dashboard(annotated)

        return annotated

    def _draw_dashboard(self, frame):
        """
        Affiche le tableau de bord avec les compteurs par classe
        et le nombre total de pistes actives.
        """
        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (310, 175), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Titre
        cv2.putText(frame, "TRAFIC DOUALA — MODULE 1",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 220, 0), 2)

        cv2.line(frame, (10, 35), (305, 35), (100, 100, 100), 1)

        # Compteurs par classe
        y = 58
        for class_id, class_name in cfg.VEHICLE_CLASSES.items():
            color = cfg.CLASS_COLORS.get(class_id, (200, 200, 200))
            count = self.class_counter.get(class_name, 0)
            cv2.putText(frame, f"{class_name[:18]:<22} {count:>4}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1)
            y += 22

        cv2.line(frame, (10, y), (305, y), (100, 100, 100), 1)
        y += 18

        # Total
        cv2.putText(frame,
                    f"TOTAL COMPTE       {self.total_vehicles:>4}",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 2)
        y += 22

        # Frame courante
        cv2.putText(frame, f"Frame #{self.total_frames}",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 150), 1)

    def _draw_fps(self, frame, fps):
        """Affiche les FPS en haut à droite."""
        h, w = frame.shape[:2]
        label = f"FPS: {fps:.1f}"
        cv2.putText(frame, label,
                    (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 100), 2)

    # ────────────────────────────────────────────────────────
    # UTILITAIRES
    # ────────────────────────────────────────────────────────

    def _open_source(self):
        """Ouvre la source vidéo et vérifie qu'elle est accessible."""
        # Conversion en entier si la source est un chiffre (webcam)
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"\n[ERREUR] Impossible d'ouvrir la source vidéo : {src}")
            if isinstance(src, int):
                print("         Vérifiez que votre webcam est connectée.")
                print("         Essayez --source 1 si la webcam intégrée est 0.")
            else:
                print(f"         Vérifiez que le fichier existe : {src}")
            return None

        print(f"[INFO] Source vidéo ouverte : {src}")
        return cap

    def _init_writer(self, w, h, fps):
        """Initialise l'enregistreur vidéo si SAVE_OUTPUT_VIDEO est True."""
        if not cfg.SAVE_OUTPUT_VIDEO:
            return None

        os.makedirs("outputs", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            cfg.OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
        print(f"[INFO] Enregistrement activé → {cfg.OUTPUT_VIDEO_PATH}")
        return writer

    def _resize_for_display(self, frame):
        """Redimensionne la frame pour l'affichage."""
        return cv2.resize(frame, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))

    def _save_screenshot(self, frame):
        """Sauvegarde une capture d'écran dans outputs/."""
        os.makedirs("outputs", exist_ok=True)
        fname = f"outputs/capture_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame)
        print(f"[INFO] Capture sauvegardée : {fname}")

    def _print_summary(self):
        """Affiche le résumé final en fin d'exécution."""
        print("\n" + "=" * 60)
        print("  RÉSUMÉ DE LA SESSION")
        print("=" * 60)
        print(f"  Frames traitées     : {self.total_frames}")
        print(f"  Véhicules comptés   : {self.total_vehicles}")
        print()
        for class_name, count in self.class_counter.items():
            print(f"    {class_name:<25} : {count}")
        if self.fps_history:
            print(f"\n  FPS moyen           : {np.mean(self.fps_history):.1f}")
        print("=" * 60 + "\n")


# ================================================================
# POINT D'ENTRÉE
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Module 1 — Détection & Suivi YOLOv8 + DeepSORT")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source vidéo : chemin fichier, 0 (webcam), 1 (webcam USB), URL RTSP"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modèle YOLOv8 : yolov8n.pt, yolov8s.pt, yolov8m.pt"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Seuil de confiance (défaut : 0.35)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Surcharge de la config par les arguments CLI
    if args.source is not None:
        cfg.VIDEO_SOURCE = args.source
    if args.model is not None:
        cfg.MODEL_PATH = args.model
    if args.conf is not None:
        cfg.CONFIDENCE_THRESHOLD = args.conf

    # Lancement du système
    detector = TrafficDetector()
    detector.run()
