"""
================================================================
SYSTÈME DE SURVEILLANCE INTELLIGENTE DU TRAFIC — DOUALA
Module 3 : Analyse Comportementale
================================================================
Détecte :
  - Fatigue conducteur   : EAR (Eye Aspect Ratio) via MediaPipe
  - Freinage brusque     : a_i(t) < -4 m/s²
  - Dépassement dangereux: d_lat < 1.5m ET v_rel > 20 km/h
  - Pose de tête         : déviation > 30° (distraction)
================================================================
"""

import cv2
import numpy as np
import time
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum

# ── Import MediaPipe (optionnel) ─────────────────────────────
# Compatibilité MediaPipe < 0.10 (mp.solutions) et >= 0.10 (mp.tasks)
MEDIAPIPE_OK = False
mp = None
_MEDIAPIPE_LEGACY = False

try:
    import mediapipe as _mp
    mp = _mp

    # Tentative d'initialisation de FaceMesh classique
    try:
        from mediapipe.solutions import face_mesh as _mp_face_mesh  # noqa
        MEDIAPIPE_OK = True
        _MEDIAPIPE_LEGACY = True
        print("[Module 3] MediaPipe (API classique) détecté ✅")
    except Exception:
        # API MediaPipe moderne détectée, mais EAR nécessite l'ancienne FaceMesh
        MEDIAPIPE_OK = False
        _MEDIAPIPE_LEGACY = False
        print("[Module 3] MediaPipe détecté, mais FaceMesh classique indisponible.")
        print("           Installez : pip install mediapipe==0.10.9")
        print("           Analyse fatigue désactivée — reste actif.")

except ImportError:
    _MEDIAPIPE_LEGACY = False
    print("[Module 3] ⚠️  MediaPipe non installé.")
    print("           Installez : pip install mediapipe==0.10.9")
    print("           Analyse fatigue désactivée — reste actif.")


class AlertLevel(Enum):
    INFO    = "INFO"
    WARNING = "WARNING"
    DANGER  = "DANGER"


class Alert:
    """Représente une alerte comportementale."""
    def __init__(self, level, category, message, track_id=None):
        self.level     = level
        self.category  = category
        self.message   = message
        self.track_id  = track_id
        self.timestamp = datetime.now()
        self.displayed = 0  # Compteur d'affichage

    @property
    def color_bgr(self):
        return {
            AlertLevel.INFO   : (0, 200, 100),
            AlertLevel.WARNING: (0, 165, 255),
            AlertLevel.DANGER : (0, 0, 255),
        }[self.level]

    def __str__(self):
        ts = self.timestamp.strftime("%H:%M:%S")
        tid = f" [ID#{self.track_id}]" if self.track_id else ""
        return f"[{ts}] {self.level.value} — {self.category}{tid}: {self.message}"


# ================================================================
# DÉTECTEUR DE FATIGUE (EAR)
# ================================================================

class FatigueDetector:
    """
    Détecte la fatigue du conducteur via l'Eye Aspect Ratio (EAR).

    Formule :
        EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)

    Alerte si EAR < 0.25 pendant ≥ 48 frames consécutives (~2s à 24fps)

    Indices MediaPipe FaceMesh pour les yeux :
        Œil gauche  : p1=33,  p2=160, p3=158, p4=133, p5=153, p6=144
        Œil droit   : p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
    """

    # Landmarks MediaPipe FaceMesh pour chaque œil
    LEFT_EYE  = [33,  160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Landmarks pour la pose de tête (nez, menton, oreilles)
    NOSE_TIP   = 1
    CHIN       = 152
    LEFT_EAR   = 234
    RIGHT_EAR  = 454
    LEFT_EYE_C = 33
    RIGHT_EYE_C= 263

    def __init__(self,
                 ear_threshold=0.25,
                 consec_frames=48,
                 head_angle_threshold=30.0):
        """
        Args:
            ear_threshold       : seuil EAR (< 0.25 = œil fermé)
            consec_frames       : frames consécutives avant alerte (~2s)
            head_angle_threshold: angle tête avant distraction (degrés)
        """
        self.ear_thresh  = ear_threshold
        self.consec_thresh = consec_frames
        self.angle_thresh= head_angle_threshold

        self.ear_counter  = 0   # Frames consécutives avec EAR bas
        self.total_blinks = 0
        self.fatigue_alerts = 0

        self.current_ear      = 0.0
        self.head_angle       = 0.0
        self.is_fatigued      = False
        self.is_distracted    = False
        self.alerted_fatigue  = False
        self.alerted_distracted = False

        self.face_mesh = None
        if MEDIAPIPE_OK and _MEDIAPIPE_LEGACY:
            try:
                from mediapipe.solutions.face_mesh import FaceMesh
                self.face_mesh = FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[Module 3] FaceMesh MediaPipe initialisé ✅")
            except Exception as e:
                print(f"[Module 3] ⚠️  Erreur init FaceMesh : {e}")
                self.face_mesh = None

        if self.face_mesh is None:
            print("[Module 3] FaceMesh désactivé (MediaPipe absent, trop récent ou import échoué)")
            print("           → pip install mediapipe==0.10.9  pour activer la détection fatigue")

    def _euclidean(self, p1, p2):
        """Distance euclidienne entre deux points 2D."""
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _compute_ear(self, landmarks, eye_indices, w, h):
        """
        Calcule l'Eye Aspect Ratio pour un œil.

        EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)
        """
        pts = []
        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append((lm.x * w, lm.y * h))

        # Distances verticales
        A = self._euclidean(pts[1], pts[5])  # ‖p2-p6‖
        B = self._euclidean(pts[2], pts[4])  # ‖p3-p5‖
        # Distance horizontale
        C = self._euclidean(pts[0], pts[3])  # ‖p1-p4‖

        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return ear

    def _compute_head_angle(self, landmarks, w, h):
        """
        Estime l'angle horizontal de la tête (yaw).
        Utilise la distance relative nez-oreilles.
        Retourne l'angle de déviation en degrés.
        """
        nose  = landmarks[self.NOSE_TIP]
        l_ear = landmarks[self.LEFT_EAR]
        r_ear = landmarks[self.RIGHT_EAR]

        nose_x  = nose.x
        l_ear_x = l_ear.x
        r_ear_x = r_ear.x

        # Ratio de position du nez entre les deux oreilles
        face_width = r_ear_x - l_ear_x
        if face_width < 0.01:
            return 0.0

        # Position relative du nez (0 = gauche, 0.5 = centre, 1 = droite)
        ratio = (nose_x - l_ear_x) / face_width
        # Conversion en angle (0.5 = 0°, déviation max ≈ 45°)
        angle = (ratio - 0.5) * 90.0
        return abs(angle)

    def process(self, frame):
        """
        Analyse une frame pour détecter fatigue et distraction.

        Returns:
            dict avec ear, head_angle, is_fatigued, is_distracted,
                       alert (Alert ou None)
        """
        result = {
            "ear"          : 0.0,
            "head_angle"   : 0.0,
            "is_fatigued"  : False,
            "is_distracted": False,
            "alert"        : None,
        }

        if not MEDIAPIPE_OK or self.face_mesh is None:
            return result

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        fm_result = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not fm_result.multi_face_landmarks:
            # Pas de visage détecté — réinitialiser compteur
            self.ear_counter = 0
            return result

        landmarks = fm_result.multi_face_landmarks[0].landmark

        # ── Calcul EAR ────────────────────────────────────────
        ear_l = self._compute_ear(landmarks, self.LEFT_EYE,  w, h)
        ear_r = self._compute_ear(landmarks, self.RIGHT_EYE, w, h)
        ear   = (ear_l + ear_r) / 2.0
        self.current_ear = ear
        result["ear"] = round(ear, 3)

        # ── Détection fatigue ─────────────────────────────────
        if ear < self.ear_thresh:
            self.ear_counter += 1
        else:
            if self.ear_counter >= 3:
                self.total_blinks += 1
            self.ear_counter = 0

        self.is_fatigued = self.ear_counter >= self.consec_thresh
        result["is_fatigued"] = self.is_fatigued

        if self.is_fatigued:
            if not self.alerted_fatigue:
                self.fatigue_alerts += 1
                self.alerted_fatigue = True
                result["alert"] = Alert(
                    AlertLevel.DANGER,
                    "FATIGUE",
                    f"Yeux fermés {self.ear_counter} frames "
                    f"(EAR={ear:.2f} < {self.ear_thresh})"
                )
        else:
            self.alerted_fatigue = False

        # ── Estimation angle tête ─────────────────────────────
        angle = self._compute_head_angle(landmarks, w, h)
        self.head_angle = angle
        result["head_angle"] = round(angle, 1)

        # ── Détection distraction ─────────────────────────────
        self.is_distracted = angle > self.angle_thresh
        result["is_distracted"] = self.is_distracted

        if self.is_distracted:
            if result["alert"] is None and not self.alerted_distracted:
                result["alert"] = Alert(
                    AlertLevel.WARNING,
                    "DISTRACTION",
                    f"Regard dévié de {angle:.0f}° "
                    f"(seuil={self.angle_thresh}°)"
                )
                self.alerted_distracted = True
        else:
            self.alerted_distracted = False

        # ── Dessin sur frame ──────────────────────────────────
        self._draw_fatigue_overlay(frame, landmarks, w, h, ear, angle)

        return result

    def _draw_fatigue_overlay(self, frame, landmarks, w, h, ear, angle):
        """Dessine les indicateurs EAR et angle sur la frame."""
        # Panneau fatigue en bas à gauche
        x, y = 5, frame.shape[0] - 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+280, y+115), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        color_ear = (0, 0, 255) if self.is_fatigued else (0, 200, 100)
        color_ang = (0, 0, 255) if self.is_distracted else (0, 200, 100)

        cv2.putText(frame, "ANALYSE CONDUCTEUR",
                    (x+8, y+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 220, 220), 2)

        cv2.putText(frame,
                    f"EAR       : {ear:.3f} "
                    f"({'ALERTE!' if self.is_fatigued else 'OK'})",
                    (x+8, y+42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, color_ear, 1)

        # Barre EAR
        bar_w = int(min(ear / 0.40, 1.0) * 200)
        cv2.rectangle(frame, (x+8, y+50), (x+210, y+60),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (x+8, y+50), (x+8+bar_w, y+60),
                      color_ear, -1)
        cv2.line(frame, (x+8+int(0.25/0.40*200), y+50),
                 (x+8+int(0.25/0.40*200), y+60), (255,255,0), 2)

        cv2.putText(frame,
                    f"Angle tête: {angle:.0f}° "
                    f"({'DISTRAIT!' if self.is_distracted else 'OK'})",
                    (x+8, y+78), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, color_ang, 1)

        cv2.putText(frame,
                    f"Clignements: {self.total_blinks}  "
                    f"Alertes: {self.fatigue_alerts}",
                    (x+8, y+100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (150, 150, 150), 1)


# ================================================================
# DÉTECTEUR D'ANOMALIES VÉHICULE
# ================================================================

class VehicleAnomalyDetector:
    """
    Détecte les anomalies comportementales des véhicules :
    - Freinage brusque  : a_i(t) < -4 m/s²
    - Dépassement dangereux : d_lat < 1.5m ET v_rel > 20 km/h
    """

    # Seuils
    BRAKING_THRESHOLD  = -4.0   # m/s² (freinage d'urgence)
    LATERAL_DIST_MIN   = 1.5    # mètres (proxmité latérale dangereuse)
    RELATIVE_SPEED_MIN = 20.0   # km/h (vitesse relative dangereuse)

    def __init__(self, fps=25.0, px_per_meter=10.0):
        self.fps          = fps
        self.px_per_meter = px_per_meter

        # Historique vitesses par track
        self.speed_kmh    = {}   # {tid: vitesse courante km/h}
        self.prev_speed   = {}   # {tid: vitesse précédente km/h}
        self.prev_center  = {}   # {tid: (cx, cy)}

        # Compteurs d'anomalies
        self.braking_events    = defaultdict(int)
        self.overtake_events   = 0
        self.total_braking     = 0

        # Historique alertes récentes (évite spam)
        self.recent_alerts     = {}   # {tid: last_alert_time}
        self.alert_cooldown    = 3.0  # secondes entre alertes du même véhicule

    def update(self, tracks, vehicle_speeds):
        """
        Analyse les pistes pour détecter les anomalies.

        Args:
            tracks         : pistes DeepSORT confirmées
            vehicle_speeds : dict {tid: vitesse km/h} du Module 2

        Returns:
            Liste d'Alert
        """
        alerts = []
        current_time = time.time()

        # Mise à jour vitesses
        for track in tracks:
            tid = track.track_id
            if tid in vehicle_speeds:
                self.speed_kmh[tid] = vehicle_speeds[tid]

        # ── Détection freinage brusque ────────────────────────
        for track in tracks:
            tid = track.track_id
            ltrb = track.to_ltrb()
            cx = int((ltrb[0] + ltrb[2]) / 2)
            cy = int((ltrb[1] + ltrb[3]) / 2)

            curr_v = self.speed_kmh.get(tid, 0.0)  # km/h

            if tid in self.prev_speed:
                prev_v = self.prev_speed[tid]
                # Conversion km/h → m/s
                dv = (curr_v - prev_v) / 3.6   # m/s
                dt = 1.0 / self.fps             # s
                # Accélération : a = Δv / Δt
                accel = dv / dt                 # m/s²

                if accel < self.BRAKING_THRESHOLD:
                    # Vérifier cooldown
                    last = self.recent_alerts.get(tid, 0)
                    if current_time - last > self.alert_cooldown:
                        self.braking_events[tid] += 1
                        self.total_braking += 1
                        self.recent_alerts[tid] = current_time
                        alerts.append(Alert(
                            AlertLevel.WARNING,
                            "FREINAGE BRUSQUE",
                            f"a={accel:.1f} m/s² < {self.BRAKING_THRESHOLD} m/s²",
                            track_id=tid
                        ))

            self.prev_speed[tid]  = curr_v
            self.prev_center[tid] = (cx, cy)

        # ── Détection dépassement dangereux ───────────────────
        track_list = list(tracks)
        for i in range(len(track_list)):
            for j in range(i+1, len(track_list)):
                ti, tj = track_list[i], track_list[j]
                id_i, id_j = ti.track_id, tj.track_id

                ltrb_i = ti.to_ltrb()
                ltrb_j = tj.to_ltrb()

                cx_i = (ltrb_i[0] + ltrb_i[2]) / 2
                cx_j = (ltrb_j[0] + ltrb_j[2]) / 2
                cy_i = (ltrb_i[1] + ltrb_i[3]) / 2
                cy_j = (ltrb_j[1] + ltrb_j[3]) / 2

                # Distance latérale (horizontale) en mètres
                d_lat_px = abs(cx_i - cx_j)
                d_lat_m  = d_lat_px / self.px_per_meter

                # Vitesse relative en km/h
                v_i = self.speed_kmh.get(id_i, 0.0)
                v_j = self.speed_kmh.get(id_j, 0.0)
                v_rel = abs(v_i - v_j)

                # Condition de dépassement dangereux
                if (d_lat_m < self.LATERAL_DIST_MIN and
                        v_rel > self.RELATIVE_SPEED_MIN):

                    pair_key = f"{min(id_i,id_j)}_{max(id_i,id_j)}"
                    last = self.recent_alerts.get(pair_key, 0)
                    if current_time - last > self.alert_cooldown:
                        self.overtake_events += 1
                        self.recent_alerts[pair_key] = current_time
                        alerts.append(Alert(
                            AlertLevel.DANGER,
                            "DÉPASSEMENT DANGEREUX",
                            f"d_lat={d_lat_m:.1f}m < {self.LATERAL_DIST_MIN}m"
                            f", v_rel={v_rel:.0f}km/h",
                            track_id=id_i
                        ))

        return alerts

    def draw_anomalies(self, frame, tracks):
        """Dessine les indicateurs d'anomalie sur les véhicules."""
        for track in tracks:
            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            n_brakes = self.braking_events.get(tid, 0)
            if n_brakes > 0:
                # Flash rouge autour du véhicule
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3),
                              (0, 0, 255), 3)
                cv2.putText(frame, f"⚠ {n_brakes}x",
                            (x1, y1-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255), 2)
        return frame

    def get_stats(self):
        return {
            "total_freinages"       : self.total_braking,
            "total_depassements"    : self.overtake_events,
            "vehicules_impliques"   : len(self.braking_events),
        }


# ================================================================
# GESTIONNAIRE D'ALERTES
# ================================================================

class AlertManager:
    """
    Centralise, affiche et enregistre toutes les alertes.
    """

    def __init__(self, max_display=5, log_path="outputs/alertes.log"):
        self.alerts       = deque(maxlen=100)  # Historique complet
        self.active       = deque(maxlen=max_display)  # Alertes à l'écran
        self.log_path     = log_path
        self.counts       = defaultdict(int)   # Par catégorie
        os.makedirs("outputs", exist_ok=True)
        # Initialiser le fichier log
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== LOG ALERTES — {datetime.now()} ===\n")

    def add(self, alert):
        if alert is None:
            return
        self.alerts.append(alert)
        self.active.append(alert)
        self.counts[alert.category] += 1
        # Écrire dans le log
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(str(alert) + "\n")
        print(f"[ALERTE] {alert}")

    def add_many(self, alert_list):
        for a in alert_list:
            self.add(a)

    def draw(self, frame):
        """Affiche les alertes actives en haut de la frame."""
        if not self.active:
            return frame

        x, y0 = 5, 5
        for i, alert in enumerate(list(self.active)[-5:]):
            y = y0 + i * 32
            # Fond alerte
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+500, y+28),
                          (20, 10, 10), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            # Bordure colorée
            cv2.rectangle(frame, (x, y), (x+500, y+28),
                          alert.color_bgr, 1)
            # Icône niveau
            icons = {AlertLevel.INFO: "ℹ",
                     AlertLevel.WARNING: "⚠",
                     AlertLevel.DANGER: "🚨"}
            # Texte
            ts = alert.timestamp.strftime("%H:%M:%S")
            tid_str = f" [#{alert.track_id}]" if alert.track_id else ""
            text = (f"{ts} {alert.level.value} "
                    f"{alert.category}{tid_str}: {alert.message}")
            cv2.putText(frame, text[:70], (x+6, y+19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        alert.color_bgr, 1)
        return frame

    def get_summary(self):
        return {
            "total_alertes": len(self.alerts),
            "par_categorie": dict(self.counts),
        }


import os  # nécessaire pour AlertManager
