"""
================================================================
SYSTÈME DE SURVEILLANCE INTELLIGENTE DU TRAFIC — DOUALA
Module 2 : Paramètres de Trafic + Heatmaps
================================================================
Calcule : densité (k), débit (q), vitesse (v) en km/h
Génère  : heatmaps de congestion, graphiques temporels
================================================================
"""

import cv2
import numpy as np
import time
import os
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime


class TrafficAnalyzer:
    """
    Calcule les paramètres de trafic depuis les pistes DeepSORT.

    Formules utilisées :
        k = N / L              (densité : véhicules/km)
        q = ΔN / Δt            (débit   : véhicules/heure)
        v = (d_px/f_px)*(fps/t_frames)*3.6  (vitesse : km/h)
        q = k × v              (relation fondamentale vérifiée)
    """

    def __init__(self, frame_width, frame_height, fps=25.0,
                 road_length_m=50.0, px_per_meter=None):
        """
        Args:
            frame_width   : largeur de la frame en pixels
            frame_height  : hauteur de la frame en pixels
            fps           : images par seconde de la vidéo
            road_length_m : longueur réelle du tronçon surveillé (mètres)
            px_per_meter  : facteur de calibration (pixels/mètre)
                            Si None → estimé automatiquement
        """
        self.W = frame_width
        self.H = frame_height
        self.fps = fps
        self.road_length_m = road_length_m

        # Facteur de calibration pixels → mètres
        # Estimation par défaut : toute la hauteur de la frame = road_length_m
        if px_per_meter is None:
            self.px_per_meter = frame_height / road_length_m
        else:
            self.px_per_meter = px_per_meter

        # ── Ligne de comptage principale (55% hauteur) ───────
        self.line_y     = int(frame_height * 0.55)
        self.line_color = (0, 255, 255)

        # ── Lignes secondaires (entrée/sortie zone) ──────────
        self.zone_top    = int(frame_height * 0.30)
        self.zone_bottom = int(frame_height * 0.80)

        # ── Données de comptage ──────────────────────────────
        self.counted_ids     = set()           # IDs déjà comptés
        self.class_counts    = defaultdict(int)# Comptage par classe
        self.total_count     = 0

        # ── Historique temporel (fenêtre glissante 60s) ──────
        self.window_size = int(fps * 60)       # 60 secondes
        self.crossing_times = deque()          # Timestamps des passages

        # ── Vitesses par véhicule ────────────────────────────
        self.speed_history   = defaultdict(deque)  # {id: deque([v1,v2,...])}
        self.prev_positions  = {}                  # {id: (x,y,t)}
        self.vehicle_speeds  = {}                  # {id: vitesse_moy}

        # ── Heatmap ──────────────────────────────────────────
        self.heatmap_accum   = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.heatmap_decay   = 0.995  # Décroissance légère pour effet temporel

        # ── Historique pour graphiques ────────────────────────
        self.history_flow    = deque(maxlen=300)  # débit par minute
        self.history_density = deque(maxlen=300)  # densité
        self.history_speed   = deque(maxlen=300)  # vitesse moyenne
        self.history_time    = deque(maxlen=300)  # timestamps

        # ── Métriques courantes ───────────────────────────────
        self.current_flow    = 0.0   # véhicules/heure
        self.current_density = 0.0   # véhicules/km
        self.current_speed   = 0.0   # km/h
        self.vehicles_in_zone= 0     # véhicules dans la zone

        # ── Timer pour mise à jour périodique ────────────────
        self.last_update_t   = time.time()
        self.update_interval = 5.0   # Recalcul toutes les 5 secondes

        # ── Dossier de sortie ─────────────────────────────────
        os.makedirs("outputs", exist_ok=True)

        print("[Module 2] TrafficAnalyzer initialisé")
        print(f"  Calibration : {self.px_per_meter:.1f} px/m")
        print(f"  Tronçon     : {road_length_m} m")
        print(f"  Ligne comptage Y = {self.line_y} px")

    # ────────────────────────────────────────────────────────
    # MISE À JOUR PRINCIPALE
    # ────────────────────────────────────────────────────────

    def update(self, tracks, frame_idx):
        """
        Met à jour toutes les métriques depuis les pistes actives.

        Args:
            tracks    : liste de pistes DeepSORT confirmées
            frame_idx : numéro de la frame courante
        """
        current_time = time.time()
        vehicles_in_zone = 0

        for track in tracks:
            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ── Heatmap : accumulation des positions ─────────
            if 0 <= cy < self.H and 0 <= cx < self.W:
                self.heatmap_accum[cy, cx] += 1.0

            # ── Comptage zone ─────────────────────────────────
            if self.zone_top <= cy <= self.zone_bottom:
                vehicles_in_zone += 1

                ######### MODIFIE ICI 
"""
            # ── Estimation vitesse ────────────────────────────
            if tid in self.prev_positions:
                px, py, pt = self.prev_positions[tid]
                dt_frames = frame_idx - pt
                if dt_frames > 0:
                    # Distance en pixels
                    d_px = np.sqrt((cx - px)**2 + (cy - py)**2)
                    # Conversion en km/h
                    # v = (d_px / px_per_meter) / (dt_frames / fps) * 3.6
                    v_ms = (d_px / self.px_per_meter) / (dt_frames / self.fps)
                    v_kmh = v_ms * 3.6

                    # Filtre : vitesse réaliste (0-150 km/h)
                    if 0 < v_kmh < 150:
                        self.speed_history[tid].append(v_kmh)
                        if len(self.speed_history[tid]) > 10:
                            self.speed_history[tid].popleft()
                        self.vehicle_speeds[tid] = float(
                            np.mean(list(self.speed_history[tid])))

            self.prev_positions[tid] = (cx, cy, frame_idx)

            # ── Comptage franchissement ligne ─────────────────
            if tid not in self.counted_ids:
                if tid in self.prev_positions:
                    prev_cy = self.prev_positions.get(tid, (0, cy, 0))[1]
                    if prev_cy < self.line_y <= cy or cy < self.line_y <= prev_cy:
                        self.counted_ids.add(tid)
                        self.total_count += 1
                        cls_id = track.det_class
                        self.class_counts[cls_id] += 1
                        self.crossing_times.append(current_time)
""" 
        ### FIN ICI 
            # NOUVEAU 

            # ── Position précédente ───────────────────────────
            prev_data = self.prev_positions.get(tid, None)

            # ── Estimation vitesse ────────────────────────────
            if prev_data is not None:
                px, py, pt = prev_data
                dt_frames = frame_idx - pt

                if dt_frames > 0:
                    d_px = np.sqrt((cx - px)**2 + (cy - py)**2)

                    v_ms = (d_px / self.px_per_meter) / (dt_frames / self.fps)
                    v_kmh = v_ms * 3.6

                    if 0 < v_kmh < 150:
                        self.speed_history[tid].append(v_kmh)

                        if len(self.speed_history[tid]) > 10:
                            self.speed_history[tid].popleft()

                        self.vehicle_speeds[tid] = float(
                            np.mean(list(self.speed_history[tid]))
                        )

            # ── Comptage franchissement ligne ─────────────────
            if tid not in self.counted_ids and prev_data is not None:
                prev_cy = prev_data[1]

                if (prev_cy < self.line_y <= cy) or \
                (cy < self.line_y <= prev_cy):

                    self.counted_ids.add(tid)
                    self.total_count += 1

                    cls_id = track.det_class
                    self.class_counts[cls_id] += 1

                    self.crossing_times.append(current_time)

            # ── Sauvegarde position actuelle ──────────────────
            self.prev_positions[tid] = (cx, cy, frame_idx)

            # fin nouvelle 

        self.vehicles_in_zone = vehicles_in_zone

        # ── Décroissance heatmap ──────────────────────────────
        self.heatmap_accum *= self.heatmap_decay

        # ── Recalcul métriques toutes les N secondes ─────────
        if current_time - self.last_update_t >= self.update_interval:
            self._recalculate_metrics(current_time)
            self.last_update_t = current_time

    # ────────────────────────────────────────────────────────
    # CALCUL DES MÉTRIQUES
    # ────────────────────────────────────────────────────────

    def _recalculate_metrics(self, current_time):
        """
        Recalcule débit, densité et vitesse moyenne.

        k = N / L
        q = ΔN / Δt  (converti en véh/heure)
        v = moyenne des vitesses individuelles
        """
        # Nettoyer les passages trop anciens (fenêtre 60s)
        cutoff = current_time - 60.0
        while self.crossing_times and self.crossing_times[0] < cutoff:
            self.crossing_times.popleft()

        # Débit q (véhicules/heure)
        n_last_minute = len(self.crossing_times)
        self.current_flow = n_last_minute * 60.0  # extrapolation horaire

        # Densité k (véhicules/km)
        road_km = self.road_length_m / 1000.0
        self.current_density = (
            self.vehicles_in_zone / road_km if road_km > 0 else 0)

        # Vitesse moyenne v (km/h)
        speeds = [v for v in self.vehicle_speeds.values() if v > 0]
        self.current_speed = float(np.mean(speeds)) if speeds else 0.0

        # Vérification relation fondamentale q = k × v
        q_check = self.current_density * self.current_speed
        # (log interne — non affiché mais vérifiable)

        # Enregistrement historique
        self.history_flow.append(self.current_flow)
        self.history_density.append(self.current_density)
        self.history_speed.append(self.current_speed)
        self.history_time.append(datetime.now().strftime("%H:%M:%S"))

    # ────────────────────────────────────────────────────────
    # DESSIN SUR LA FRAME
    # ────────────────────────────────────────────────────────

    def draw(self, frame, tracks):
        """
        Dessine les éléments du Module 2 sur la frame :
        - Zone de surveillance
        - Ligne de comptage
        - Vitesse individuelle par véhicule
        - Tableau de bord métriques
        - Overlay heatmap
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # ── Zone de surveillance (rectangle semi-transparent) ─
        cv2.rectangle(overlay,
                      (0, self.zone_top),
                      (w, self.zone_bottom),
                      (255, 200, 0), -1)
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)

        # Bordures de zone
        cv2.line(frame, (0, self.zone_top), (w, self.zone_top),
                 (255, 200, 0), 1)
        cv2.line(frame, (0, self.zone_bottom), (w, self.zone_bottom),
                 (255, 200, 0), 1)

        # ── Ligne de comptage principale ──────────────────────
        cv2.line(frame, (0, self.line_y), (w, self.line_y),
                 self.line_color, 2)
        cv2.putText(frame, f"COMPTAGE — {self.total_count} véhicules",
                    (10, self.line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    self.line_color, 2)

        # ── Vitesse par véhicule ──────────────────────────────
        for track in tracks:
            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            if tid in self.vehicle_speeds:
                spd = self.vehicle_speeds[tid]
                # Couleur selon vitesse
                if spd < 20:
                    spd_color = (0, 0, 255)    # Rouge = lent / embouteillage
                elif spd < 50:
                    spd_color = (0, 165, 255)  # Orange = modéré
                else:
                    spd_color = (0, 200, 0)    # Vert = fluide

                cv2.putText(frame, f"{spd:.0f} km/h",
                            (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            spd_color, 2)

        # ── Tableau de bord métriques ─────────────────────────
        self._draw_metrics_panel(frame)

        return frame

    def _draw_metrics_panel(self, frame):
        """Panneau de métriques en bas à droite."""
        h, w = frame.shape[:2]
        pw, ph = 310, 160
        px, py = w - pw - 10, h - ph - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 200, 200), 1)

        cv2.putText(frame, "PARAMÈTRES TRAFIC",
                    (px+10, py+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 220, 220), 2)
        cv2.line(frame, (px+5, py+30), (px+pw-5, py+30), (60,60,80), 1)

        metrics = [
            ("Débit  (q)", f"{self.current_flow:.0f} véh/h",
             self._flow_color()),
            ("Densité (k)", f"{self.current_density:.1f} véh/km",
             (200, 200, 0)),
            ("Vitesse (v)", f"{self.current_speed:.1f} km/h",
             self._speed_color()),
            ("Zone active", f"{self.vehicles_in_zone} véhicules",
             (150, 150, 255)),
            ("Total compté", f"{self.total_count}",
             (255, 255, 255)),
        ]

        for i, (label, value, color) in enumerate(metrics):
            y = py + 52 + i * 22
            cv2.putText(frame, f"{label:<14}", (px+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (170, 170, 170), 1)
            cv2.putText(frame, value, (px+160, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1)

    def _flow_color(self):
        q = self.current_flow
        if q > 800:   return (0, 0, 255)
        elif q > 400: return (0, 165, 255)
        else:         return (0, 200, 0)

    def _speed_color(self):
        v = self.current_speed
        if v < 15:    return (0, 0, 255)
        elif v < 40:  return (0, 165, 255)
        else:         return (0, 200, 0)

    # ────────────────────────────────────────────────────────
    # HEATMAP
    # ────────────────────────────────────────────────────────

    def get_heatmap_overlay(self, frame, alpha=0.45):
        """
        Génère et superpose la heatmap de densité sur la frame.

        Utilise un noyau gaussien σ=20 px pour le lissage spatial :
        H_smooth = H_accum * G_σ
        """
        if self.heatmap_accum.max() == 0:
            return frame

        # Lissage gaussien (σ = 20 pixels)
        hm = cv2.GaussianBlur(self.heatmap_accum, (0, 0), 20)

        # Normalisation 0-255
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
        hm_uint8 = hm_norm.astype(np.uint8)

        # Application colormap JET (bleu=faible, rouge=fort)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

        # Fusion avec la frame originale
        result = cv2.addWeighted(frame, 1 - alpha, hm_color, alpha, 0)
        return result

    # ────────────────────────────────────────────────────────
    # GÉNÉRATION DES GRAPHIQUES
    # ────────────────────────────────────────────────────────

    def generate_plots(self, save_path="outputs/traffic_metrics.png"):
        """
        Génère les graphiques temporels des métriques de trafic.
        Sauvegarde dans outputs/traffic_metrics.png
        """
        if len(self.history_flow) < 2:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.patch.set_facecolor('#0A1931')

        times = list(self.history_time)
        x = range(len(times))

        # Débit
        ax1 = axes[0]
        ax1.set_facecolor('#0D2136')
        flow_data = list(self.history_flow)
        ax1.fill_between(x, flow_data, alpha=0.3, color='#00C8C8')
        ax1.plot(x, flow_data, color='#00C8C8', linewidth=2)
        ax1.set_ylabel('Débit q (véh/h)', color='white', fontsize=11)
        ax1.set_title('Débit du Trafic — Ndokoti, Douala',
                      color='#FFD700', fontsize=13, fontweight='bold')
        ax1.tick_params(colors='gray')
        ax1.spines[:].set_color('#1F3A5A')
        ax1.yaxis.label.set_color('white')
        # Seuils de congestion
        ax1.axhline(y=600, color='orange', linestyle='--',
                    alpha=0.6, label='Seuil modéré (600)')
        ax1.axhline(y=900, color='red', linestyle='--',
                    alpha=0.6, label='Seuil congestionné (900)')
        ax1.legend(facecolor='#0D2136', labelcolor='white', fontsize=9)

        # Densité
        ax2 = axes[1]
        ax2.set_facecolor('#0D2136')
        dens_data = list(self.history_density)
        ax2.fill_between(x, dens_data, alpha=0.3, color='#FFD700')
        ax2.plot(x, dens_data, color='#FFD700', linewidth=2)
        ax2.set_ylabel('Densité k (véh/km)', color='white', fontsize=11)
        ax2.set_title('Densité du Trafic',
                      color='#FFD700', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='gray')
        ax2.spines[:].set_color('#1F3A5A')

        # Vitesse
        ax3 = axes[2]
        ax3.set_facecolor('#0D2136')
        spd_data = list(self.history_speed)
        colors_spd = ['#FF4444' if v < 20 else
                      '#FFA500' if v < 50 else '#00C800'
                      for v in spd_data]
        ax3.bar(x, spd_data, color=colors_spd, alpha=0.8, width=0.8)
        ax3.set_ylabel('Vitesse moy. v (km/h)', color='white', fontsize=11)
        ax3.set_title('Vitesse Moyenne des Véhicules',
                      color='#FFD700', fontsize=13, fontweight='bold')
        ax3.tick_params(colors='gray')
        ax3.spines[:].set_color('#1F3A5A')
        # Légende couleurs vitesse
        patches = [
            mpatches.Patch(color='#FF4444', label='< 20 km/h (embouteillage)'),
            mpatches.Patch(color='#FFA500', label='20-50 km/h (dense)'),
            mpatches.Patch(color='#00C800', label='> 50 km/h (fluide)'),
        ]
        ax3.legend(handles=patches, facecolor='#0D2136',
                   labelcolor='white', fontsize=9)

        # Ticks X (afficher quelques timestamps)
        for ax in axes:
            step = max(1, len(times) // 8)
            ax.set_xticks(list(x)[::step])
            ax.set_xticklabels(times[::step], rotation=30,
                               color='gray', fontsize=8)

        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0A1931')
        plt.close()
        print(f"[Module 2] Graphiques sauvegardés : {save_path}")

    def save_heatmap(self, save_path="outputs/heatmap_densite.png"):
        """Sauvegarde la heatmap de densité en image haute résolution."""
        if self.heatmap_accum.max() == 0:
            return
        hm = cv2.GaussianBlur(self.heatmap_accum, (0, 0), 25)
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
        hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8),
                                     cv2.COLORMAP_JET)
        cv2.imwrite(save_path, hm_color)
        print(f"[Module 2] Heatmap sauvegardée : {save_path}")

    def get_summary(self):
        """Retourne un résumé des métriques sous forme de dictionnaire."""
        return {
            "total_vehicules"  : self.total_count,
            "debit_veh_heure"  : round(self.current_flow, 1),
            "densite_veh_km"   : round(self.current_density, 2),
            "vitesse_moy_kmh"  : round(self.current_speed, 1),
            "vehicules_en_zone": self.vehicles_in_zone,
            "comptage_classes" : dict(self.class_counts),
            "relation_q_kv"    : round(
                self.current_density * self.current_speed, 1),
        }
