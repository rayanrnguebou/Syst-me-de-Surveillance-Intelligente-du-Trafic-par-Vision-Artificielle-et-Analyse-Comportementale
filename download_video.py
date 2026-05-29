"""
================================================================
download_video.py — Téléchargement de vidéos de test
================================================================

Ce script télécharge des vidéos de trafic pour tester le système.
Il cherche automatiquement des vidéos de carrefours de Douala
sur YouTube, ou vous pouvez fournir une URL directe.

Utilisation :
    python download_video.py
    python download_video.py --url "URL_YOUTUBE"
    python download_video.py --demo   (vidéo de démonstration générique)
================================================================
"""

import argparse
import os
import sys


def download_youtube(url, output_path="data/video_test.mp4"):
    """Télécharge une vidéo YouTube avec yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        print("[ERREUR] yt-dlp non installé.")
        print("         Exécutez : pip install yt-dlp")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    print(f"\n[INFO] Téléchargement de la vidéo...")
    print(f"       URL : {url}")
    print(f"       Destination : {output_path}\n")

    ydl_opts = {
        # Qualité max 720p pour équilibrer qualité/taille
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"\n[OK] Vidéo téléchargée : {output_path}")
        print(f"     Pour lancer la détection :")
        print(f"     python module1_detection_suivi.py --source {output_path}\n")
        return output_path

    except Exception as e:
        print(f"\n[ERREUR] Échec du téléchargement : {e}")
        print("         Vérifiez votre connexion internet et l'URL.")
        return None


def create_demo_video(output_path="data/demo_traffic.mp4"):
    """
    Crée une vidéo de démonstration synthétique si aucune
    vidéo réelle n'est disponible.

    Génère des rectangles simulant des véhicules en mouvement
    sur un fond gris (route simulée), permettant de tester
    le pipeline sans vidéo réelle.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[ERREUR] opencv-python ou numpy non installé.")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    print("\n[INFO] Création d'une vidéo de démonstration synthétique...")
    print(f"       Destination : {output_path}")

    W, H = 1280, 720
    FPS = 25
    DURATION_S = 30  # 30 secondes

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    # Définition de véhicules simulés
    # Format : [x, y, largeur, hauteur, vitesse_x, vitesse_y, couleur, label]
    vehicles = [
        [100,  200, 80, 50,  3,  0, (0, 200, 0),   "Voiture"],
        [200,  350, 60, 40,  4,  0, (0, 165, 255), "Moto"],
        [400,  150, 100, 65, 2,  0, (255, 50, 50), "Bus"],
        [600,  450, 120, 70, 2,  0, (0, 0, 220),   "Camion"],
        [900,  300, 75, 48,  3,  0, (0, 200, 0),   "Voiture"],
        [50,   500, 65, 42,  5,  0, (0, 165, 255), "Moto"],
        [700,  100, 95, 60,  2,  0, (255, 50, 50), "Bus"],
        [300,  600, 80, 50, -3,  0, (0, 200, 0),   "Voiture"],
        [1100, 400, 60, 38, -4,  0, (0, 165, 255), "Moto"],
    ]

    np.random.seed(42)
    total_frames = FPS * DURATION_S

    for frame_idx in range(total_frames):
        # Fond : route gris foncé
        frame = np.full((H, W, 3), 60, dtype=np.uint8)

        # Marquages routiers
        cv2.line(frame, (0, H//2), (W, H//2), (80, 80, 80), 3)
        for x in range(0, W, 120):
            cv2.rectangle(frame, (x, H//2 - 2), (x + 60, H//2 + 2), (200, 200, 200), -1)

        # Mise à jour et dessin des véhicules
        for v in vehicles:
            x, y, w, h, vx, vy, color, label = v

            # Mouvement
            v[0] = (x + vx) % W
            v[1] = y + np.random.randint(-1, 2)  # légère variation verticale
            v[1] = max(50, min(H - 80, v[1]))

            # Dessin du véhicule
            cx, cy = int(v[0]), int(v[1])
            cv2.rectangle(frame,
                          (cx, cy),
                          (cx + w, cy + h),
                          color, -1)
            cv2.rectangle(frame,
                          (cx, cy),
                          (cx + w, cy + h),
                          (255, 255, 255), 1)
            cv2.putText(frame, label,
                        (cx + 4, cy + h//2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)

        # Texte de démonstration
        cv2.putText(frame, f"VIDEO DE DEMONSTRATION — Frame {frame_idx}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (200, 200, 0), 2)
        cv2.putText(frame,
                    "REMPLACEZ PAR UNE VRAIE VIDEO DE DOUALA",
                    (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (100, 100, 255), 1)

        writer.write(frame)

        # Progression
        if frame_idx % (FPS * 5) == 0:
            pct = frame_idx * 100 // total_frames
            print(f"  Progression : {pct}%")

    writer.release()
    print(f"\n[OK] Vidéo de démonstration créée : {output_path}")
    print(f"     Durée : {DURATION_S}s  |  {total_frames} frames  |  {W}x{H} px")
    print(f"\n     Pour tester :")
    print(f"     python module1_detection_suivi.py --source {output_path}\n")
    return output_path


# ── Suggestions de vidéos YouTube de Douala ─────────────────
SUGGESTED_URLS = [
    ("Embouteillage Douala Ndokoti",
     "https://www.youtube.com/results?search_query=embouteillage+douala+ndokoti"),
    ("Trafic carrefour Douala",
     "https://www.youtube.com/results?search_query=trafic+carrefour+douala+cameroun"),
    ("Circulation Bonamoussadi",
     "https://www.youtube.com/results?search_query=circulation+bonamoussadi+douala"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargement de vidéos de trafic pour les tests")
    parser.add_argument("--url", type=str, default=None,
                        help="URL YouTube directe à télécharger")
    parser.add_argument("--demo", action="store_true",
                        help="Créer une vidéo de démonstration synthétique")
    parser.add_argument("--output", type=str, default="data/video_test.mp4",
                        help="Chemin de sortie (défaut: data/video_test.mp4)")
    args = parser.parse_args()

    if args.demo:
        create_demo_video(args.output)

    elif args.url:
        download_youtube(args.url, args.output)

    else:
        print("\n" + "=" * 60)
        print("  TÉLÉCHARGEUR DE VIDÉOS — Trafic Douala")
        print("=" * 60)
        print("\nOptions :")
        print("  1. Télécharger depuis YouTube :")
        print("     python download_video.py --url <URL_YOUTUBE>")
        print("\n  2. Créer une vidéo de démonstration synthétique :")
        print("     python download_video.py --demo")
        print("\nRecherches YouTube suggérées pour Douala :")
        for label, url in SUGGESTED_URLS:
            print(f"\n  [{label}]")
            print(f"  → {url}")
        print("\n  Copiez l'URL d'une vraie vidéo puis :")
        print("  python download_video.py --url <URL_COPIEE>\n")


if __name__ == "__main__":
    main()
