# ============================================================
# config.py — Configuration pour le modèle FINE-TUNÉ
# Dataset : lynkeus/vehicle-detection-mgjdd
# ============================================================

# Source vidéo
VIDEO_SOURCE = 0  # 0=webcam, ou chemin vers votre vidéo

# Modèle fine-tuné (remplace yolov8n.pt)
MODEL_PATH = "models/yolov8_douala_best.pt"

# Seuil de confiance
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Classes du modèle fine-tuné (lynkeus dataset)
# IDs dans l'ordre du data.yaml
VEHICLE_CLASSES = {
    0: "Bus", 1: "Car", 2: "Microbus", 3: "Motorbike", 4: "Pickup-van", 5: "Truck"
}

# Toutes les classes sont des véhicules
DETECT_ALL_CLASSES = True  # Si True, détecte toutes les classes du modèle

# Couleurs (BGR)
CLASS_COLORS = {
    0: (0, 200, 0),      # car        — vert
    1: (0, 165, 255),    # motorbike  — orange
    2: (255, 50, 50),    # bus        — bleu
    3: (0, 0, 220),      # truck      — rouge
    4: (255, 0, 200),    # microbus   — violet
    5: (255, 200, 0),    # pickup-van — cyan
}

# DeepSORT
MAX_AGE = 30
N_INIT  = 3
MAX_COSINE_DISTANCE = 0.4
NN_BUDGET = 100

# Affichage
SHOW_FPS          = True
SHOW_TRAJECTORIES = True
TRAJECTORY_LENGTH = 40
SHOW_COUNTER      = True
DISPLAY_WIDTH     = 1280
DISPLAY_HEIGHT    = 720

# Enregistrement
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = "outputs/detection_finetuned.mp4"

# Ligne de comptage
COUNTING_LINE_Y     = 0.55
COUNTING_LINE_COLOR = (0, 255, 255)
