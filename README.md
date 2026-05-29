# 🚦 Système de Surveillance Intelligente du Trafic — Douala
## Module 1 : Détection & Suivi (YOLOv8 + DeepSORT)

> Projet Tutoré 4ème année — Informatique / Intelligence Artificielle  
> Auteurs : BEKOU Adrien, NGUEBOU TEMGOUA Rayan  
> Encadrants : M. NDJE MAN D.F., M. KAZE Roger (3iL / Laboratoire MIA)

---

## 📁 Structure du projet

```
traffic_system/
│
├── module1_detection_suivi.py   ← SCRIPT PRINCIPAL (Module 1)
├── config.py                    ← Configuration (à modifier selon vos besoins)
├── download_video.py            ← Téléchargement de vidéos de test
├── requirements.txt             ← Dépendances Python
├── README.md                    ← Ce fichier
│
├── data/                        ← Vos vidéos de test (créé automatiquement)
├── outputs/                     ← Vidéos annotées + captures (créé automatiquement)
└── models/                      ← Modèles YOLOv8 (téléchargés automatiquement)
```

---

## ⚙️ Installation

### Étape 1 — Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Étape 2 — Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Note :** Le modèle YOLOv8n (~25 Mo) sera téléchargé automatiquement
> au premier lancement. Téléchargez-le à l'avance si votre connexion est limitée :
> ```bash
> python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
> ```

---

## 🚀 Utilisation

### Option A — Webcam (test rapide)

```bash
# Webcam intégrée (par défaut)
python module1_detection_suivi.py --source 0

# Webcam USB externe
python module1_detection_suivi.py --source 1
```

### Option B — Vidéo locale (recommandé)

```bash
# Avec votre vidéo de Douala
python module1_detection_suivi.py --source data/douala_ndokoti.mp4
```

### Option C — Télécharger une vidéo YouTube

```bash
# Étape 1 : Télécharger une vidéo de Douala
python download_video.py --url "https://www.youtube.com/watch?v=VOTRE_ID"

# Étape 2 : Lancer la détection
python module1_detection_suivi.py --source data/video_test.mp4
```

### Option D — Vidéo de démonstration synthétique

```bash
# Créer une vidéo de test sans connexion internet
python download_video.py --demo

# Puis tester
python module1_detection_suivi.py --source data/demo_traffic.mp4
```

---

## ⌨️ Contrôles pendant l'exécution

| Touche | Action                        |
|--------|-------------------------------|
| `Q`    | Quitter                       |
| `Échap`| Quitter                       |
| `P`    | Pause / Reprendre             |
| `S`    | Sauvegarder capture d'écran   |

---

## ⚙️ Configuration

Modifiez `config.py` pour adapter le système à votre environnement :

```python
VIDEO_SOURCE = "data/ma_video.mp4"   # Source vidéo
MODEL_PATH = "yolov8n.pt"            # Modèle YOLOv8
CONFIDENCE_THRESHOLD = 0.35          # Seuil de confiance
COUNTING_LINE_Y = 0.55               # Position de la ligne de comptage (55% de la hauteur)
SAVE_OUTPUT_VIDEO = True             # Enregistrer la vidéo annotée
```

---

## 📊 Ce que le Module 1 produit

- ✅ **Boîtes englobantes** colorées par classe de véhicule
- ✅ **ID unique** persistant pour chaque véhicule (DeepSORT)
- ✅ **Trajectoires** dessinées sur la vidéo
- ✅ **Comptage** par classe (voiture, moto, bus, camion)
- ✅ **FPS** en temps réel
- ✅ **Vitesse estimée** en pixels/frame (calibration km/h → Module 2)
- ✅ **Vidéo annotée** sauvegardée dans `outputs/`

---

## 🔜 Modules suivants

| Module | Contenu |
|--------|---------|
| **Module 2** | Paramètres de trafic (densité, débit, vitesse km/h) + Heatmaps |
| **Module 3** | Analyse comportementale (fatigue EAR, freinage brusque, dépassement) |
| **Module 4** | Dashboard Streamlit temps réel + export rapport |

---

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| `Webcam inaccessible` | Essayez `--source 1` au lieu de `0` |
| `YOLO trop lent` | Utilisez `yolov8n.pt` (nano) au lieu de `yolov8m.pt` |
| `Trop de faux positifs` | Augmentez `CONFIDENCE_THRESHOLD` à `0.50` dans `config.py` |
| `Véhicules non détectés` | Diminuez `CONFIDENCE_THRESHOLD` à `0.25` dans `config.py` |
| `Module deep_sort_realtime introuvable` | `pip install deep-sort-realtime` |
