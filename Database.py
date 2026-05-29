"""
================================================================
SYSTÈME DE SURVEILLANCE TRAFIC — DOUALA
database.py : Gestionnaire MongoDB + Sécurité
================================================================
Fonctionnalités :
  - Connexion MongoDB sécurisée (TLS optionnel)
  - Collections : vehicules, metriques, alertes, users, sessions
  - Anonymisation automatique des plaques (SHA-256)
  - Chiffrement des données sensibles (Fernet)
  - Fallback fichier JSON si MongoDB indisponible
================================================================
"""

import os
import json
import hashlib
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict

# ── MongoDB ──────────────────────────────────────────────────
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_OK = True
except ImportError:
    MONGO_OK = False
    print("[DB] ⚠️  pymongo non installé → pip install pymongo")

# ── Chiffrement ───────────────────────────────────────────────
try:
    from cryptography.fernet import Fernet
    CRYPTO_OK = True
except ImportError:
    CRYPTO_OK = False
    print("[DB] ⚠️  cryptography non installé → pip install cryptography")

# ── Authentification JWT ──────────────────────────────────────
try:
    import jwt
    import bcrypt
    JWT_OK = True
except ImportError:
    JWT_OK = False
    print("[DB] ⚠️  jwt/bcrypt non installés → pip install PyJWT bcrypt")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ================================================================
# GESTIONNAIRE DE SÉCURITÉ
# ================================================================

class SecurityManager:
    """
    Gère la sécurité des données :
    - Anonymisation des plaques (SHA-256 + salt)
    - Chiffrement Fernet des données sensibles
    - Hachage bcrypt des mots de passe
    - Tokens JWT pour les sessions
    """

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.environ.get(
            "TRAFFIC_SECRET", "douala_traffic_secret_2025_mia")
        self.jwt_secret = self.secret_key + "_jwt"
        self.plate_salt = self.secret_key + "_plates"

        # Clé de chiffrement Fernet
        if CRYPTO_OK:
            key_bytes = hashlib.sha256(
                self.secret_key.encode()).digest()
            import base64
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            self.fernet = Fernet(fernet_key)
        else:
            self.fernet = None

    def anonymize_plate(self, plate: str) -> str:
        """
        Anonymise une plaque d'immatriculation par hachage SHA-256.
        La plaque originale ne peut pas être retrouvée.

        Ex: "LT 1234 A" → "anon_a3f4b2c1..."
        """
        if not plate:
            return "anon_unknown"
        salted = (self.plate_salt + plate.upper().strip()).encode()
        hashed = hashlib.sha256(salted).hexdigest()[:16]
        return f"anon_{hashed}"

    def hash_password(self, password: str) -> str:
        """Hache un mot de passe avec bcrypt."""
        if not JWT_OK:
            # Fallback SHA-256 si bcrypt absent
            return hashlib.sha256(
                (password + self.secret_key).encode()).hexdigest()
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Vérifie un mot de passe contre son hash."""
        if not JWT_OK:
            expected = hashlib.sha256(
                (password + self.secret_key).encode()).hexdigest()
            return expected == hashed
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False

    def create_token(self, user_id: str, role: str,
                     expires_hours: int = 8) -> str:
        """Crée un token JWT pour une session administrateur."""
        if not JWT_OK:
            # Fallback simple
            payload = f"{user_id}:{role}:{time.time()}"
            return hashlib.sha256(
                (payload + self.jwt_secret).encode()).hexdigest()
        payload = {
            "sub"     : user_id,
            "role"    : role,
            "iat"     : datetime.utcnow(),
            "exp"     : datetime.utcnow() + timedelta(hours=expires_hours),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_token(self, token: str) -> dict:
        """
        Vérifie et décode un token JWT.
        Retourne le payload ou None si invalide/expiré.
        """
        if not JWT_OK:
            return {"sub": "admin", "role": "admin"}  # fallback
        try:
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("[SÉCURITÉ] Token expiré")
            return None
        except jwt.InvalidTokenError:
            logger.warning("[SÉCURITÉ] Token invalide")
            return None

    def encrypt(self, data: str) -> str:
        """Chiffre une chaîne sensible (Fernet AES-128)."""
        if not self.fernet:
            return data
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        """Déchiffre une chaîne chiffrée."""
        if not self.fernet:
            return data
        try:
            return self.fernet.decrypt(data.encode()).decode()
        except Exception:
            return data


# ================================================================
# GESTIONNAIRE BASE DE DONNÉES
# ================================================================

class DatabaseManager:
    """
    Interface centrale MongoDB pour le système de surveillance.

    Collections :
        vehicules  — passages détectés (horodatés)
        metriques  — flux trafic toutes les 5 secondes
        alertes    — événements comportementaux
        users      — comptes administrateurs
        sessions   — sessions de surveillance
    """

    def __init__(self,
                 uri: str = "mongodb://localhost:27017/",
                 db_name: str = "traffic_douala",
                 use_tls: bool = False):
        """
        Args:
            uri      : URI MongoDB (ex: "mongodb://localhost:27017/")
            db_name  : Nom de la base de données
            use_tls  : Activer TLS/SSL (recommandé en production)
        """
        self.uri     = uri
        self.db_name = db_name
        self.use_tls = use_tls
        self.security= SecurityManager()
        self.client  = None
        self.db      = None
        self.online  = False

        # Fallback JSON local si MongoDB indisponible
        self.fallback_dir = "outputs/db_fallback"
        os.makedirs(self.fallback_dir, exist_ok=True)

        # Buffer d'écriture (batch insert pour performance)
        self._buffer = defaultdict(list)
        self._buffer_size = 5         # Flush toutes les 5 entrées (temps réel)
        self._last_flush  = time.time()
        self._flush_interval = 3.0    # ou toutes les 3 secondes

        self._connect()

    def _connect(self):
        """Établit la connexion MongoDB."""
        if not MONGO_OK:
            logger.warning("[DB] pymongo absent — mode fichier JSON activé")
            return

        try:
            kwargs = {
                "serverSelectionTimeoutMS": 3000,
                "connectTimeoutMS"        : 3000,
            }
            if self.use_tls:
                kwargs["tls"] = True
                kwargs["tlsAllowInvalidCertificates"] = False
                logger.info("[DB] Connexion TLS activée")

            self.client = MongoClient(self.uri, **kwargs)
            # Test de connexion
            self.client.admin.command("ping")
            self.db     = self.client[self.db_name]
            self.online = True

            self._create_indexes()
            self._ensure_admin()
            logger.info(f"[DB] ✅ MongoDB connecté : {self.uri} → {self.db_name}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"[DB] ⚠️  MongoDB inaccessible ({e})")
            logger.warning("[DB]    Mode fichier JSON de secours activé")
            self.online = False

    def _create_indexes(self):
        """Crée les index pour optimiser les requêtes temporelles."""
        try:
            # Index TTL : suppression auto après 30 jours
            self.db.vehicules.create_index(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=30*24*3600,
                name="ttl_vehicules"
            )
            self.db.metriques.create_index(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=90*24*3600,
                name="ttl_metriques"
            )
            self.db.alertes.create_index(
                [("timestamp", ASCENDING)],
                name="idx_alertes_time"
            )
            self.db.alertes.create_index(
                [("niveau", ASCENDING)],
                name="idx_alertes_niveau"
            )
            self.db.users.create_index(
                [("username", ASCENDING)],
                unique=True, name="idx_users_unique"
            )
            logger.info("[DB] Index créés avec succès")
        except Exception as e:
            logger.warning(f"[DB] Index : {e}")

    def _ensure_admin(self):
        """Crée le compte admin par défaut s'il n'existe pas."""
        if self.db.users.count_documents({"username": "admin"}) == 0:
            self.create_user(
                username="admin",
                password="Admin@Douala2025",
                role="admin",
                full_name="Administrateur Système"
            )
            logger.info("[DB] Compte admin créé (admin / Admin@Douala2025)")

    # ────────────────────────────────────────────────────────
    # GESTION UTILISATEURS
    # ────────────────────────────────────────────────────────

    def create_user(self, username: str, password: str,
                    role: str = "operator",
                    full_name: str = "") -> bool:
        """
        Crée un compte utilisateur.

        Rôles disponibles :
            admin    — accès complet, gestion utilisateurs
            operator — lecture + acquittement alertes
            viewer   — lecture seule
        """
        doc = {
            "username"    : username.lower().strip(),
            "password_hash": self.security.hash_password(password),
            "role"        : role,
            "full_name"   : full_name,
            "created_at"  : datetime.utcnow(),
            "last_login"  : None,
            "active"      : True,
        }
        try:
            if self.online:
                self.db.users.insert_one(doc)
            else:
                self._fallback_write("users", doc)
            logger.info(f"[DB] Utilisateur créé : {username} ({role})")
            return True
        except Exception as e:
            logger.error(f"[DB] Erreur création user : {e}")
            return False

    def authenticate(self, username: str,
                     password: str) -> dict | None:
        """
        Authentifie un utilisateur et retourne un token JWT.

        Returns:
            dict {"token": ..., "user": ..., "role": ...}
            ou None si échec
        """
        try:
            if self.online:
                user = self.db.users.find_one(
                    {"username": username.lower(), "active": True})
            else:
                user = self._fallback_get_user(username)

            if not user:
                logger.warning(f"[AUTH] Utilisateur inconnu : {username}")
                return None

            if not self.security.verify_password(
                    password, user["password_hash"]):
                logger.warning(f"[AUTH] Mauvais mot de passe : {username}")
                return None

            # Mise à jour last_login
            if self.online:
                self.db.users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"last_login": datetime.utcnow()}}
                )

            token = self.security.create_token(
                str(user.get("_id", username)),
                user["role"]
            )
            logger.info(f"[AUTH] ✅ Connexion réussie : {username}")
            return {
                "token"    : token,
                "username" : username,
                "role"     : user["role"],
                "full_name": user.get("full_name", ""),
            }
        except Exception as e:
            logger.error(f"[AUTH] Erreur : {e}")
            return None

    # ────────────────────────────────────────────────────────
    # INSERTION DONNÉES
    # ────────────────────────────────────────────────────────

    def insert_vehicle(self, track_id: int, class_name: str,
                       speed_kmh: float, position: tuple,
                       plate: str = None,
                       intersection: str = "Ndokoti"):
        """
        Enregistre le passage d'un véhicule.
        La plaque est automatiquement anonymisée.
        """
        doc = {
            "timestamp"   : datetime.utcnow(),
            "track_id"    : track_id,
            "classe"      : class_name,
            "vitesse_kmh" : round(speed_kmh, 1),
            "position"    : {"x": position[0], "y": position[1]},
            "intersection": intersection,
            # Plaque anonymisée si fournie
            "plaque_anon" : self.security.anonymize_plate(plate)
                            if plate else None,
        }
        self._buffer_insert("vehicules", doc)

    def insert_metric(self, flow: float, density: float,
                      speed: float, vehicles_in_zone: int,
                      class_counts: dict,
                      intersection: str = "Ndokoti",
                      ear: float = None,
                      is_fatigued: bool = False):
        """Enregistre les métriques de trafic."""
        doc = {
            "timestamp"       : datetime.utcnow(),
            "intersection"    : intersection,
            "debit_veh_h"     : round(flow, 1),
            "densite_veh_km"  : round(density, 2),
            "vitesse_moy_kmh" : round(speed, 1),
            "vehicules_zone"  : vehicles_in_zone,
            "comptage_classes": class_counts,
            # Relation fondamentale q = k × v
            "q_kv_check"      : round(density * speed, 1),
            # Analyse comportementale conducteur
            "ear"             : round(ear, 3) if ear is not None else None,
            "is_fatigued"     : is_fatigued,
        }
        self._buffer_insert("metriques", doc)

    def insert_alert(self, level: str, category: str,
                     message: str, track_id: int = None,
                     intersection: str = "Ndokoti"):
        """Enregistre une alerte comportementale."""
        doc = {
            "timestamp"   : datetime.utcnow(),
            "intersection": intersection,
            "niveau"      : level,        # DANGER / WARNING / INFO
            "categorie"   : category,
            "message"     : message,
            "track_id"    : track_id,
            "acquittee"   : False,        # À valider par un opérateur
            "acquittee_par": None,
            "acquittee_at" : None,
        }
        # Les alertes DANGER sont insérées immédiatement (pas de buffer)
        if level == "DANGER":
            self._direct_insert("alertes", doc)
        else:
            self._buffer_insert("alertes", doc)

    # ────────────────────────────────────────────────────────
    # REQUÊTES DONNÉES
    # ────────────────────────────────────────────────────────

    def get_metrics_last_n(self, n: int = 60) -> list:
        """Retourne les N dernières métriques (temps réel)."""
        if not self.online:
            return self._fallback_read("metriques", n)
        cursor = self.db.metriques.find(
            {}, {"_id": 0}
        ).sort("timestamp", DESCENDING).limit(n)
        return list(cursor)

    def get_alerts(self, level: str = None,
                   unacknowledged_only: bool = False,
                   limit: int = 50) -> list:
        """Retourne les alertes selon les filtres."""
        query = {}
        if level:
            query["niveau"] = level
        if unacknowledged_only:
            query["acquittee"] = False

        if not self.online:
            return self._fallback_read("alertes", limit)
        cursor = self.db.alertes.find(
            query, {"_id": 0}
        ).sort("timestamp", DESCENDING).limit(limit)
        return list(cursor)

    def acknowledge_alert(self, alert_id: str,
                          operator: str) -> bool:
        """Acquitte une alerte (action opérateur)."""
        if not self.online:
            return False
        from bson import ObjectId
        result = self.db.alertes.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {
                "acquittee"    : True,
                "acquittee_par": operator,
                "acquittee_at" : datetime.utcnow(),
            }}
        )
        return result.modified_count > 0

    def get_stats_summary(self, hours: int = 24) -> dict:
        """Statistiques agrégées sur les N dernières heures."""
        since = datetime.utcnow() - timedelta(hours=hours)
        if not self.online:
            return {}

        pipeline = [
            {"$match": {"timestamp": {"$gte": since}}},
            {"$group": {
                "_id"            : None,
                "debit_moy"      : {"$avg": "$debit_veh_h"},
                "vitesse_moy"    : {"$avg": "$vitesse_moy_kmh"},
                "densite_moy"    : {"$avg": "$densite_veh_km"},
                "debit_max"      : {"$max": "$debit_veh_h"},
                "vitesse_min"    : {"$min": "$vitesse_moy_kmh"},
                "total_docs"     : {"$sum": 1},
            }}
        ]
        result = list(self.db.metriques.aggregate(pipeline))
        if not result:
            return {}
        r = result[0]
        r.pop("_id", None)

        # Alertes
        n_alertes = self.db.alertes.count_documents(
            {"timestamp": {"$gte": since}})
        n_danger = self.db.alertes.count_documents(
            {"timestamp": {"$gte": since}, "niveau": "DANGER"})

        return {
            **{k: round(v, 2) if isinstance(v, float) else v
               for k, v in r.items()},
            "alertes_total"  : n_alertes,
            "alertes_danger" : n_danger,
            "periode_heures" : hours,
        }

    def get_time_series(self, metric: str = "debit_veh_h",
                        minutes: int = 60) -> list:
        """
        Retourne une série temporelle d'une métrique.
        Utilisé pour les graphiques temps réel du dashboard.
        """
        since = datetime.utcnow() - timedelta(minutes=minutes)
        if not self.online:
            return []
        cursor = self.db.metriques.find(
            {"timestamp": {"$gte": since}},
            {"timestamp": 1, metric: 1, "_id": 0}
        ).sort("timestamp", ASCENDING)
        return list(cursor)

    # ────────────────────────────────────────────────────────
    # BUFFER & FLUSH
    # ────────────────────────────────────────────────────────

    def _buffer_insert(self, collection: str, doc: dict):
        """Ajoute au buffer — flush si plein ou délai écoulé."""
        self._buffer[collection].append(doc)
        if (len(self._buffer[collection]) >= self._buffer_size or
                time.time() - self._last_flush > self._flush_interval):
            self._flush(collection)

    def _direct_insert(self, collection: str, doc: dict):
        """Insertion immédiate (alertes critiques)."""
        try:
            if self.online:
                self.db[collection].insert_one(doc)
            else:
                self._fallback_write(collection, doc)
        except Exception as e:
            logger.error(f"[DB] Insert error ({collection}): {e}")
            self._fallback_write(collection, doc)

    def _flush(self, collection: str = None):
        """Vide le buffer vers MongoDB."""
        collections = [collection] if collection else list(self._buffer.keys())
        for col in collections:
            if not self._buffer[col]:
                continue
            docs = self._buffer[col].copy()
            self._buffer[col].clear()
            try:
                if self.online:
                    self.db[col].insert_many(docs, ordered=False)
                else:
                    for d in docs:
                        self._fallback_write(col, d)
            except Exception as e:
                logger.error(f"[DB] Flush error ({col}): {e}")
                for d in docs:
                    self._fallback_write(col, d)
        self._last_flush = time.time()

    def flush_all(self):
        """Force le flush de tous les buffers (appel en fin de session)."""
        self._flush()

    # ────────────────────────────────────────────────────────
    # FALLBACK JSON
    # ────────────────────────────────────────────────────────

    def _fallback_write(self, collection: str, doc: dict):
        """Écrit dans un fichier JSON local si MongoDB absent."""
        path = os.path.join(self.fallback_dir, f"{collection}.jsonl")
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(doc, default=str) + "\n")

    def _fallback_read(self, collection: str, n: int) -> list:
        path = os.path.join(self.fallback_dir, f"{collection}.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]

    def _fallback_get_user(self, username: str) -> dict | None:
        users = self._fallback_read("users", 100)
        for u in users:
            if u.get("username") == username.lower():
                return u
        return None

    def close(self):
        """Ferme proprement la connexion."""
        self.flush_all()
        if self.client:
            self.client.close()
        logger.info("[DB] Connexion fermée proprement")
