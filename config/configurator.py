import yaml

from core.authenticator import Authenticator
from core.capturer import Capturer
from core.detector import Detector
from core.embedder import Embedder
from db.database import Database


class Configurator:
    def __init__(self, config):
        self.config = config

        self.detector_conf = {
            "device": config.get("device", "cpu"),
            "min_confidence": config.get("min_confidence", 0.9),
            "min_face_size": config.get("min_face_size", 300),
        }

        self.capturer_conf = {
            "min_look_time": config.get("min_look_time", 1.0),
            "symmetry_threshold": config.get("symmetry_threshold", 10),
            "debounce_time": config.get("debounce_time", 2.0),
        }

        self.embedder_conf = {
            "device": config.get("device", "cpu"),
            "image_size": config.get("image_size", 160),
            "post_process": config.get("post_process", True),
            "margin": config.get("margin", 0),
        }

        self.authenticator_conf = {
            "embedding_dim": config.get("embedding_dim", 512),
            "threshold": config.get("threshold", 0.8),
        }

        self.db_conf = {
            "embedding_dim": config.get("embedding_dim", 512),
            "db_path": config.get("db_path", "db/faces.db"),
        }

    @classmethod
    def parse_conf(cls, conf):
        with open(conf, "r") as f:
            config = yaml.safe_load(f) or {}
        return cls(config)

    def create_detector(self):
        return Detector(**self.detector_conf)

    def create_capturer(self):
        return Capturer(**self.capturer_conf)

    def create_embedder(self):
        return Embedder(**self.embedder_conf)

    def create_authenticator(self):
        return Authenticator(**self.authenticator_conf)

    def create_database(self):
        return Database(**self.db_conf)
