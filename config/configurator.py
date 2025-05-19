import os
from datetime import datetime

import yaml

from core.authenticator import Authenticator
from core.capturer import Capturer
from core.detector import Detector
from core.embedder import Embedder
from db.database import Database


class Configurator:
    def __init__(self, config):
        self.config = config

        device = config.get("device", "cpu")
        detector = config.get("detector", {})
        capturer = config.get("capturer", {})
        embedder = config.get("embedder", {})
        authenticator = config.get("authenticator", {})
        database = config.get("database", {})
        misc = config.get("misc", {})

        self.detector_conf = {
            "device": device,
            "min_confidence": detector.get("min_confidence", 0.9),
            "min_face_size": detector.get("min_face_size", 300),
        }

        self.capturer_conf = {
            "min_look_time": capturer.get("min_look_time", 1.0),
            "symmetry_threshold": capturer.get("symmetry_threshold", 10),
            "debounce_time": capturer.get("debounce_time", 2.0),
            "check_is_facing": capturer.get("check_is_facing", True),
        }

        self.embedder_conf = {
            "device": device,
            "image_size": embedder.get("image_size", 160),
            "post_process": embedder.get("post_process", True),
            "margin": embedder.get("margin", 0),
        }

        self.authenticator_conf = {
            "embedding_dim": authenticator.get("embedding_dim", 512),
            "threshold": authenticator.get("threshold", 0.8),
        }

        self.db_conf = {
            "embedding_dim": database.get("embedding_dim", 512),
            "db_path": database.get("db_path", "db/faces.db"),
        }

        self.video = misc.get("video", {})
        self.verbose = misc.get("verbose", False)
        self.headless = misc.get("headless", False)

    @classmethod
    def parse_conf(cls, conf_path):
        with open(conf_path, "r") as f:
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

    def verbose_output(self):
        return self.verbose

    def is_headless(self):
        return self.headless

    def setup_video(self):
        video = self.video.copy()

        current_date = datetime.now().strftime("%d-%m-%Y")
        vid_dir = os.path.join("videos", current_date)
        os.makedirs(vid_dir, exist_ok=True)

        default_filename = datetime.now().strftime("video-%H-%M-%S.avi")
        user_filename = video.get("output_path", "").strip()

        final_path = os.path.join(vid_dir, user_filename or default_filename)
        video["dynamic_path"] = final_path
        return video
