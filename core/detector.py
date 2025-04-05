import cv2
from facenet_pytorch import MTCNN
from PIL import Image


class Detector:
    def __init__(self, device="cpu", min_confidence=0.9, min_face_size=200):
        self.mtcnn = MTCNN(keep_all=False, device=device, min_face_size=min_face_size)
        self.min_confidence = min_confidence

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        boxes, conf, landmarks = self.mtcnn.detect(img, landmarks=True)

        if boxes is not None and len(boxes):
            if conf[0] >= self.min_confidence:
                return {
                    "box": boxes[0].astype(int),
                    "confidence": conf[0],
                    "landmarks": landmarks[0],
                }
        return None
