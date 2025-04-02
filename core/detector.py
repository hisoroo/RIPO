import cv2
from facenet_pytorch import MTCNN
from PIL import Image


class Detector:
    def __init__(self, device="cpu", min_confidence=0.9, min_face_height=100):
        self.mtcnn = MTCNN(keep_all=False, device=device)
        self.min_confidence = min_confidence
        self.min_face_height = min_face_height

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        boxes, conf, landmarks = self.mtcnn.detect(img, landmarks=True)

        if boxes.size > 0:
            box = boxes[0].astype(int)
            confidence = conf[0]
            height = box[3] - box[1]

            if confidence >= self.min_confidence and height >= self.min_face_height:
                return {
                    "box": box,
                    "confidence": confidence,
                    "landmarks": landmarks[0],
                }
        return None
