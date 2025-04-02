import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


class Embedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.mtcnn = MTCNN(image_size=160, device=device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def get_embedding(self, frame):
        face = self.mtcnn(frame)
        if face is None:
            return None
        with torch.no_grad():
            face = face.unsqueeze(0).to(self.device)
            embedding = self.model(face)
        return embedding.squeeze(0)
