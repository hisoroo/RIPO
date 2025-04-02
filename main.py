import time

import cv2
import torch
from PIL import Image

from core.authenticator import Authenticator
from core.capturer import Capturer
from core.detector import Detector
from core.embedder import Embedder
from db.database import Database

device = "cuda" if torch.cuda.is_available() else "cpu"

detector = Detector(device=device)
capturer = Capturer()
embedder = Embedder(device=device)
database = Database()
authenticator = Authenticator()

embeddings, user_ids = database.load_all_embeddings()
authenticator.load_embeddings(embeddings, user_ids)

cap = cv2.VideoCapture(0)

last_result_time = 0
last_result_text = ""
result_display_duration = 3

print("🔄 System uruchomiony. Wciśnij 'q' aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detection = detector.detect(frame)
    if detection:
        is_facing = capturer.is_facing_forward(detection["landmarks"])
        if capturer.check_capture(is_facing):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = embedder.get_embedding(pil_image)
            user_id = authenticator.authenticate(embedding)

            if user_id:
                last_result_text = f"Rozpoznano: {user_id}"
                print(f"✅ {last_result_text}")
            else:
                last_result_text = "Nieznana osoba"
                print("❌ Nieznana osoba")

            last_result_time = time.time()

    if time.time() - last_result_time < result_display_duration:
        cv2.putText(
            frame,
            last_result_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if "Rozpoznano" in last_result_text else (0, 0, 255),
            2,
        )

    cv2.imshow("Autentykacja", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
