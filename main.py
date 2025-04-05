import time

import cv2
from PIL import Image

from config.configurator import Configurator
from utils.verbose import draw_overlay, print_logs

configurator = Configurator.parse_conf("config/config.yaml")

is_verbose = configurator.verbose_output()

detector = configurator.create_detector()
capturer = configurator.create_capturer()
embedder = configurator.create_embedder()
authenticator = configurator.create_authenticator()
database = configurator.create_database()

embeddings, user_ids = database.load_all_embeddings()
authenticator.load_embeddings(embeddings, user_ids)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Autentykacja", cv2.WINDOW_NORMAL)

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
            user_id, distance = authenticator.authenticate(embedding)

            if user_id:
                last_result_text = f"Rozpoznano: {user_id}"
                print(f"✅ Rozpoznano: {user_id}")
                if is_verbose:
                    print_logs(user_id, distance, detection)
            else:
                last_result_text = "Nieznana osoba"
                print("❌ Nieznana osoba")
                if is_verbose:
                    print_logs(user_id=None, distance=distance, detection=detection)

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
        if is_verbose and detection:
            draw_overlay(frame, detection, distance)

    cv2.imshow("Autentykacja", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
