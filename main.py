import time

import cv2
from PIL import Image

from config.configurator import Configurator
from utils.verbose import draw_overlay, print_logs
from utils.video import setup_capture, setup_recorder

configurator = Configurator.parse_conf("config/config.yaml")

is_verbose = configurator.verbose_output()

detector = configurator.create_detector()
capturer = configurator.create_capturer()
embedder = configurator.create_embedder()
authenticator = configurator.create_authenticator()
database = configurator.create_database()

embeddings, user_ids = database.load_all_embeddings()
authenticator.load_embeddings(embeddings, user_ids)

video_conf = configurator.setup_video()
mode = video_conf.get("mode", "live")
record_video = video_conf.get("record_video", False)
video_path = video_conf.get("video_path")
output_path = video_conf["dynamic_path"]

cap = setup_capture(mode, video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    print(f"Warning: Camera/Video FPS reported as {fps}. Defaulting to 30 FPS.")
    fps = 30.0
target_frame_duration = 1.0 / fps

out = setup_recorder(cap, output_path, fps) if record_video and mode == "live" else None

if record_video:
    print(f"Nagrywanie aktywne — zapis do: {output_path} (FPS: {fps:.2f})")
else:
    print("Nagrywanie wyłączone")

cv2.namedWindow("Autentykacja", cv2.WINDOW_NORMAL)

last_result_time = 0
last_result_text = ""
result_display_duration = 3  # seconds

print("System uruchomiony. Wciśnij 'q' aby zakończyć.")

while True:
    loop_start_time = time.time()

    distance_metric = None

    ret, frame = cap.read()
    if not ret:
        print("Koniec pliku lub problem z kamerą.")
        break

    detection = detector.detect(frame)
    if detection:
        is_facing = capturer.is_facing_forward(detection["landmarks"])
        if capturer.check_capture(is_facing):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = embedder.get_embedding(pil_image)
            user_id, auth_distance = authenticator.authenticate(embedding)
            distance_metric = auth_distance

            if user_id:
                last_result_text = f"Rozpoznano: {user_id}"
                print(f"✅ Rozpoznano: {user_id}")
                if is_verbose:
                    print_logs(user_id, distance_metric, detection)
            else:
                last_result_text = "Nieznana osoba"
                print("❌ Nieznana osoba")
                if is_verbose:
                    print_logs(user_id=None, distance=distance_metric, detection=detection)

            last_result_time = time.time()

    current_display_time = time.time()
    if current_display_time - last_result_time < result_display_duration:
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
            draw_overlay(frame, detection, distance_metric)

    cv2.imshow("Autentykacja", frame)

    if out:
        out.write(frame)

    processing_time = time.time() - loop_start_time

    wait_duration_ms = int(max(1, (target_frame_duration - processing_time) * 1000))

    if cv2.waitKey(wait_duration_ms) & 0xFF == ord("q"):
        print("Zamykanie programu...")
        break
if out:
    out.release()
cap.release()
cv2.destroyAllWindows()
