import time
import math

import cv2
from PIL import Image

from config.configurator import Configurator
from utils.verbose import draw_overlay, print_logs
from utils.video import setup_capture, setup_recorder

configurator = Configurator.parse_conf("config/config.yaml")
is_verbose = configurator.verbose_output()
is_headless = configurator.is_headless()

detector = configurator.create_detector()
capturer = configurator.create_capturer()
embedder = configurator.create_embedder()
authenticator = configurator.create_authenticator()
database = configurator.create_database()

embeddings, user_ids = database.load_all_embeddings()
authenticator.load_embeddings(embeddings, user_ids)

video_conf = configurator.setup_video()
mode = video_conf.get("mode", "live")
file_path = video_conf.get("file_path")
record_video = video_conf.get("record_video", False)
output_path = video_conf["dynamic_path"]

cap = setup_capture(mode, file_path)

if mode == "image":
    frame = cap
    detection = detector.detect(frame)
    distance_metric = None
    user_id_result = None

    if detection:
        is_facing = capturer.is_facing_forward(detection["landmarks"])
        if capturer.check_capture(is_facing):
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = embedder.get_embedding(pil_image)
            
            auth_result = authenticator.authenticate(embedding)
            user_id = None
            auth_distance = None

            if auth_result is not None:
                user_id, auth_distance = auth_result
            
            distance_metric = auth_distance
            user_id_result = user_id

            if user_id:
                result_text = f"✅ Rozpoznano: {user_id}"
                print(result_text)
            else:
                result_text = "❌ Nieznana osoba"
                print(result_text)

            if not is_headless:
                cv2.putText(
                    frame,
                    result_text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if "Rozpoznano" in result_text else (0, 0, 255),
                    2,
                )

            if is_verbose:
                print_logs(user_id_result, distance_metric, detection)
                if not is_headless:
                    draw_overlay(frame, detection, distance_metric)
        else:
            print("Twarz nie jest skierowana do przodu.")
    else:
        print("❗ Nie wykryto twarzy na zdjęciu.")

    if not is_headless:
        cv2.imshow("Autentykacja", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    print(f"Warning: Camera/Video FPS reported as {fps}. Defaulting to 30 FPS.")
    fps = 30.0
target_frame_duration = 1.0 / fps

out = setup_recorder(cap, output_path) if record_video and mode == "live" else None

if record_video:
    print(f"Nagrywanie aktywne — zapis do: {output_path} (FPS: {fps:.2f})")
else:
    print("Nagrywanie wyłączone")

if not is_headless:
    cv2.namedWindow("Autentykacja", cv2.WINDOW_NORMAL)

last_result_time = 0
last_result_text = ""
result_display_duration = 3

last_detection = None
last_distance = None
last_detection_time = 0

print("System uruchomiony. Wciśnij 'q' aby zakończyć.")

while True:
    loop_start_time = time.time()

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

            auth_result = authenticator.authenticate(embedding)
            user_id = None
            auth_distance = None

            if auth_result is not None:
                user_id, auth_distance = auth_result
            
            distance_metric = auth_distance

            if user_id:
                current_result_text = f"Rozpoznano: {user_id}"
                if last_result_text != current_result_text or (time.time() - last_result_time > result_display_duration):
                    print(f"✅ {current_result_text}")
                last_result_text = current_result_text
                if is_verbose:
                    print_logs(user_id, distance_metric, detection)
            else:
                current_result_text = "Nieznana osoba"
                if last_result_text != current_result_text or (time.time() - last_result_time > result_display_duration):
                    print(f"❌ {current_result_text}")
                last_result_text = current_result_text
                if is_verbose:
                    print_logs(None, distance_metric, detection)
            
            last_result_time = time.time()
            last_detection_time = time.time()
            last_detection = detection
            last_distance = distance_metric
    else:
        if time.time() - last_detection_time < result_display_duration:
            detection = last_detection
            distance_metric = last_distance
        else:
            detection = None
            distance_metric = None

    current_display_time = time.time()
    if not is_headless and current_display_time - last_result_time < result_display_duration:
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
            draw_overlay(frame, detection, last_distance)

    if not is_headless:
        cv2.imshow("Autentykacja", frame)

    processing_time = time.time() - loop_start_time
    
    if out:
        num_writes = int(max(1, math.ceil(processing_time / target_frame_duration)))
        for _ in range(num_writes):
            out.write(frame)
    
    wait_duration_ms = int(max(1, (target_frame_duration - processing_time) * 1000))

    if cv2.waitKey(wait_duration_ms) & 0xFF == ord("q"):
        print("Zamykanie programu...")
        break

if out:
    out.release()
cap.release()
if not is_headless:
    cv2.destroyAllWindows()
