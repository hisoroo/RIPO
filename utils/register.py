import cv2
from PIL import Image

from config.configurator import Configurator


def register_user_from_camera(user_id: str):
    configurator = Configurator.parse_conf("config/config.yaml")
    detector = configurator.create_detector()
    database = configurator.create_database()
    embedder = configurator.create_embedder()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Rejestracja użytkownika", cv2.WINDOW_NORMAL)
    print("📷 Uruchamianie kamery... Spójrz prosto w obiektyw.")
    registered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detector.detect(frame)
        if detection:
            box = detection["box"]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Wykryto twarz - nacisnij 'c', aby zarejestrowac",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Rejestracja użytkownika", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and detection:
            print("📸 Przechwytywanie obrazu...")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = embedder.get_embedding(image)

            if embedding is not None:
                database.save_embedding(user_id, embedding)
                print(f"✅ Zarejestrowano nowego użytkownika: {user_id}")
                registered = True
            else:
                print("❌ Nie udało się zarejestrować.")
            break
        elif key == ord("q"):
            print("❌ Rejestracja anulowana.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return registered


def main():
    print("👤 Rejestracja użytkownika")
    user_id = input("Podaj identyfikator użytkownika: ").strip()
    if user_id:
        success = register_user_from_camera(user_id)
        if success:
            print("✅ Rejestracja zakończona sukcesem.")
        else:
            print("❌ Rejestracja nie powiodła się.")
    else:
        print("❌ Nieprawidłowy identyfikator użytkownika.")


if __name__ == "__main__":
    main()
