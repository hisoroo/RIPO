import cv2
from PIL import Image
import os 
import numpy as np 

from config.configurator import Configurator


SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

def _list_images(photos_dir_path: str):
    """Lists supported image files in the given directory."""
    if not os.path.isdir(photos_dir_path):
        return []
    
    images = []
    for f_name in os.listdir(photos_dir_path):
        if os.path.splitext(f_name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(f_name)
    return sorted(images)


def _register_from_camera(user_id: str, configurator: Configurator, detector, database, embedder):
    is_headless = configurator.is_headless()
    if is_headless:
        print("❌ Rejestracja użytkownika z kamery nie jest dostępna w trybie headless.")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Nie można otworzyć kamery.")
        cap.release() 
        return False

    cv2.namedWindow("Rejestracja użytkownika", cv2.WINDOW_NORMAL)
    print("Uruchamianie kamery... Spójrz prosto w obiektyw.")
    registered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Nie udało się przechwycić obrazu z kamery.")
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
            print("Przechwytywanie obrazu...")
            
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = embedder.get_embedding(image_pil)

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


def _register_from_file(user_id: str, image_path: str, configurator: Configurator, detector, database, embedder):
   
    if not image_path or not os.path.exists(image_path):
        print(f"❌ Ścieżka do pliku jest nieprawidłowa lub plik nie istnieje: {image_path}")
        return False

    try:
        print(f"Przetwarzanie obrazu z pliku: {image_path}...")
        pil_image = Image.open(image_path).convert("RGB")

        
        frame_for_detector = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        detection = detector.detect(frame_for_detector)

        if not detection:
            print("❌ Nie wykryto twarzy na obrazie.")
            return False

        
        embedding = embedder.get_embedding(pil_image)

        if embedding is not None:
            database.save_embedding(user_id, embedding)
            print(f"✅ Zarejestrowano nowego użytkownika: {user_id} na podstawie pliku.")
            return True
        else:
            print("❌ Nie udało się wygenerować wektora cech z obrazu.")
            return False
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Wystąpił błąd podczas przetwarzania obrazu z pliku: {e}")
        return False


def register_user(user_id: str, source_type: str, image_path: str = None):
    configurator = Configurator.parse_conf("config/config.yaml")
    detector = configurator.create_detector()
    database = configurator.create_database()
    embedder = configurator.create_embedder()

    if source_type == "camera":
        return _register_from_camera(user_id, configurator, detector, database, embedder)
    elif source_type == "file":
        if not image_path:
            print("❌ Nie podano ścieżki do pliku obrazu.")
            return False
        return _register_from_file(user_id, image_path, configurator, detector, database, embedder)
    else:
        print(f"❌ Nieznany typ źródła: {source_type}")
        return False


def main():
    print("Rejestracja użytkownika")
    user_id = input("Podaj identyfikator użytkownika: ").strip()
    if not user_id:
        print("❌ Nieprawidłowy identyfikator użytkownika.")
        return

    while True:
        source_choice = input("Wybierz metodę rejestracji (kamera/plik): ").strip().lower()
        if source_choice in ["kamera", "plik"]:
            break
        print("❌ Nieprawidłowy wybór. Wpisz 'kamera' lub 'plik'.")

    success = False
    if source_choice == "kamera":
        success = register_user(user_id, source_type="camera")
    elif source_choice == "plik":
        photos_dir = "photos"

        
        print(f"Jeśli używasz Dockera, upewnij się, że zdjecia sa w katalogu {os.path.abspath(photos_dir)}")

        if not os.path.isdir(photos_dir):
            print(f"❌ Katalog '{photos_dir}' nie istnieje. Utwórz go i dodaj zdjęcia, lub sprawdź konfigurację woluminu Docker.")
            return

        available_images = _list_images(photos_dir)

        if not available_images:
            print(f"❌ Nie znaleziono żadnych obsługiwanych plików obrazów w katalogu '{photos_dir}'.")
            print(f"   Obsługiwane formaty: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
            return

        print("\nDostępne zdjęcia w katalogu '{}':".format(photos_dir))
        for i, img_name in enumerate(available_images):
            print(f"  {i + 1}. {img_name}")

        while True:
            try:
                choice = input(f"Wybierz numer zdjęcia (1-{len(available_images)}) lub 'q' aby anulować: ").strip()
                if choice.lower() == 'q':
                    print("❌ Rejestracja z pliku anulowana.")
                    return
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_images):
                    selected_image_name = available_images[choice_idx]
                    image_path = os.path.join(photos_dir, selected_image_name)
                    print(f"Wybrano: {image_path}")
                    success = register_user(user_id, source_type="file", image_path=image_path)
                    break
                else:
                    print(f"❌ Nieprawidłowy wybór. Podaj numer od 1 do {len(available_images)}.")
            except ValueError:
                print("❌ Nieprawidłowy format. Podaj numer.")
            except Exception as e:
                print(f"❌ Wystąpił nieoczekiwany błąd: {e}")
                return


    if success:
        print("✅ Rejestracja zakończona sukcesem.")
    else:
        print("❌ Rejestracja nie powiodła się.")


if __name__ == "__main__":
    main()
