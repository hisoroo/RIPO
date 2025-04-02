import time


class Capturer:
    def __init__(self, min_look_time=1.0, symmetry_threshold=10, debounce_time=2.0):
        self.min_look_time = min_look_time
        self.symmetry_threshold = symmetry_threshold
        self.debounce_time = debounce_time

        self.looking_since = None
        self.last_capture_time = 0

    def is_facing_forward(self, landmarks):
        left_eye, right_eye, nose, mouth_left, mouth_right = landmarks

        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
        nose_x = nose[0]

        symmetrical_eyes = abs(nose_x - eye_center_x) < self.symmetry_threshold
        symmetrical_mouth = abs(nose_x - mouth_center_x) < self.symmetry_threshold

        return symmetrical_eyes and symmetrical_mouth

    def check_capture(self, is_facing):
        now = time.time()

        if is_facing:
            if self.looking_since is None:
                self.looking_since = now
            elif (
                now - self.looking_since >= self.min_look_time
                and now - self.last_capture_time >= self.debounce_time
            ):
                self.last_capture_time = now
                return True
        else:
            self.looking_since = None

        return False

