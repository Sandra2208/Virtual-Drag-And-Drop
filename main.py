import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Color for the rectangle
colorR = (255, 0, 255)

# Class to handle dragging of rectangles
class DragRect:
    def __init__(self, pos_center, size=[200, 200]):
        self.pos_center = pos_center
        self.size = size
        self.dragging = False
        self.offset_x, self.offset_y = 0, 0

    def update(self, cursor):
        cx, cy = self.pos_center
        w, h = self.size

        # Check if the cursor is within the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            if not self.dragging:
                # Update offset when starting to drag
                self.offset_x, self.offset_y = cursor[0] - cx, cursor[1] - cy
                self.dragging = True
        else:
            self.dragging = False

        if self.dragging:
            # Update position when dragging
            self.pos_center = cursor[0] - self.offset_x, cursor[1] - self.offset_y

# List of rectangles for dragging
rect_list = []
for x in range(5):
    rect_list.append(DragRect((x * 250 + 150, 150)))  # Perbaikan di sini

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar kerangka tangan
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = img.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Update rectangles when the index finger is close
            for rect in rect_list:
                rect.update((cx, cy))

    # Draw semi-transparent rectangles
    overlay = img.copy()
    for rect in rect_list:
        cx, cy = rect.pos_center
        w, h = rect.size
        overlay = cv2.rectangle(overlay, (int(cx - w // 2), int(cy - h // 2)),
                                (int(cx + w // 2), int(cy + h // 2)), colorR, cv2.FILLED)

    # Apply semi-transparent overlay
    alpha = 0.6  # Transparency level (0: fully transparent, 1: fully opaque)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
