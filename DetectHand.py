from threading import Thread
import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import imutils
from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw

# Configuration initiale pour réduire la charge de traitement
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pré-définir les index des landmarks pour éviter les appels répétés
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_FINGER_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
RING_FINGER_TIP = mp_hands.HandLandmark.RING_FINGER_TIP
PINKY_TIP = mp_hands.HandLandmark.PINKY_TIP

previous_y_position = None
tab_gesture_start_time = None
is_alt_tab_active = False
initial_center_x = None
icon_state = 'white'
gesture_cooldown = 0.5
last_gesture_time = {}

def create_image(color='white'):
    width, height = 64, 64
    image = Image.new('RGB', (width, height), 'white')  # Fond blanc pour visibilité
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, width-1, height-1), fill=color)
    return image

def update_icon():
    global icon_state
    icon = Icon("Hand tracking")
    icon.icon = create_image(icon_state)  # Utilise l'état actuel pour définir la couleur de l'icône
    icon.menu = Menu(MenuItem('Quit', lambda icon: icon.stop()))
    icon.run()

def set_icon_state(new_state):
    global icon
    icon.icon = create_image(new_state)

def distance_between_points(landmark1, landmark2):
    # Utilisez numpy directement pour calculer la distance, ce qui est plus rapide
    return np.linalg.norm(np.array([landmark1.x, landmark1.y]) - np.array([landmark2.x, landmark2.y]))

def calculate_hand_center(hand_landmarks, image_width, image_height):
    x_coords = np.array([landmark.x for landmark in hand_landmarks.landmark])
    y_coords = np.array([landmark.y for landmark in hand_landmarks.landmark])
    center_x, center_y = np.mean(x_coords) * image_width, np.mean(y_coords) * image_height
    return int(center_x), int(center_y)

def detect_gesture(hand_landmarks, center_y, center_x, frame_time):
    global previous_y_position, tab_gesture_start_time, is_alt_tab_active, initial_center_x, last_gesture_time

    # Simplification du calcul des distances
    thumb_tip = np.array([hand_landmarks.landmark[THUMB_TIP].x, hand_landmarks.landmark[THUMB_TIP].y])
    index_tip = np.array([hand_landmarks.landmark[INDEX_FINGER_TIP].x, hand_landmarks.landmark[INDEX_FINGER_TIP].y])
    middle_tip = np.array([hand_landmarks.landmark[MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[MIDDLE_FINGER_TIP].y])
    ring_tip = np.array([hand_landmarks.landmark[RING_FINGER_TIP].x, hand_landmarks.landmark[RING_FINGER_TIP].y])
    pinky_tip = np.array([hand_landmarks.landmark[PINKY_TIP].x, hand_landmarks.landmark[PINKY_TIP].y])

    # Utilisez le poignet comme point de référence pour la hauteur des bouts des doigts
    wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])

    # Compter combien de doigts sont fermés (bouts des doigts inférieurs au poignet sur l'axe Y)
    fingers_closed = sum([tip[1] > wrist[1] for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]])

    gesture = ""
    current_gesture = "none"
    if any(gesture in last_gesture_time and frame_time - last_gesture_time[gesture] < gesture_cooldown for gesture in ["tab", "scroll", "alt_tab"]):
        return "Cooldown en cours"
    elif fingers_closed >= 4:  # Seuil pour considérer la main comme fermée
        gesture = "Main fermée détectée"
    elif np.linalg.norm(thumb_tip - index_tip) < 0.05:  # Seuil de détection du geste "touche"
        gesture = "Geste touche détecté"
        set_icon_state('green')
        if previous_y_position is not None:
            y_movement = center_y - previous_y_position
            pyautogui.scroll(int(y_movement * 10))  # Inversion et ajustement de la sensibilité
        previous_y_position = center_y
    elif np.linalg.norm(thumb_tip - pinky_tip) < 0.03:  # Seuil de détection du geste "tab"
        current_time = time.time()
        set_icon_state('green')
        if tab_gesture_start_time is None:
            tab_gesture_start_time = current_time
            initial_center_x = center_x  # Mémoriser la position initiale pour le mouvement latéral
        elif current_time - tab_gesture_start_time > 1 and not is_alt_tab_active:
            # Déclencher Alt+Tab et garder Alt pressé
            pyautogui.keyDown('alt')
            pyautogui.press('tab')
            is_alt_tab_active = True
            gesture = "Alt+Tab activé"
        elif is_alt_tab_active:
            # Détecter le mouvement latéral pour changer de fenêtre
            movement_threshold = 50  # Définir une valeur de seuil appropriée pour la détection de mouvement
            if abs(center_x - initial_center_x) > movement_threshold:
                direction = "right" if center_x < initial_center_x else "left"
                pyautogui.press(['right', 'left'][direction == "left"])  # Simuler appui sur flèche droite ou gauche
                initial_center_x = center_x  # Réinitialiser la position pour le prochain mouvement
                set_icon_state('green')
    else:
        if is_alt_tab_active:
            # Relâcher Alt si les doigts se séparent
            pyautogui.keyUp('alt')
            is_alt_tab_active = False
        tab_gesture_start_time = None  # Réinitialiser le suivi du temps pour le geste Tab
        gesture = "Autre geste"
        previous_y_position = None
        previous_y_position = None if fingers_closed < 4 else previous_y_position

    return gesture


def capture_video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = time.time()

        # Détectez et dessinez le visage avant la détection des mains
        frame = detect_hand(frame, frame_time)

        cv2.imshow('Hand Detection', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_hand(frame, frame_time):
    # Réduire la taille de l'image pour une meilleure performance
    frame = imutils.resize(frame, width=600)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_height, image_width, _ = frame.shape

    if results.multi_hand_landmarks:
        set_icon_state('blue')
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            center_x, center_y = calculate_hand_center(hand_landmarks, image_width, image_height)
            gesture = detect_gesture(hand_landmarks, center_y, center_x, frame_time)
    else:
        set_icon_state('white')

    return frame

def main():
    global icon
    icon = Icon("Hand tracking")
    icon.icon = create_image('white')
    icon.menu = Menu(MenuItem('Quit', lambda icon: icon.stop()))

    # Démarrez l'icône dans son propre thread
    icon_thread = Thread(target=lambda: icon.run())
    icon_thread.start()

    # Maintenant, lancez la capture vidéo dans le thread principal ou un autre thread si nécessaire
    capture_video()

    # Assurez-vous que l'icône s'arrête proprement à la fin
    icon.stop()

if __name__ == "__main__":
    main()