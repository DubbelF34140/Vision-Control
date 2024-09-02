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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=6,min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
gesture_cooldown = 0
last_gesture_time = {}
index_positions = []

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

def calculate_rotation_direction(positions):
    # Calcul sommaire pour déterminer le sens de rotation
    sum = 0
    for i in range(len(positions)):
        x1, y1 = positions[i - 1]
        x2, y2 = positions[i]
        sum += (x2 - x1) * (y2 + y1)
    return "clockwise" if sum > 0 else "counterclockwise"


def distance_between_points(landmark1, landmark2):
    # Utilisez numpy directement pour calculer la distance, ce qui est plus rapide
    return np.linalg.norm(np.array([landmark1.x, landmark1.y]) - np.array([landmark2.x, landmark2.y]))

def calculate_hand_center(hand_landmarks, image_width, image_height):
    x_coords = np.array([landmark.x for landmark in hand_landmarks.landmark])
    y_coords = np.array([landmark.y for landmark in hand_landmarks.landmark])
    center_x, center_y = np.mean(x_coords) * image_width, np.mean(y_coords) * image_height
    return int(center_x), int(center_y)

def perform_alt_tab(direction):
    """
    Réalise l'action Alt+Tab dans un thread séparé.
    """
    pyautogui.keyDown('alt')
    pyautogui.press('tab')
    time.sleep(0.1)  # Donne du temps pour que l'action soit exécutée
    if direction:
        pyautogui.press(['right', 'left'][direction == "left"])
    pyautogui.keyUp('alt')


def execute_action(action, args=()):
    """
    Exécute une action donnée dans un thread séparé pour ne pas bloquer le traitement principal.
    Les actions peuvent inclure des interactions avec pyautogui ou des mises à jour d'interface utilisateur.
    """
    if action == "scroll":
        set_icon_state('green')
        Thread(target=lambda: pyautogui.scroll(int(args[0] * 10))).start()
    elif action == "alt_tab_start":
        set_icon_state('green')
        Thread(target=lambda: pyautogui.keyDown('alt')).start()
        Thread(target=lambda: pyautogui.press('tab')).start()
    elif action == "alt_tab_switch":
        set_icon_state('green')
        direction = "right" if args[0] == "right" else "left"
        Thread(target=lambda: pyautogui.press(direction)).start()
    elif action == "alt_tab_end":
        set_icon_state('green')
        Thread(target=lambda: pyautogui.keyUp('alt')).start()
    elif action == "update_icon":
        set_icon_state('green')
        Thread(target=lambda: set_icon_state(args[0])).start()
    elif action == "adjust_volume":
        if args[0] == "clockwise":
            print("Augmenter le volume")
            # pyautogui.press('volumeup') ou une autre méthode selon votre setup
        elif args[0] == "counterclockwise":
            print("Diminuer le volume")
            # pyautogui.press('volumedown') ou une autre méthode selon votre setup

def detect_gesture(hand_landmarks, center_y, center_x, frame_time):
    global previous_y_position, tab_gesture_start_time, is_alt_tab_active, initial_center_x, last_gesture_time

    # Récupération des positions des bouts des doigts
    thumb_tip, index_tip, pinky_tip, wrist = [
        np.array([hand_landmarks.landmark[pos].x, hand_landmarks.landmark[pos].y])
        for pos in [THUMB_TIP, INDEX_FINGER_TIP, PINKY_TIP, mp_hands.HandLandmark.WRIST]
    ]

    # Récupération des points des doigts non utilisés pour les gestes (majeur, annulaire)
    middle_tip = np.array([hand_landmarks.landmark[MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[MIDDLE_FINGER_TIP].y])
    ring_tip = np.array([hand_landmarks.landmark[RING_FINGER_TIP].x, hand_landmarks.landmark[RING_FINGER_TIP].y])
    index_tip = np.array([hand_landmarks.landmark[INDEX_FINGER_TIP].x, hand_landmarks.landmark[INDEX_FINGER_TIP].y])
    index_positions.append((index_tip[0], index_tip[1]))

    fingers_closed = sum(tip[1] > wrist[1] for tip in [thumb_tip, index_tip, pinky_tip])

    # Calcul des distances entre les doigts non utilisés pour vérifier la neutralité de la pose
    middle_ring_distance = np.linalg.norm(middle_tip - ring_tip)
    index_middle_distance = np.linalg.norm(index_tip - middle_tip)
    ring_pinky_distance = np.linalg.norm(ring_tip - pinky_tip)

    # Seuil de distance pour considérer les doigts comme "trop collés"
    distance_threshold = 0.04  # À ajuster selon les besoins

    if len(index_positions) > 30:
        index_positions.pop(0)

    if any(frame_time - last_gesture_time.get(gesture, 0) < gesture_cooldown for gesture in ["tab", "scroll", "alt_tab"]):
        return "Cooldown en cours"
    elif all(distance < distance_threshold for distance in [middle_ring_distance, index_middle_distance, ring_pinky_distance]):
        gesture = "Neutre"  # La main est dans une position neutre
    elif fingers_closed >= 4:
        execute_action("update_icon", ('green',))
        gesture = "Main fermée détectée"
    elif np.linalg.norm(thumb_tip - index_tip) < 0.03:
        if previous_y_position is not None:
            y_movement = center_y - previous_y_position
            execute_action("scroll", (y_movement,))
        previous_y_position = center_y
        gesture = "Geste touche détecté"
    elif np.linalg.norm(thumb_tip - pinky_tip) < 0.03:
        current_time = time.time()
        if tab_gesture_start_time is None:
            tab_gesture_start_time = current_time
            initial_center_x = center_x
            execute_action("alt_tab_start")
            is_alt_tab_active = True
        elif current_time - tab_gesture_start_time > 1 and is_alt_tab_active:
            if abs(center_x - initial_center_x) > 50:
                direction = "right" if center_x < initial_center_x else "left"
                execute_action("alt_tab_switch", (direction,))
                initial_center_x = center_x
        gesture = "Alt+Tab activé"
    else:
        if is_alt_tab_active:
            execute_action("alt_tab_end")
            is_alt_tab_active = False
        tab_gesture_start_time = None
        gesture = "Autre geste"
        previous_y_position = None if fingers_closed < 4 else previous_y_position

    return gesture

def capture_video():
    cap = cv2.VideoCapture(1)

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


main_hand_id = None

def detect_hand(frame, frame_time):
    global main_hand_id  # Utilisez la variable globale pour maintenir l'ID de la main principale

    frame = imutils.resize(frame, width=600)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_height, image_width, _ = frame.shape

    if results.multi_hand_landmarks:
        set_icon_state('blue')

        # Vérifiez si nous n'avons pas encore identifié la main principale
        if main_hand_id is None and results.multi_handedness:
            # Définissez l'ID de la première main suivie comme la main principale
            main_hand_id = results.multi_handedness[0].classification[0].index

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Obtention de l'ID de suivi pour chaque main
            hand_id = results.multi_handedness[hand_index].classification[0].index

            center_x, center_y = calculate_hand_center(hand_landmarks, image_width, image_height)

            # Choisissez une couleur pour la main principale et une autre pour les mains secondaires
            if hand_id == main_hand_id:
                # Main principale en vert
                color = (0, 255, 0)
                gesture = detect_gesture(hand_landmarks, center_y, center_x, frame_time)
            else:
                # Mains secondaires en bleu
                color = (0, 0, 255)
                gesture = ""  # Vous pouvez choisir de ne pas détecter des gestes pour les mains secondaires

            # Dessiner les landmarks et les connections avec la couleur choisie
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))

            # Dessiner un rectangle et le geste détecté autour de la main principale
            bounding_box_size = int(max(image_width, image_height) * 0.15)
            cv2.rectangle(frame,
                          (center_x - bounding_box_size, center_y - bounding_box_size),
                          (center_x + bounding_box_size, center_y + bounding_box_size),
                          color, 2)
            if gesture:
                cv2.putText(frame, gesture,
                            (center_x - bounding_box_size, center_y - bounding_box_size - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        set_icon_state('white')

    return frame


def main():
    global icon
    icon = Icon("Hand tracking")
    icon.icon = create_image('white')
    icon.menu = Menu(MenuItem('Quit', lambda icon: icon.stop()))
    icon_thread = Thread(target=lambda: icon.run())
    icon_thread.start()
    capture_video()
    icon.stop()

if __name__ == "__main__":
    main()