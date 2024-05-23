import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import warnings 

warnings.filterwarnings('ignore')

# Cargar el modelo
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'Ñ', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}

# Función para procesar el video y detectar letras
def process_video(word):
    cap = cv2.VideoCapture(0)
    detected_letters = []
    letter_duration = 0
    last_detected_letter = None
    current_letter_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W)
                y1 = int(min(y_) * H)
                x2 = int(max(x_) * W)
                y2 = int(max(y_) * H)

                prediction = model.predict([np.asarray(data_aux[:42] * 2)])
                predicted_character = labels_dict[int(prediction[0])]

                # Mostrar el rectángulo y la letra detectada en el cuadro de video
                color = (0, 255, 0) if predicted_character == word[current_letter_idx] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                if predicted_character == last_detected_letter:
                    letter_duration += 1
                else:
                    letter_duration = 0

                if letter_duration > 30:  # Aproximadamente 1 segundo
                    if predicted_character == word[current_letter_idx]:
                        detected_letters.append(predicted_character)
                        current_letter_idx += 1
                        if current_letter_idx >= len(word):
                            result_label.config(text="Palabra correcta")
                            detected_letters = []
                            current_letter_idx = 0
                            break
                    else:
                        detected_letters = []
                        current_letter_idx = 0
                    letter_duration = 0

                last_detected_letter = predicted_character

        # Mostrar el video
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Actualizar la etiqueta de resultados en la interfaz gráfica
        matched_letters = "".join(detected_letters)
        result_label.config(text=f"Letras detectadas: {matched_letters}")

    cap.release()
    cv2.destroyAllWindows()

# Crear la ventana principal
root = tk.Tk()
root.title("Detección de Letras por Figuras")

# Crear el marco principal
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Agregar un campo de entrada para la palabra
ttk.Label(frame, text="Ingrese la palabra:").grid(row=0, column=0, sticky=tk.W)
word_entry = ttk.Entry(frame, width=20)
word_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

# Crear un campo para mostrar el resultado
result_label = ttk.Label(frame, text="", font=("Helvetica", 16))
result_label.grid(row=1, column=0, columnspan=2, pady=10)

# Función para iniciar el procesamiento de video en un hilo separado
def start_video_processing():
    word = word_entry.get().upper()  # Convertir la palabra a mayúsculas
    threading.Thread(target=process_video, args=(word,)).start()

# Agregar un botón para iniciar la detección
start_button = ttk.Button(frame, text="Iniciar", command=start_video_processing)
start_button.grid(row=2, column=0, columnspan=2)

# Ejecutar el bucle principal de la interfaz
root.mainloop()
