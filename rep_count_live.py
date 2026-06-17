import tensorflow as tf
import numpy as np
import cv2
import math
import time

target_width = 256
target_height = 256
TRUST_THRESHOLD = 0.40
MAX_ANGLE_VARIATION_THRESHOLD = 10
TIME_BEFORE_SET = 5
webcam_id = 0

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def draw_text(frame, string, position):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1
    cor = (255, 255, 255)
    espessura = 2

    cv2.putText(frame, string, position, fonte, escala, cor, espessura)

def calcular_angulo(A, B, C):
    """
    Calcula o ângulo ABC (em graus)
    A, B, C são arrays [y, x, score]
    """
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])

    produto = np.dot(BA, BC)
    mag_BA = np.linalg.norm(BA)
    mag_BC = np.linalg.norm(BC)

    if mag_BA == 0 or mag_BC == 0:
        return None

    cos = produto / (mag_BA * mag_BC)
    cos = np.clip(cos, -1.0, 1.0)   # evita erros numéricos

    angulo = math.degrees(math.acos(cos))
    return angulo

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}



# Load model
#interpreter = tf.lite.Interpreter(model_path='models/movenet_lightning/3.tflite')
interpreter = tf.lite.Interpreter(model_path='models/movenet_thunder_tflite/3.tflite')
interpreter.allocate_tensors()
# Setup input and output 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set side
#side = input("side: ")
side = 'r'
hip = 12
knee = 14
heel = 16
if side == "l":
    hip = 11
    knee = 13
    heel = 15
elif side == "r":
    hip = 12
    knee = 14
    heel = 16
# variables used to count reps
movement_initiated = False
rep_count = 0
previous_keypoints = None
current_hip_knee_heel_angle = -1
previous_hip_knee_heel_angle = -1

start_countdown = False
countdown_timer_start = -1
time_left_before_set = -1
time_elapsed = 0

# Make detections
cap = cv2.VideoCapture(webcam_id)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    h, w, _ = frame.shape
    #print(f"frame shape: {h}, {w}")
    size = min(h, w)
    x_start = (w - size) // 2
    y_start = (h - size) // 2
    cropped = frame[y_start:y_start + size, x_start:x_start + size]
    #print(f"cropped shape: {cropped.shape}")

    # Reshape image
    #img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(cropped, axis=0), target_height=target_height, target_width=target_width)
    input_image = tf.cast(img, dtype=tf.float32)
    #print(f"img shape: {img.shape}")
    #print(f"input_image shape: {input_image.shape}")



    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    previous_hip_knee_heel_angle = current_hip_knee_heel_angle
    current_hip_knee_heel_angle = calcular_angulo(keypoints_with_scores[0][0][hip],
                                                      keypoints_with_scores[0][0][knee],
                                                      keypoints_with_scores[0][0][heel])
    trust_hip = keypoints_with_scores[0][0][hip][2]
    trust_knee = keypoints_with_scores[0][0][knee][2]
    trust_heel = keypoints_with_scores[0][0][heel][2]
    #print(f"hip knee heel angle: {current_hip_knee_heel_angle}; "
    #    f"trust score: {trust_hip:.2f}, {trust_knee:.2f}, {trust_heel:.2f}")

    # only analyze frames with high trust
    if trust_hip > TRUST_THRESHOLD and trust_knee > TRUST_THRESHOLD and trust_heel > TRUST_THRESHOLD:
        # ignore first frame
        if previous_hip_knee_heel_angle != -1:
            # ignore frame if angle variation was too great
            if abs(previous_hip_knee_heel_angle - current_hip_knee_heel_angle) < MAX_ANGLE_VARIATION_THRESHOLD:
                # only analyze frame if person is not standing. delta_y < delta_x / 2
                if (abs(keypoints_with_scores[0][0][hip][0] - keypoints_with_scores[0][0][knee][0]) <
                        abs(keypoints_with_scores[0][0][hip][1] - keypoints_with_scores[0][0][knee][1]) / 2):
                    # only analyze frame if person is facing the correct side
                    if ((side == 'r' and keypoints_with_scores[0][0][hip][1] < keypoints_with_scores[0][0][knee][1]) or
                            (side == 'l' and keypoints_with_scores[0][0][hip][1] > keypoints_with_scores[0][0][knee][1])):
                        if not movement_initiated and current_hip_knee_heel_angle < 100:
                            movement_initiated = True
                            print("Movement initiated")
                        if movement_initiated and current_hip_knee_heel_angle > 130:
                            print("Movement finalized")
                            movement_initiated = False
                            rep_count += 1
                    else:
                        movement_initiated = False
                else:
                    movement_initiated = False

    # Rendering 
    draw_connections(cropped, keypoints_with_scores, EDGES, TRUST_THRESHOLD)
    draw_keypoints(cropped, keypoints_with_scores, TRUST_THRESHOLD)

    # Draw information on screen
    if start_countdown:
        time_elapsed = time.time() - countdown_timer_start
        if time_elapsed!= 0:
            time_left_before_set = round(TIME_BEFORE_SET - time_elapsed)
        draw_text(cropped, f"time left: {time_left_before_set}", (10,60))
        if time_elapsed > TIME_BEFORE_SET:
            start_countdown = False

    draw_text(cropped, f"reps: {rep_count}", (10,30))

    cv2.imshow('MoveNet Lightning', cropped)
    #cv2.imshow('original', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    if cv2.waitKey(10) & 0xFF == ord('l'):
        side = 'l'
        hip = 11
        knee = 13
        heel = 15
        start_countdown = True
        countdown_timer_start = time.time()
    if cv2.waitKey(10) & 0xFF == ord('r'):
        side = 'r'
        hip = 12
        knee = 14
        heel = 16
        start_countdown = True
        countdown_timer_start = time.time()

cap.release()
cv2.destroyAllWindows()
