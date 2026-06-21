import tensorflow as tf
import numpy as np
import cv2
import math
import time

target_width = 256
target_height = 256
TRUST_THRESHOLD = 0.1
MAX_ANGLE_VARIATION_THRESHOLD = 10
TIME_BEFORE_SET = 5
MAX_MISSING_FRAMES = 10
START_ANGLE = 105
END_ANGLE = 150
FRAMES_PER_STATE = 2
MAX_STANDING_UP_FRAMES = 5
MAX_LOW_TRUST_FRAMES = 10
MAX_FACING_WRONG_SIDE_FRAMES = 5
webcam_id = 1
total_frames = 0

# Movement states
STATE_RESTING = "REST" # hip-knee-heel angle below START_ANGLE
STATE_EXTENDED = "EXTENDED" # hip-knee-heel angle above END_ANGLE
STATE_LESS_THAN_HALFWAY = "LESS THAN HALF" # START_ANGLE <= current_hip_knee_heel_angle <= halfway
STATE_MORE_THAN_HALFWAY = "MORE THAN HALF" # halfway <= current_hip_knee_heel_angle <= END_ANGLE
STATE_NONE = "NONE" # state of the movement should be STATE_NONE when :
                    #   - Program has just started
                    #   - No person detected in the frame
                    #   - Person not in the correct position
halfway = START_ANGLE + (END_ANGLE - START_ANGLE) / 2


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
#interpreter = tf.lite.Interpreter(model_path='models/movenet_lightning_tflite/3.tflite')
interpreter = tf.lite.Interpreter(model_path='models/movenet_thunder_tflite/3.tflite')
interpreter.allocate_tensors()
# Setup input and output 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Side the person is facing according to the model
predicted_side = 'r'

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

# variables used to start the countdown after switching sides
start_countdown = False
countdown_timer_start = -400
time_left_before_set = -400
time_elapsed = 0

# variables used to check if the person is in the correct position
position_consecutive_frame_counter = 0
pose_state = STATE_NONE
standing_up_consecutive_frame_counter = 0
low_trust_consecutive_frame_counter = 0
facing_wrong_side_consecutive_frame_counter = 0
start_time = 0

cap = cv2.VideoCapture(webcam_id)
start_time_global = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    total_frames += 1

    if not ret:
        print("Failed to capture frame")
        break

    h, w, _ = frame.shape
    size = min(h, w)
    x_start = (w - size) // 2
    y_start = (h - size) // 2
    cropped = frame[y_start:y_start + size, x_start:x_start + size]

    # Reshape image
    img = tf.image.resize_with_pad(np.expand_dims(cropped, axis=0), target_height=target_height, target_width=target_width)
    input_image = tf.cast(img, dtype=tf.float32)

    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    trust_hip = keypoints_with_scores[0][0][hip][2]
    trust_knee = keypoints_with_scores[0][0][knee][2]
    trust_heel = keypoints_with_scores[0][0][heel][2]
    # ignore frames with low trust
    if trust_hip > TRUST_THRESHOLD and trust_knee > TRUST_THRESHOLD and trust_heel > TRUST_THRESHOLD:
        low_trust_consecutive_frame_counter = 0 # reset low_trust_consecutive_frame_counter
        # only analyze frame if person is not standing. delta_y < delta_x / 2
        if (abs(keypoints_with_scores[0][0][hip][0] - keypoints_with_scores[0][0][knee][0]) <
            abs(keypoints_with_scores[0][0][hip][1] - keypoints_with_scores[0][0][knee][1]) / 2):
            standing_up_consecutive_frame_counter = 0 # reset standing_up_consecutive_frame_counter
            # only analyze frame if person is facing the correct side
            if ((side == 'r' and keypoints_with_scores[0][0][hip][1] < keypoints_with_scores[0][0][knee][1]) or
                    (side == 'l' and keypoints_with_scores[0][0][hip][1] > keypoints_with_scores[0][0][knee][1])):
                if side == 'r':
                    predicted_side = 'r'
                else:
                    predicted_side = 'l'

                facing_wrong_side_consecutive_frame_counter = 0 # reset facing_wrong_side_consecutive_frame_counter

                current_hip_knee_heel_angle = calcular_angulo(keypoints_with_scores[0][0][hip],
                                                              keypoints_with_scores[0][0][knee],
                                                              keypoints_with_scores[0][0][heel])
                #print(f"current angle: {current_hip_knee_heel_angle}")

                if pose_state != STATE_NONE:
                    if pose_state == STATE_RESTING:
                        if START_ANGLE <= current_hip_knee_heel_angle <= halfway:
                            position_consecutive_frame_counter += 1
                        if position_consecutive_frame_counter == FRAMES_PER_STATE:
                            position_consecutive_frame_counter = 0
                            pose_state = STATE_LESS_THAN_HALFWAY
                    elif pose_state == STATE_LESS_THAN_HALFWAY:
                        if halfway <= current_hip_knee_heel_angle <= END_ANGLE:
                            position_consecutive_frame_counter += 1
                        if position_consecutive_frame_counter == FRAMES_PER_STATE:
                            position_consecutive_frame_counter = 0
                            pose_state = STATE_MORE_THAN_HALFWAY
                    elif pose_state == STATE_MORE_THAN_HALFWAY:
                        if END_ANGLE <= current_hip_knee_heel_angle:
                            position_consecutive_frame_counter += 1
                        if position_consecutive_frame_counter == FRAMES_PER_STATE:
                            position_consecutive_frame_counter = 0
                            pose_state = STATE_EXTENDED
                            rep_count += 1
                    elif pose_state == STATE_EXTENDED:
                        if current_hip_knee_heel_angle <= START_ANGLE:
                            position_consecutive_frame_counter += 1
                        if position_consecutive_frame_counter == FRAMES_PER_STATE:
                            position_consecutive_frame_counter = 0
                            pose_state = STATE_RESTING
                else:
                    if current_hip_knee_heel_angle <= START_ANGLE:
                        pose_state = STATE_RESTING
            else:
                if side == 'r':
                    predicted_side = 'l'
                else:
                    predicted_side = 'r'
                if facing_wrong_side_consecutive_frame_counter != -1:
                    facing_wrong_side_consecutive_frame_counter += 1
                if facing_wrong_side_consecutive_frame_counter == MAX_FACING_WRONG_SIDE_FRAMES:
                    print("Facing wrong side!")
                    pose_state = STATE_NONE
                    position_consecutive_frame_counter = 0
                    facing_wrong_side_consecutive_frame_counter = -1
        else:
            if standing_up_consecutive_frame_counter != -1:
                standing_up_consecutive_frame_counter += 1
            if standing_up_consecutive_frame_counter == MAX_STANDING_UP_FRAMES:
                print("Subject standing up!")
                pose_state = STATE_NONE
                position_consecutive_frame_counter = 0
                standing_up_consecutive_frame_counter = -1
    else:
        if low_trust_consecutive_frame_counter != -1:
            low_trust_consecutive_frame_counter += 1
        if low_trust_consecutive_frame_counter == MAX_LOW_TRUST_FRAMES:
            print("Low trust frames!")
            pose_state = STATE_NONE
            position_consecutive_frame_counter = 0
            low_trust_consecutive_frame_counter = -1

    # Rendering 
    draw_connections(cropped, keypoints_with_scores, EDGES, TRUST_THRESHOLD)
    draw_keypoints(cropped, keypoints_with_scores, TRUST_THRESHOLD)

    # show countdown on screen
    if start_countdown:
        time_elapsed = time.time() - countdown_timer_start
        if time_elapsed!= 0:
            time_left_before_set = round(TIME_BEFORE_SET - time_elapsed)
        draw_text(cropped, f"time left: {time_left_before_set}", (200,30))
        if time_elapsed > TIME_BEFORE_SET:
            start_countdown = False



    # Draw information on screen
    draw_text(cropped, f"reps: {rep_count}", (10,30))
    draw_text(cropped, f"state: {pose_state}", (10,60))
    draw_text(cropped, f"expected side: {side}", (10,120))
    draw_text(cropped, f"predicted side: {predicted_side}", (10,150))

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        side = 'l'
        hip = 11
        knee = 13
        heel = 15
        start_countdown = True
        countdown_timer_start = time.time()
    elif key == ord('r'):
        side = 'r'
        hip = 12
        knee = 14
        heel = 16
        start_countdown = True
        countdown_timer_start = time.time()

    end_time = time.time()
    real_fps = 1 / (end_time - start_time)
    # used to calculate FPS
    start_time = time.time()
    draw_text(cropped, f"fps: {real_fps:.2f}", (10,90))
    cv2.imshow('MoveNet Thunder', cropped)
    # cv2.imshow('original', frame)

end_time_global = time.time()
print(f"FPS AVG: {total_frames / (end_time_global - start_time_global)}")
cap.release()
cv2.destroyAllWindows()
