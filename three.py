import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0
stage = ''

# Setup media pipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get left coordinates
            LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get right coordinates
            RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle_left_shoulder = calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
            angle_left_elbow = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
            angle_right_shoulder = calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
            angle_right_elbow = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

            # Visualize angle for testing
            # cv2.putText(image, str(angle_left_shoulder),
            #             tuple(np.multiply(LEFT_SHOULDER, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            #
            # cv2.putText(image, str(angle_right_shoulder),
            #             tuple(np.multiply(RIGHT_SHOULDER, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            #
            # cv2.putText(image, str(angle_left_elbow),
            #             tuple(np.multiply(LEFT_WRIST, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            #
            # cv2.putText(image, str(angle_right_elbow),
            #             tuple(np.multiply(RIGHT_WRIST, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            #             )

            # Left counter logic
            if angle_left_shoulder > 90 and angle_right_shoulder > 90 and angle_right_elbow < 60 and (stage == '' or stage == "right rainbow"):
                stage = "left rainbow"
                counter += 1
            if angle_left_shoulder > 90 and angle_right_shoulder > 90 and angle_left_elbow < 60 and (stage == '' or stage == "left rainbow"):
                stage = "right rainbow"
                counter += 1


        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (700, 73), (245, 117, 16), -1)

        # Rep data
        if counter < 5:
            level = 1
        elif 10 > counter > 5:
            level = 2
        elif 15 > counter > 10:
            level = 3
        elif counter > 15:
            level = 4

        rep_data = 'Count : ' + str(counter) + ' ' + ' Level : ' + str(level)
        cv2.putText(image, rep_data,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Media pipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
