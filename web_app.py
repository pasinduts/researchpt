import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

red_balloon = cv2.imread('img/balloon.png')

camera = cv2.VideoCapture(0)

score_list = [0, 0, 0, 0]


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def game1():  # generate frame by frame from camera
    # emotion detection
    counter = 0
    stage = ''
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Curl counter variables
            # Curl counter variables

            # Setup media pipe instance
            with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:

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

                    # Get right coordinates
                    RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                    # Calculate angle
                    angle_left = calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
                    angle_right = calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)

                    # Visualize angle for testing
                    # cv2.putText(image, str(angle_left),
                    #             tuple(np.multiply(LEFT_SHOULDER, [640, 480]).astype(int)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    #             )
                    #
                    # cv2.putText(image, str(angle_right),
                    #             tuple(np.multiply(RIGHT_SHOULDER, [640, 480]).astype(int)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    #             )

                    # Left counter logic
                    if angle_left > 150:
                        stage = "left up"
                    if angle_left < 30 and stage == 'right up':
                        stage = "left down"
                        counter += 1
                        print(counter)

                    # Right counter logic
                    if angle_left > 150:
                        stage = "right up"
                    if angle_left < 30 and stage == 'left up':
                        stage = "right down"
                        counter += 1
                        print(counter)

                except:
                    pass

                # Render curl counter
                # Setup status box
                print(image.shape[2])
                cv2.rectangle(image, (0, 0), (image.shape[1], 73), (245, 117, 16), -1)
                cv2.rectangle(image, (0, 146), (image.shape[1], 73), (245, 0, 0), -1)

                # Rep data
                if counter < 5:
                    level = 1
                elif 10 > counter > 5:
                    level = 2
                elif 15 > counter > 10:
                    level = 3
                elif counter > 15:
                    level = 4

                if counter > score_list[0]:
                    score_list[0] = counter

                rep_data = 'Count : ' + str(counter) + ' ' + ' Level : ' + str(level)
                cv2.putText(image, rep_data,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                if level == 1:
                    cv2.putText(image, 'Level One - Try now',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                elif level == 2:
                    cv2.putText(image, 'Level Two - Congratulation',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 3:
                    cv2.putText(image, 'Level Three - Keep Up',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 4:
                    cv2.putText(image, 'Level One - Highest Level',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3,
                                                                 circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=3, circle_radius=2)
                                          )
                image = overlay_transparent(image, red_balloon,
                                            int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x) * 100,
                                            int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y) * 100)
                # print("Real ", landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x)
                # print("int ", int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x)*100)

                resized_img = cv2.resize(image, (1000, 700))
                ret, buffer = cv2.imencode('.jpg', resized_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def game2():  # generate frame by frame from camera
    # emotion detection
    counter = 0
    stage = ''
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Curl counter variables
            # Curl counter variables

            # Setup media pipe instance
            with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
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
                    angle_left_shoulder = calculate_angle(RIGHT_SHOULDER, LEFT_SHOULDER, LEFT_ELBOW)
                    angle_left_elbow = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
                    angle_right_shoulder = calculate_angle(LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW)
                    angle_right_elbow = calculate_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

                    # Visualize angle for testing
                    # cv2.putText(image, str(angle_left_shoulder),
                    #             tuple(np.multiply(LEFT_SHOULDER, [640, 480]).astype(int)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    #             )
                    # #
                    # cv2.putText(image, str(angle_right_shoulder),
                    #             tuple(np.multiply(RIGHT_SHOULDER, [640, 480]).astype(int)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    #             )

                    if angle_left_shoulder < 30 and angle_left_elbow > 80 and (
                            stage == '' or stage == "right punch"):
                        stage = "left punch"
                        counter += 1
                    if angle_right_shoulder < 30 and angle_right_elbow > 80 and (
                            stage == '' or stage == "left punch"):
                        stage = "right punch"
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

                if counter > score_list[1]:
                    score_list[1] = counter

                rep_data = 'Count : ' + str(counter) + ' ' + ' Level : ' + str(level)
                cv2.putText(image, rep_data,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                if level == 1:
                    cv2.putText(image, 'Level One - Try now',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                elif level == 2:
                    cv2.putText(image, 'Level Two - Congratulation',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 3:
                    cv2.putText(image, 'Level Three - Keep Up',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 4:
                    cv2.putText(image, 'Level One - Highest Level',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                resized_img = cv2.resize(image, (1000, 700))
                ret, buffer = cv2.imencode('.jpg', resized_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def game3():  # generate frame by frame from camera
    # emotion detection
    counter = 0
    stage = ''
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Curl counter variables
            # Curl counter variables

            # Setup media pipe instance
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

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
                    if angle_left_shoulder > 90 and angle_right_shoulder > 90 and angle_right_elbow < 60 and (
                            stage == '' or stage == "right rainbow"):
                        stage = "left rainbow"
                        counter += 1
                    if angle_left_shoulder > 90 and angle_right_shoulder > 90 and angle_left_elbow < 60 and (
                            stage == '' or stage == "left rainbow"):
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

                if counter > score_list[2]:
                    score_list[2] = counter

                rep_data = 'Count : ' + str(counter) + ' ' + ' Level : ' + str(level)
                cv2.putText(image, rep_data,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                if level == 1:
                    cv2.putText(image, 'Level One - Try now',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                elif level == 2:
                    cv2.putText(image, 'Level Two - Congratulation',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 3:
                    cv2.putText(image, 'Level Three - Keep Up',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 4:
                    cv2.putText(image, 'Level One - Highest Level',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                resized_img = cv2.resize(image, (1000, 700))
                ret, buffer = cv2.imencode('.jpg', resized_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def game4():  # generate frame by frame from camera
    # emotion detection

    counter = 0
    stage = ''
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Curl counter variables
            # Curl counter variables

            # Setup media pipe instance
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

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

                    # Get right coordinates
                    RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    # Calculate angle
                    angle_left_hip = calculate_angle(LEFT_SHOULDER, LEFT_HIP, RIGHT_HIP)
                    angle_right_hip = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP)

                    # # Visualize angle for testing
                    cv2.putText(image, str(angle_left_hip),
                                tuple(np.multiply(LEFT_SHOULDER, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    cv2.putText(image, str(angle_right_hip),
                                tuple(np.multiply(RIGHT_SHOULDER, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    # Left counter logic
                    if angle_left_hip < 75 and (stage == '' or stage == "right trunk"):
                        stage = "left trunk"
                        counter += 1
                    if angle_right_hip < 75 and (stage == '' or stage == "left trunk"):
                        stage = "right trunk"
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

                if counter > score_list[3]:
                    score_list[3] = counter

                rep_data = 'Count : ' + str(counter) + ' ' + ' Level : ' + str(level)
                cv2.putText(image, rep_data,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                if level == 1:
                    cv2.putText(image, 'Level One - Try now',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                elif level == 2:
                    cv2.putText(image, 'Level Two - Congratulation',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 3:
                    cv2.putText(image, 'Level Three - Keep Up',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                elif level == 4:
                    cv2.putText(image, 'Level One - Highest Level',
                                (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                resized_img = cv2.resize(image, (1000, 700))
                ret, buffer = cv2.imencode('.jpg', resized_img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
