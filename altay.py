import cv2
import numpy as np
from collections import deque
import serial
import time


ser = serial.Serial ('COM5', 9600)
time.sleep(2)

# Constants
REAL_RADIUS = 3.15  # Real radius of the ball in cm
FOCAL_LENGTH = 885  # Camera focal length in mm
DODGE_THRESHOLD = 135  # Minimum distance required to dodge in cm
DODGE_SIGNAL_FRAME_VALUE = 20  # Number of dodge signals to display in the window
RESET_COUNTER = 100  # Number of "Stand Still" commands before resetting
num_frames_average = 30 # Define the number of frames to consider for average calculation

# Initialize video capture
cap = cv2.VideoCapture(2)

# Check if the camera is opened
if not cap.isOpened():
    print("Failed to open the camera.")
    exit(1)

# Get the frame dimensions
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
middle_point = int(width // 2)

# Blur kernel size for image blurring
blur_kernel_size = (25, 25)

# Variables for blurred frame and HSV color space
blurred_frame = None
hsv = None

# Queue to store center points and distance values
center_points = deque(maxlen=num_frames_average)
distance_values = deque(maxlen=num_frames_average)
dodge_signals = deque(maxlen=DODGE_SIGNAL_FRAME_VALUE)# Queue to store dodge signals

# Counters
stand_still_counter = 0 # Counter for "Stand Still" commands
dodge_signal_counter = 0 # Counter for dodge signals
last_signal = None


# Text settings
text_size = 0.6
text_thickness = 1
text_margin = 10

# Create a window to display dodge signals
dodge_signals_window_width = 400
dodge_signals_window_height = 600
cv2.namedWindow("dodge_signals", cv2.WINDOW_NORMAL)
cv2.moveWindow("dodge_signals", int(width), 0)
cv2.resizeWindow("dodge_signals", dodge_signals_window_width, dodge_signals_window_height)


def draw_text(image, text, position, color):
    text_size_actual = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness
    )[0]

    text_position = (position[0], position[1] + text_size_actual[1] + text_margin)

    cv2.putText(
        image,
        text,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        color,
        text_thickness,
        cv2.LINE_AA,
    )



# Main loop
while True:
    
    dodge_signal = "Stand Still"
    main_signal = "None"

    # Read the camera frame
    ret, frame = cap.read()

    # Break if failed to read the frame
    if not ret:
        print("Failed to read the frame.")
        break

    # Blur the frame and convert to HSV color space
    blurred_frame = cv2.GaussianBlur(frame, blur_kernel_size, 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper HSV color range for yellowish green
    lower_yellowish_green = np.array([35, 100, 90])
    upper_yellowish_green = np.array([77, 255, 255])

    # Create a mask using the color range and dilate it
    mask = cv2.inRange(hsv, lower_yellowish_green, upper_yellowish_green)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw reference lines on the frame
    cv2.line(frame, (middle_point, 0), (middle_point, int(height)), (255, 255, 255), 1)
    cv2.line(frame, (0, int(height // 2)), (int(width), int(height // 2)), (255, 255, 255), 1)
    cv2.putText(frame, "(0, 0)", (middle_point + 5, int(height // 2) - 5), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            # Draw the circle around the ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Calculate the center of the ball
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Draw the center of the ball
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            # Calculate the distance to the ball
            distance = (REAL_RADIUS * FOCAL_LENGTH) / radius

            # Calculate the X position of the ball
            ball_x, ball_y = center[0], center[1]
            relative_x = ball_x - middle_point

            # Display X position and distance on the frame
            draw_text(frame, "X: {}".format(relative_x), (10, 70), (255, 255, 255))
            draw_text(frame, "Distance: {:.2f} cm".format(distance), (10, 100), (255, 255, 255))

            # Store center and distance values
            center_points.append(center)
            distance_values.append(distance)

            # If 30 values are available, calculate movement and distance change
            if len(center_points) >= num_frames_average:
                x_changes = [center_points[i][0] - center_points[i - 1][0] for i in range(1, len(center_points))]
                avg_x_change = sum(x_changes[-num_frames_average:]) / num_frames_average
                avg_distance_change = (distance_values[-1] - distance_values[0]) / num_frames_average


                # Display average X change and distance change
                draw_text(frame, "Avg X Change: {:.2f}".format(avg_x_change), (10, 130), (255, 255, 255))
                draw_text(frame, "Avg Distance Change: {:.2f} cm".format(avg_distance_change), (10, 160), (255, 255, 255))

                # Determine the dodge strategy
                if 40 < distance < 300 and avg_distance_change < -0.8 and distance <= DODGE_THRESHOLD:
                    if avg_x_change < -1.3:
                        dodge_signal = "Dodge Right!"
                        dodge_signals.appendleft(1)
                        dodge_signal_counter += 1

                    elif avg_x_change > 1.3:
                        dodge_signal = "Dodge Left!"
                        dodge_signals.appendleft(-1)
                        dodge_signal_counter += 1

                    elif -1.3 < avg_x_change < 1.3:
                        dodge_signal = "Dodge Left or Right!"
                        dodge_signals.appendleft(0)
                        dodge_signal_counter += 1

                        

                    # Display Dodge Right, Dodge Left, or Dodge Left or Right
                    if dodge_signals_sum > 0 and dodge_signal_counter == 7:
                        main_signal = "Right"
                        if main_signal != last_signal:
                            last_signal = main_signal
                            print(main_signal,DODGE_THRESHOLD, relative_x, avg_x_change, distance)
                            ser.write(b'F')

                            
                    elif dodge_signals_sum < 0 and dodge_signal_counter == 7:
                        main_signal = "Left"
                        if main_signal != last_signal:
                            last_signal = main_signal
                            print(main_signal,DODGE_THRESHOLD, relative_x, avg_x_change, distance)

                    elif dodge_signals_sum == 0 and  dodge_signal_counter == 7 and 41 >= relative_x > 0:
                            main_signal = "Left(X)"
                            if main_signal != last_signal:
                                last_signal = main_signal
                                print(main_signal,DODGE_THRESHOLD, relative_x, avg_x_change, distance)

                    elif dodge_signals_sum == 0 and  dodge_signal_counter == 7 and -41 <= relative_x < 0:
                            main_signal = "Right(X)"
                            if main_signal != last_signal:
                                last_signal = main_signal
                                print(main_signal,DODGE_THRESHOLD, relative_x, avg_x_change, distance)

                    #####################################################################################

                    elif dodge_signals_sum == 0 and  dodge_signal_counter == 7 and relative_x < -41:
                            main_signal = "Left(XY)"
                            if main_signal != last_signal:
                                last_signal = main_signal
                                print(main_signal,DODGE_THRESHOLD, relative_x, avg_x_change, distance)

                    elif dodge_signals_sum == 0 and  dodge_signal_counter == 7 and relative_x > 41:
                            main_signal = "Right(XY)"
                            if main_signal != last_signal:
                                last_signal = main_signal
                                print(main_signal,DODGE_THRESHOLD, relative_x, avg_x_change, distance)            

                    
            # Display dodge command on the frame and print to console
            draw_text(frame, dodge_signal, (10, 190), (255, 255, 255))
            draw_text(frame, last_signal, (10, 250), (255, 0, 255))

            # If "Stand Still" command is given 100 times, reset deque and counters
            if dodge_signal == "Stand Still":
                stand_still_counter += 1

            if stand_still_counter >= RESET_COUNTER:
                dodge_signals.clear()
                last_signal = "None"
                stand_still_counter = 0
                dodge_signal_counter = 0

    # Show the frame
    cv2.imshow("frame", frame)

    # Display dodge signals in the window
    dodge_signals_str = ", ".join(str(signal) for signal in list(dodge_signals)[:DODGE_SIGNAL_FRAME_VALUE])
    dodge_signals_frame = np.zeros((dodge_signals_window_height, dodge_signals_window_width, 3), dtype=np.uint8)

    dodge_signals_sum = sum(list(dodge_signals)[:DODGE_SIGNAL_FRAME_VALUE])

    draw_text(dodge_signals_frame, "Dodge Signals:", (10, 90), (255, 255, 255))
    draw_text(dodge_signals_frame, "Sum: {}".format(dodge_signals_sum), (10, 30), (255, 255, 255))
    draw_text(dodge_signals_frame, "Dodge Signal Counter: {}".format(dodge_signal_counter), (10, 60), (255, 255, 255))

    signal_position_y = 120
    for signal in list(dodge_signals)[:DODGE_SIGNAL_FRAME_VALUE]:
        signal_text = str(signal)
        draw_text(dodge_signals_frame, signal_text, (10, signal_position_y), (255, 255, 255))
        signal_position_y += 20

    cv2.imshow("dodge_signals", dodge_signals_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ser.close()
# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

