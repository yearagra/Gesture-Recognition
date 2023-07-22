import cv2
import mediapipe as mp
import pygame

# Initialize Pygame
pygame.init()

# Set up the Pygame window
window_width, window_height = 800, 600
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Hand Gesture Recognition")

# Set up the fonts
font = pygame.font.SysFont(None, 48)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture
cap = cv2.VideoCapture(0)


#video
while True:
    # Read frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB and process it with MediaPipe Hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x:
                handedness = "Left"
            else:
                handedness = "Right"

            # Geting the landmarks for thumb, index, middle, ring, and pinky fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Geting the coordinates of the thumb tip
            thumb_tip_x = int(thumb_tip.x * frame.shape[1])
            thumb_tip_y = int(thumb_tip.y * frame.shape[0])

            # Geting the coordinates of the other finger tips
            index_tip_x = int(index_tip.x * frame.shape[1])
            index_tip_y = int(index_tip.y * frame.shape[0])
            middle_tip_x = int(middle_tip.x * frame.shape[1])
            middle_tip_y = int(middle_tip.y * frame.shape[0])
            ring_tip_x = int(ring_tip.x * frame.shape[1])
            ring_tip_y = int(ring_tip.y * frame.shape[0])
            pinky_tip_x = int(pinky_tip.x * frame.shape[1])
            pinky_tip_y = int(pinky_tip.y * frame.shape[0])

            # Calculate the distance between thumb tip and other finger tips
            index_dist = ((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2) ** 0.5
            middle_dist = ((thumb_tip_x - middle_tip_x) ** 2 + (thumb_tip_y - middle_tip_y) ** 2) ** 0.5
            ring_dist = ((thumb_tip_x - ring_tip_x) ** 2 + (thumb_tip_y - ring_tip_y) ** 2) ** 0.5
            pinky_dist = ((thumb_tip_x - pinky_tip_x) ** 2 + (thumb_tip_y - pinky_tip_y) ** 2) ** 0.5

            # Display the text on the screen based on the handedness
            if handedness == 'Left':
                if index_dist < 30:
                    text = font.render("Left hand: Thumb finger tip touching Index finger tip", True, (255, 255, 255))
                elif middle_dist < 30:
                    text = font.render("Left hand: Thumb finger tip touching Middle finger tip", True, (255, 255, 255))
                elif ring_dist < 30:
                    text = font.render("Left hand: Thumb finger tip touching Ring finger tip", True, (255, 255, 255))
                elif pinky_dist < 30:
                    text = font.render("Left hand: Thumb finger tip touching Pinky finger tip", True, (255, 255, 255))
                else:
                    text = font.render("Left hand: No gesture detected", True, (255, 255, 255))
            elif handedness == 'Right':
                if index_dist < 30:
                    text = font.render("Right hand: Thumb finger tip touching Index finger tip", True, (255, 255, 255))
                elif middle_dist < 30:
                    text = font.render("Right hand: Thumb finger tip touching Middle finger tip", True, (255, 255, 255))
                elif ring_dist < 30:
                    text = font.render("Right hand: Thumb finger tip touching Ring finger tip", True, (255, 255, 255))
                elif pinky_dist < 30:
                    text = font.render("Right hand: Thumb finger tip touching Pinky finger tip", True, (255, 255, 255))
                else:
                    text = font.render("Right hand: No gesture detected", True, (255, 255, 255))

            # Display the frame with landmarks
            mp_drawing = mp.solutions.drawing_utils
            annotated_image = frame.copy()
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow('Hand Gesture Recognition', annotated_image)

            # Clear the screen
            screen.fill((0, 0, 0))

            # Display the text in the center of the screen
            text_rect = text.get_rect(center=(window_width // 2, window_height // 2))
            screen.blit(text, text_rect)

    # Update the display
    pygame.display.flip()

    # Check for events and exit if 'q' is pressed
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            cap.release()
            pygame.quit()
            cv2.destroyAllWindows()
            exit()

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
