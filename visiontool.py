import cv2
import numpy as np

def perspective_correction(frame):
    """Applies perspective correction to the input frame."""

    height, width = frame.shape[:2]

    # Define the source points (corners of the frame)
    src_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

    # Define the destination points (adjusted corners)
    new_width = int(width * 0.5)  # 80% of original width
    dst_pts = np.float32([[0, 0], [new_width -1, 0], [0, height - 1], [new_width-1, height - 1]])


    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transform
    warped_frame = cv2.warpPerspective(frame, M, (new_width, height)) #output size change
    return warped_frame


def main():
    """Captures video from the camera, applies perspective correction, and displays the result."""

    cap = cv2.VideoCapture(0)  # 0 usually represents the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fps = 30  # Target frame rate
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        corrected_frame = perspective_correction(frame)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(corrected_frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Corrected Frame", corrected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import time
    main()