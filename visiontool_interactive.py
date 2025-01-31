import cv2
import numpy as np
import argparse
import time

import cv2
import numpy as np
import argparse
import time

import cv2
import numpy as np
import argparse
import time

import cv2
import numpy as np
import argparse
import time

def perspective_correction(frame, top_percent):
    """Applies perspective correction with integrated shift."""

    height, width = frame.shape[:2]

    src_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

    new_width_top = int(width * top_percent)

    shift_x = int((width - new_width_top) / 2) #calculate the shift

    # Adjust destination points for both top corners
    dst_pts = np.float32([[shift_x, 0], [new_width_top + shift_x - 1, 0], [0, height - 1], [width - 1, height - 1]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_frame = cv2.warpPerspective(frame, M, (width, height)) #output size is the original size
    return warped_frame

def main():
    """Captures video, applies perspective correction, and displays."""

    parser = argparse.ArgumentParser(description="Apply perspective correction to camera feed.")
    parser.add_argument("--top_percent", type=float, default=0.8,
                        help="Percentage of width for top corner adjustment (e.g., 0.8 for 80%).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)

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

        corrected_frame = perspective_correction(frame, args.top_percent)  # Pass the argument

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(corrected_frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Corrected Frame", corrected_frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('+'):  # Increase percentage
            args.top_percent = min(1.0, args.top_percent + 0.05) #limit to 100%
            print(f"Top Percentage: {args.top_percent:.2f}")
        elif key & 0xFF == ord('-'):  # Decrease percentage
            args.top_percent = max(0.1, args.top_percent - 0.05) #limit to 10%
            print(f"Top Percentage: {args.top_percent:.2f}")



    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()