import cv2
import numpy as np
import argparse
import time

def perspective_correction(frame, top_percent, bottom_percent):
    """Applies perspective correction with adjustable top and bottom."""

    height, width = frame.shape[:2]

    src_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

    new_width_top = int(width * top_percent)
    new_width_bottom = int(width * bottom_percent)

    shift_x_top = int((width - new_width_top) / 2)
    shift_x_bottom = int((width - new_width_bottom) / 2)

    dst_pts = np.float32([[shift_x_top, 0], [new_width_top + shift_x_top - 1, 0],
                          [shift_x_bottom, height - 1], [new_width_bottom + shift_x_bottom - 1, height - 1]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_frame = cv2.warpPerspective(frame, M, (width, height))
    return warped_frame


def main():
    parser = argparse.ArgumentParser(description="Apply perspective correction.")
    parser.add_argument("--top_percent", type=float, default=0.8, help="Top percentage.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    bottom_percent = args.top_percent #start bottom with the same percentage as top

    fps = 30
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        corrected_frame = perspective_correction(frame, args.top_percent, bottom_percent)

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
        elif key & 0xFF == ord('+'):
            args.top_percent = min(1.0, args.top_percent + 0.05)
            print(f"Top Percentage: {args.top_percent:.2f}")
        elif key & 0xFF == ord('-'):
            args.top_percent = max(0.1, args.top_percent - 0.05)
            print(f"Top Percentage: {args.top_percent:.2f}")
        elif key & 0xFF == ord('0'):  # Increase bottom percentage
            bottom_percent = min(1.0, bottom_percent + 0.05)
            print(f"Bottom Percentage: {bottom_percent:.2f}")
        elif key & 0xFF == ord('9'):  # Decrease bottom percentage
            bottom_percent = max(0.1, bottom_percent - 0.05)
            print(f"Bottom Percentage: {bottom_percent:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()