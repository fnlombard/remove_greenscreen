import os
from pathlib import Path
import cv2
import numpy as np
import argparse

Matt = np.ndarray[np.uint8]
Video = np.ndarray


def load_video(input_file: Path) -> Video:
    cap = cv2.VideoCapture(input_file)
    file_type = input_file.suffix
    file_name = input_file.stem

    # grab one frame
    scale = 0.5
    _, frame = cap.read()
    h, w = frame.shape[:2]
    h = int(h * scale)
    w = int(w * scale)

    # videowriter
    res = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("test_vid.avi", fourcc, 30.0, res)

    # loop
    done = False
    while not done:
        # get frame
        ret, img = cap.read()
        if not ret:
            done = True
            continue

        # resize
        img = cv2.resize(img, res)

        # change to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # get uniques
        unique_colors, counts = np.unique(s, return_counts=True)

        # sort through and grab the most abundant unique color
        big_color = None
        biggest = -1
        for a in range(len(unique_colors)):
            if counts[a] > biggest:
                biggest = counts[a]
                big_color = int(unique_colors[a])

        # get the color mask
        margin = 50
        mask = cv2.inRange(s, big_color - margin, big_color + margin)


def remove_screen(image: Matt) -> Matt:
    # load image
    img = cv2.imread("greenscreen.jpg")

    # convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # extract A channel
    A = lab[:, :, 1]

    # threshold A channel
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # blur threshold image
    blur = cv2.GaussianBlur(
        thresh, (0, 0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_DEFAULT
    )

    # stretch so that 255 -> 255 and 127.5 -> 0
    blur_clipped = np.clip(blur, 127.5, 255).astype(np.uint8)
    mask = cv2.normalize(
        blur_clipped,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    # add mask to image as alpha channel
    result = img.copy()
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    # save output
    cv2.imwrite("greenscreen_thresh.png", thresh)
    cv2.imwrite("greenscreen_mask.png", mask)
    cv2.imwrite("greenscreen_antialiased.png", result)

    # Display various images to see the steps
    cv2.imshow("A", A)
    cv2.imshow("thresh", thresh)
    cv2.imshow("blur", blur)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_file", help="Input video file path.", required=True
    )
    args = parser.parse_args()

    if not (input_file := Path(args.input_file)).exists() and not input_file.is_file():
        raise RuntimeError(f"{input_file} does not exist or is not a file.")

    output_directory = input_file.parent
    file_type = input_file.suffix
    file_name = input_file.stem

    input_video = load_video(input_file)


if __name__ == "__main__":
    main()
