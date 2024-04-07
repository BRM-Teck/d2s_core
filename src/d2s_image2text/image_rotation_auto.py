import math

import cv2
import numpy as np

from .settings import BASE_DIR


def detect_angle(image: np.ndarray):
    # Convert the image to grayscale
    num_channels = len(image.shape)
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if num_channels != 2 else image
    # Apply a Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply the Canny edge detection algorithm
    edges = cv2.Canny(gray, 50, 150)

    # Use the Hough transform to detect lines in the image
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )

    if not isinstance(lines, np.ndarray) or not lines.all():
        return 0

    # Calculate the average angle of the lines
    angles = []

    for [[x1, y1, x2, y2]] in lines:
        # cv2.line(gray, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    return np.median(angles)


def rotate(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def detect_angle_rotate(img: np.ndarray, output_path=None):
    if output_path is None:
        output_path = BASE_DIR / "data/test/outputs/image_preprocess"
        if not output_path.exists():
            output_path.mkdir(exist_ok=True, parents=True)
        nbr_files = len(list(output_path.iterdir()))
        output_path = output_path / f"image_{nbr_files}.png"
        output_path = str(output_path.resolve().absolute())
    median_angle = detect_angle(img)
    img_rotated = rotate(img, median_angle)
    cv2.imwrite(output_path, img_rotated)
    return output_path, img_rotated


if __name__ == "__main__":
    from utils import image_from_file

    file = BASE_DIR / "data/test/images/FAC FINE EXPORT 3009-1.png"
    detect_angle_rotate(image_from_file(file))
