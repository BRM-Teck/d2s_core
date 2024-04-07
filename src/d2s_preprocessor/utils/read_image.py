from pathlib import Path

import cv2

from d2s_image2text.settings import BASE_DIR


def image_from_file(file: Path):
    file = str(file.resolve().absolute())
    return cv2.imread(file)


if __name__ == "__main__":
    image_dir = BASE_DIR / "data" / "test" / "images"
    img_file = list(image_dir.glob("*.png"))[0]
    image = image_from_file(img_file)
