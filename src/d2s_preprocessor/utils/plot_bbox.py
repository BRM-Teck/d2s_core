from pathlib import Path, PosixPath
import random

import cv2
import pandas as pd
from matplotlib import pyplot as plt

from settings import BASE_DIR
from utils.image2box_utils import load_data_bboxes


def put_text(
    img,
    text,
    left,
    top,
    color=(0, 0, 255),
    thickness=1,
    font_scale=1.0,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    return cv2.putText(
        img, text, (left, top), font, font_scale, color, thickness, cv2.LINE_AA, False
    )


def plot_by_line_number(
    dataframes: pd.DataFrame,
    img_file: Path,
    save=False,
    filepath=None,
    figsize=(40, 20),
    text_top=10,
    write_text=True,
    text_font_scale=0.5,
    thickness=1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    assert "left" in dataframes
    assert "top" in dataframes
    assert "width" in dataframes
    assert "height" in dataframes
    # Plot each line with a different color
    img = cv2.imread(img_file)

    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _, row in dataframes.iterrows():
        x, y, w, h = row["left"], row["top"], row["width"], row["height"]
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if write_text:
            img = put_text(
                img,
                row["text"],
                x,
                y - text_top,
                color=color,
                font_scale=text_font_scale,
                thickness=thickness,
                font=font,
            )

    if save and filepath:
        if isinstance(filepath, PosixPath):
            filepath = str(filepath.absolute())
        cv2.imwrite(filepath, img)
    else:
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap="rainbow")


if __name__ == "__main__":
    file = BASE_DIR / "images/fact-bieco-5.png"
    bboxes = load_data_bboxes(limit=1, images_file=[file], clean=True, with_bbox=True)[
        0
    ]
    plot_by_line_number(
        bboxes,
        file,
        figsize=(40, 20),
        write_text=True,
        text_top=2,
        save=True,
        filepath=str((BASE_DIR / "images/fact-bieco-5-text.png").resolve()),
    )
