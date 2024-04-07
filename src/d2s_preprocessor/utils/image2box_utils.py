import re
import numpy as np
import pandas as pd

from image2text import get_extractor
from image2text.paddle_ocr import TextExtractorPaddleOCR
from settings import BASE_DIR


def dataframe_cleanup(dataframe: pd.DataFrame):
    mask = ~dataframe.text.isna() & ~dataframe.text.str.strip().isin(["|", ""])
    dataframe = dataframe[mask]
    dataframe.reset_index(drop=True)
    return dataframe


def load_data_bboxes(
    limit=None, images_file=None, clean=True, with_bbox=False, log=True
):
    if images_file is None:
        images_dir = BASE_DIR / "data" / "test" / "images"
        images_file = images_dir.iterdir()

    extractor = get_extractor(TextExtractorPaddleOCR.engine_name)
    dataframe_list = []
    index = 0
    full_text_dict = {}
    for img_file in images_file:
        if img_file.is_file() and img_file.suffix == ".png":
            if log:
                print(
                    f"===============================> Process {img_file.stem} <=============================="
                )
            if with_bbox:
                boxes_df = extractor.extract_boxes(str(img_file.absolute()))
                if clean:
                    boxes_df = dataframe_cleanup(boxes_df)
                boxes_df["filename"] = np.full(len(boxes_df), img_file.stem)
            else:
                full_text = extractor.extract(str(img_file.absolute()))

                if limit == 1:
                    full_text_dict = full_text
                else:
                    full_text_dict[img_file.stem] = full_text

                text = re.split("\n|\|", full_text)
                boxes_df = pd.DataFrame(
                    {"text": text, "filename": [img_file.stem] * len(text)}
                )

            dataframe_list.append(boxes_df)
            index += 1
        if limit is not None and index == limit:
            break
    return pd.concat(dataframe_list).reset_index(drop=True), full_text_dict
