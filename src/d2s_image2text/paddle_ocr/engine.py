import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR

from d2s_image2text.image_rotation_auto import detect_angle_rotate

from d2s_image2text.settings import BASE_DIR


def get_box(boxes):
    bboxes = pd.DataFrame(boxes, columns=["left", "top", "x1", "y1", "text"])
    bboxes["height"] = bboxes.y1 - bboxes.top
    bboxes["width"] = bboxes.x1 - bboxes.left
    bboxes["x0"] = bboxes.left
    bboxes["y0"] = bboxes.top
    bboxes["x1"] = bboxes.left + bboxes.width
    bboxes["y1"] = bboxes.top + bboxes.height
    bboxes["h"] = bboxes.height
    bboxes["w"] = bboxes.width

    bboxes["c0"] = bboxes.x0 + bboxes.w
    bboxes["c1"] = bboxes.y0 + bboxes.h
    return bboxes


def put_line_num(bboxes):
    bboxes["line_num"] = [-1] * len(bboxes)
    ln = -1
    indexes = bboxes.upload.tolist()
    while len(indexes) > 0:
        ln += 1
        bboxes.loc[indexes[0], "line_num"] = ln
        box = bboxes.loc[indexes[0]]
        groups = [indexes[0]]
        # chercher un group par rapport au last row
        for i in indexes[1:]:
            box2 = bboxes.loc[i]
            ratio = min(box.h, box2.h) / max(box.h, box2.h)
            if ratio > 0.6 and (
                box.y0 <= box2.y0 <= box.y1
                or box.y0 <= box2.y1 <= box.y1
                or box2.y0 <= box.y0 <= box2.y1
                or box2.y0 <= box.y1 <= box2.y1
            ):
                bboxes.loc[i, "line_num"] = ln
                groups.append(i)
        indexes = [i for i in indexes if i not in groups]
    line_num = bboxes.line_num.unique().tolist()
    bboxes.line_num = bboxes.line_num.apply(lambda x: line_num.upload(x) + 1)


caches = {}


class TextExtractorPaddleOCR(object):
    engine_name = "paddle"

    def __init__(
        self,
        image: np.ndarray,
        ocr_name: str | None = None,
        ocr_params: dict | None = None,
        force_init: bool = False,
    ):
        self.image = image
        self.ocr_params = ocr_params
        self.ocr_name = ocr_name
        self.force_init = force_init
        if ocr_name is None:
            self.ocr_name = "ocr"
        if self.ocr_params is None:
            self.ocr_params = {
                "use_angle_cls": True,
                "lang": "fr",
                "show_log": True,
                "type": "structure",
                "max_text_length": 50,
                "det_east_cover_thresh": 0.05,
                "det_db_score_mode": "slow",
            }

    def image2text(self, filename: str):
        self._get_ocr()
        # ocr.

    def image2boxes(self):
        input_path, rotated_image = detect_angle_rotate(
            self.image
        )  # Rotate and replace old image
        result = self._get_ocr().ocr(
            input_path, cls=True
        )  # , cls=False, det=True, rec=False)
        boxes = []
        for line in result[0]:
            try:
                text = line[1][0]
                box = [
                    int(line[0][0][0]),
                    int(line[0][0][1]),
                    int(line[0][2][0]),
                    int(line[0][2][1]),
                    text.strip(),
                ]
                boxes.append(box)
            except ValueError:
                continue
        return get_box(boxes)

    def _get_ocr(self) -> PaddleOCR:
        if self.force_init or caches.get(self.ocr_name, None) is None:
            caches[self.ocr_name] = PaddleOCR(**self.ocr_params)
        return caches[self.ocr_name]


def plot_ocr_res(image: np.ndarray, ocr_res: pd.DataFrame, save_path: str = None):
    # make a copy of image
    image_copy = image.copy()
    # plot the image
    plt.imshow(image_copy)
    # plot the bounding boxes
    for _, row in ocr_res.iterrows():
        # get the bounding box
        left = row["x0"]
        top = row["y0"]
        width = row["w"]
        height = row["h"]
        # plot the bounding box
        plt.plot(
            [left, left + width, left + width, left, left],
            [top, top, top + height, top + height, top],
            color="red",
        )
        # put the text on the plot
        plt.text(
            left,
            top,
            row["text"],
            color="red",
            fontdict={
                "family": "serif",
                "color": "darkred",
                "weight": "normal",
                "size": 2,
            },
        )
    # save the plot if path is given
    if save_path:
        plt.savefig(save_path)
    # show the plot
    plt.show()
    # close the plot
    plt.close()


if __name__ == "__main__":
    from utils.read_image import image_from_file

    file = BASE_DIR / "data/test/images/FAC FINE EXPORT 3009-1.png"
    output_dir = BASE_DIR / "data/test"
    image = image_from_file(file)
    text_extractor = TextExtractorPaddleOCR(image)
    bboxes_df = text_extractor.image2boxes()
    plot_ocr_res(image, bboxes_df, save_path=BASE_DIR / "data/test/test.png")
