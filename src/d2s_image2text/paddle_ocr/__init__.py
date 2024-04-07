from d2s_image2text.paddle_ocr.engine import TextExtractorPaddleOCR
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def get_extractor():
    return TextExtractorPaddleOCR()
