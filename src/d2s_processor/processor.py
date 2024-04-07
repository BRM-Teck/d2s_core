import logging

import numpy as np
from typing import List
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from d2s_preprocessor import InvoiceDataExtractor

_logger = logging.getLogger(__name__)


class ImageProcessor(object):
    def __init__(self, images: List[np.ndarray], images_name):
        if not all(isinstance(img, np.ndarray) for img in images):
            raise Exception("Toutes les images doivent être de type np.ndarray (cv2)")
        if len(images) != len(images_name):
            raise Exception("")
        self.images = images
        self.images_name = images_name

    def to_dict(self):
        """Process images to extract structures date from it.
        ================================================================
        """
        results = []
        _logger.info("Start Process")
        with logging_redirect_tqdm(loggers=[_logger]):
            for image, filename in tqdm(
                zip(self.images, self.images_name), desc="Processing items"
            ):
                _logger.info(f" Item Name : {filename}")
                data_extractor = InvoiceDataExtractor(image, filename=filename)
                result = data_extractor.result()
                result.replace(np.nan, None, inplace=True)
                results.append(result.to_dict())

        return results


def _main():
    from utils.read_image import image_from_file
    from settings import BASE_DIR
    import pprint

    image_dir = BASE_DIR / "data/test/images"
    img_file = list(image_dir.glob("*.png"))[0]
    image = image_from_file(img_file)
    image_processor = ImageProcessor(images=[image], images_name=[img_file.stem])
    data = image_processor.to_dict()
    _logger.info(pprint.pprint(data))


if __name__ == "__main__":
    _main()
