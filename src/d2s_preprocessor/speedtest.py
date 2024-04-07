import logging
import pprint

from pathlib import Path

from preprocessor import InvoiceDataExtractor

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "test" / "outputs"


def main():
    output = str(OUTPUT_DIR.resolve())
    img_file = str(
        (OUTPUT_DIR.parent / "images" / "FAC FINE EXPORT 3009-1.png").resolve()
    )

    data_extractor = InvoiceDataExtractor(
        img_file, output_dir=output, root_dir=Path("..").resolve(), logger=logger
    )
    return data_extractor.result()


if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    pprint.pprint(main())
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
