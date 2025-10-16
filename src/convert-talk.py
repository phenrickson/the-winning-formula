import argparse
import logging
from pdf2image import convert_from_path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PDF pages to images")
    parser.add_argument(
        "--start", type=int, required=True, help="First page to convert"
    )
    parser.add_argument("--end", type=int, required=True, help="Last page to convert")
    return parser.parse_args()


def main():
    args = parse_args()
    start_page = args.start
    end_page = args.end

    logger.info(f"Converting pages {start_page} to {end_page}")

    # Higher dpi → sharper images
    pages = convert_from_path(
        "references/the-curse-of-knowledge.pdf",
        dpi=300,
        first_page=start_page,
        last_page=end_page,
    )  # default is 200–300

    for page_num, page in enumerate(pages, start=start_page):
        output_path = f"images/talk/page_{page_num}.png"
        page.save(output_path, "PNG")
        logger.info(f"Saved page {page_num} to {output_path}")


if __name__ == "__main__":
    main()
