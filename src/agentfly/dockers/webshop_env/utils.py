import bisect
import hashlib
import logging
import random
from os.path import dirname, abspath, join
from bs4 import BeautifulSoup

BASE_DIR = dirname(abspath(__file__))

# these paths must be initialized with init_basedir
DEFAULT_ATTR_PATH = ""  # join(BASE_DIR, 'data/items_ins_v2_1000.json')
DEFAULT_FILE_PATH = ""  # join(BASE_DIR, 'data/items_shuffle_1000.json')

FEAT_CONV = join(BASE_DIR, "data/feat_conv.pt")  # NOT USED in RAGEN version
FEAT_IDS = join(BASE_DIR, "data/feat_ids.pt")  # NOT USED in RAGEN version

HUMAN_ATTR_PATH = join(BASE_DIR, "data/items_human_ins.json")


def init_basedir(dataset: str = "small") -> None:
    """Initialize the base directory for the package and update related paths.

    Args:
        dataset (str): Dataset size to use - either 'small' or 'full'. Default is 'small'.
    """
    global \
        BASE_DIR, \
        DEFAULT_ATTR_PATH, \
        DEFAULT_FILE_PATH, \
        FEAT_CONV, \
        FEAT_IDS, \
        HUMAN_ATTR_PATH

    if dataset not in ["small", "full"]:
        raise ValueError("dataset must be either 'small' or 'full'")

    # Set file paths based on dataset size
    if dataset == "small":
        attr_file = "items_ins_v2_1000.json"
        items_file = "items_shuffle_1000.json"
    else:  # full
        attr_file = "items_ins_v2.json"
        items_file = "items_shuffle.json"

    DEFAULT_ATTR_PATH = join(BASE_DIR, f"data/{dataset}/{attr_file}")
    DEFAULT_FILE_PATH = join(BASE_DIR, f"data/{dataset}/{items_file}")

    FEAT_CONV = join(BASE_DIR, "data/feat_conv.pt")  # NOT USED in RAGEN version
    FEAT_IDS = join(BASE_DIR, "data/feat_ids.pt")  # NOT USED in RAGEN version

    HUMAN_ATTR_PATH = join(BASE_DIR, "data/items_human_ins.json")


def get_base_dir() -> str:
    """Get the base directory for the package.

    Returns:
        str: The base directory path.
    """
    return BASE_DIR


def get_attr_path() -> str:
    return DEFAULT_ATTR_PATH


def get_file_path() -> str:
    return DEFAULT_FILE_PATH


def get_human_attr_path() -> str:
    return HUMAN_ATTR_PATH


def get_feat_conv_path() -> str:
    return FEAT_CONV


def get_feat_ids_path() -> str:
    return FEAT_IDS


def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then
    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx


def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(user_log_dir / f"{session_id}.jsonl", mode="w")
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()


def html_to_markdown(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    markdown = ""

    # Convert headings
    for i in range(1, 7):
        for heading in soup.find_all(f"h{i}"):
            markdown += f"{'#' * i} {heading.get_text().strip()}\n\n"

    # Convert paragraphs
    for p in soup.find_all("p"):
        markdown += f"{p.get_text().strip()}\n\n"

    # Convert links
    for a in soup.find_all("a"):
        markdown += f"[{a.get_text().strip()}]({a.get('href', '')})\n\n"

    # Convert inputs
    for inp in soup.find_all("input"):
        markdown += f"[{inp.get('placeholder', 'Input')}]\n\n"

    # Convert lists
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li"):
            markdown += f"* {li.get_text().strip()}\n"
        markdown += "\n"

    for ol in soup.find_all("ol"):
        for i, li in enumerate(ol.find_all("li")):
            markdown += f"{i + 1}. {li.get_text().strip()}\n"
        markdown += "\n"

    return markdown.strip()
