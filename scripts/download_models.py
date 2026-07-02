import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub.constants import HF_HUB_CACHE

from utils.log import print_error, print_good, print_info
from utils.models import MODELS, download, present


def main() -> int:
    print_info(f"HF cache: {HF_HUB_CACHE}")
    for model in MODELS:
        if present(model):  # cached & loadable ⇒ skip, zero network
            print_good(f"already present: {model.repo}")
            continue
        print_info(f"downloading {model.repo} (one-time, may be large) ...")
        try:
            download(model)
        except Exception as e:  # network / proxy / HF error — surfaced, never a silent hang
            print_error(f"failed to download {model.repo}: {e}")
            print_info("Check your network/proxy and re-run this script.")
            return 1
        if not present(model):  # dog-food the gate's oracle: confirm the files actually landed
            print_error(f"{model.repo}: still incomplete after download — remove its cache dir and retry.")
            return 1
        print_good(f"done: {model.repo}")
    print_good("all retriever models are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
