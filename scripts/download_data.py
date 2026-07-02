import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import DATA, download, present
from utils.log import print_error, print_good, print_info


def main() -> int:
    for dataset in DATA:
        if present(dataset):  # already on disk ⇒ skip, zero network
            print_good(f"already present: {dataset.name}")
            continue
        print_info(f"downloading {dataset.name} (one-time, may be large) ...")
        try:
            download(dataset)
        except Exception as e:  # network / proxy / bucket error — surfaced, never a silent hang
            print_error(f"failed to download {dataset.name}: {e}")
            print_info("Check your network/proxy and re-run this script.")
            return 1
        if not present(dataset):  # dog-food the gate's oracle: confirm the files actually landed
            print_error(f"{dataset.name}: still incomplete after download — remove its assets dir and retry.")
            return 1
        print_good(f"done: {dataset.name}")
    print_good("all objathor data is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
