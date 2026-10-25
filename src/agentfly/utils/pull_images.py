"""``agentfly pull-images``: pre-pull the enroot images a dataset needs, before training.

Container tasks (SWE-bench, R2E-Gym, SWE-smith) name a Docker image per task in ``image_id``.
A missing image is pulled the first time a rollout needs it, which is slow and puts heavy load
on the network mid-training, so pull them in advance::

    export ENROOT_IMAGES_PATH=/big/disk/enroot/images/r2e-gym-lite
    agentfly pull-images data/rlhf/os/r2e-gym-lite.json

Images are saved under the names enroot-py looks up at run time (``namanjain12/aiohttp_final:abc``
-> ``namanjain12+aiohttp_final+abc.sqsh``). Existing images are skipped, so an interrupted run
can simply be restarted; each pull writes to a ``.partial/`` folder first, so an interrupted
pull never leaves a truncated image behind.
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List


def image_file_name(image: str) -> str:
    """The ``.sqsh`` file name enroot-py uses for ``image`` (``repo/name:tag`` -> ``repo+name+tag.sqsh``)."""
    return image.replace("/", "+").replace(":", "+") + ".sqsh"


def read_images(paths: Iterable[str], field: str) -> List[str]:
    """Distinct values of ``field`` across the records of JSON (list) or JSONL datasets, in first-seen order."""
    images = {}
    for path in paths:
        with open(path) as f:
            if path.endswith(".jsonl"):
                records = (json.loads(line) for line in f if line.strip())
            else:
                records = json.load(f)
            for record in records:
                image = record.get(field)
                if image:
                    images.setdefault(image, None)
    return list(images)


def pull_one(image: str, directory: str, retries: int, registry: str = "") -> str:
    """Pull ``image`` into ``directory`` (from ``registry``, default Docker Hub); return "" on
    success or the last error message."""
    from enroot.image import pull_image

    partial_dir = os.path.join(directory, ".partial")
    os.makedirs(partial_dir, exist_ok=True)
    name = image_file_name(image)
    error = ""
    for attempt in range(1, retries + 1):
        partial = os.path.join(partial_dir, name)
        if os.path.exists(partial):
            os.remove(partial)
        try:
            pull_image(image, registry_host=registry, directory=partial_dir)
            os.replace(partial, os.path.join(directory, name))
            return ""
        except Exception as e:  # noqa: BLE001 - report and retry any pull failure
            error = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            if attempt < retries:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    return error


def main(argv=None):
    """Parse arguments and pull every missing image."""
    parser = argparse.ArgumentParser(
        prog="agentfly pull-images",
        description="Pre-pull the enroot images named by a dataset's image_id field.",
    )
    parser.add_argument("datasets", nargs="+", help="Dataset files (.json list or .jsonl).")
    parser.add_argument(
        "--to",
        default=None,
        help="Directory to save images in (default: $ENROOT_IMAGES_PATH, else ~/.cache/enroot/images). "
        "Training must run with ENROOT_IMAGES_PATH set to the same directory.",
    )
    parser.add_argument("--field", default="image_id", help="Record field holding the image (default: image_id).")
    parser.add_argument(
        "--registry",
        default="",
        help="Registry to pull from, e.g. ghcr.io (default: Docker Hub). Image files keep the same "
        "names, so training finds them either way.",
    )
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel pulls (default: 4).")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per image (default: 3).")
    parser.add_argument("--dry-run", action="store_true", help="Only report how many images are missing.")
    args = parser.parse_args(argv)

    if args.to:
        directory = args.to
    else:
        from enroot.config import ENROOT_IMAGES_PATH

        directory = ENROOT_IMAGES_PATH
    directory = os.path.abspath(os.path.expanduser(directory))

    # Task files hosted on Agent-One/AgentFly-Train (e.g. data/rlhf/os/r2e-gym-lite.json) are
    # downloaded when missing, as for training.
    from .download import ensure_training_files

    ensure_training_files(args.datasets)
    missing_files = [p for p in args.datasets if not os.path.exists(p)]
    if missing_files:
        print(f"Dataset file(s) not found: {', '.join(missing_files)}", file=sys.stderr)
        return 1
    images = read_images(args.datasets, args.field)
    missing = [im for im in images if not os.path.exists(os.path.join(directory, image_file_name(im)))]
    print(
        f"{len(images)} images in the dataset(s); {len(images) - len(missing)} already in {directory}; "
        f"{len(missing)} to pull."
    )
    failed_list = os.path.join(directory, "failed_pulls.txt")
    if args.dry_run:
        return 0
    if not missing:
        if os.path.exists(failed_list):  # from an earlier run whose failures are now pulled
            os.remove(failed_list)
        return 0
    if shutil.which("enroot") is None:
        print("enroot is not installed; see docs/start/installation.md.", file=sys.stderr)
        return 1

    os.makedirs(directory, exist_ok=True)
    shutil.rmtree(os.path.join(directory, ".partial"), ignore_errors=True)  # leftovers of interrupted pulls
    failed = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(pull_one, im, directory, args.retries, args.registry): im for im in missing}
        for done, future in enumerate(as_completed(futures), 1):
            image = futures[future]
            error = future.result()
            if error:
                failed[image] = error
            print(f"[{done}/{len(missing)}] {'FAILED' if error else 'ok'} {image}" + (f": {error}" if error else ""))
    shutil.rmtree(os.path.join(directory, ".partial"), ignore_errors=True)

    print(f"Pulled {len(missing) - len(failed)} of {len(missing)} images into {directory}.")
    if failed:
        with open(failed_list, "w") as f:
            f.write("\n".join(failed) + "\n")
        print(f"{len(failed)} failed (listed in {failed_list}); rerun the same command to retry them.")
        return 1
    if os.path.exists(failed_list):  # from an earlier run whose failures are now pulled
        os.remove(failed_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
