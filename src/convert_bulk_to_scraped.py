import os
import json
from typing import Dict
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def process_bulk_json(
    bulk_path: str = "./shared_data/data/bulk/arxiv-metadata-oai-snapshot.json",
    output_dir: str = "./shared_data/data/scraped",
    n_processes: int = 128,
):
    pool = ThreadPool(processes=n_processes)
    with open(bulk_path, "r") as f:
        for line in tqdm(f.read().split("\n")):
            pool.apply_async(write_doc_json, args=(line, output_dir))

    pool.close()
    pool.join()


def write_doc_json(line: str, output_dir: str = "./shared_data/data/scraped"):
    doc = json.loads(line)
    filename = os.path.join(output_dir, "{}.json".format(doc.get("id")))
    # print("writing {}".format(filename))
    with open(filename, "w+") as f:
        json.dump(doc, f)


if __name__ == "__main__":
    process_bulk_json()
