import gzip
import os
from huggingface_hub import hf_hub_download
import shutil
from ... import AGENT_CACHE_DIR

def download_tool_data(tool_name: str):
    """
    This is used to download tool-related data.
    """
    global AGENT_CACHE_DIR
    if tool_name == "asyncdense_retrieve":
        data_dir = os.path.join(AGENT_CACHE_DIR, "data", "search")
        corpus_file = os.path.join(data_dir, "wiki-18.jsonl")
        index_file = os.path.join(data_dir, "e5_Flat.index")
        if not os.path.exists(corpus_file):
            if not os.path.exists(os.path.join(data_dir, "wiki-18.jsonl.gz")):
                repo_id = "PeterJinGo/wiki-18-corpus"
                hf_hub_download(
                        repo_id=repo_id,
                        filename="wiki-18.jsonl.gz",
                        repo_type="dataset",
                        local_dir=data_dir,
                )
            # Unzip the file
            print(f"Unzipping {os.path.join(data_dir, 'wiki-18.jsonl.gz')}")
            gz_path = os.path.join(data_dir, "wiki-18.jsonl.gz")
            if os.path.exists(gz_path):
                with gzip.open(gz_path, 'rb') as f_in, open(corpus_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        if not os.path.exists(index_file):
            if not os.path.exists(os.path.join(data_dir, "part_aa")):
                repo_id = "PeterJinGo/wiki-18-e5-index"
                for file in ["part_aa", "part_ab"]:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file,  # e.g., "e5_Flat.index"
                        repo_type="dataset",
                        local_dir=data_dir,
                    )
            print(f"Concatenating {os.path.join(data_dir, 'part_*')} > {os.path.join(data_dir, 'e5_Flat.index')}")
            os.system(f"cat {os.path.join(data_dir, 'part_*')} > {os.path.join(data_dir, 'e5_Flat.index')}")

if __name__ == "__main__":
    download_tool_data("asyncdense_retrieve")
