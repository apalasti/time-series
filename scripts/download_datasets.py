from pathlib import Path

import requests
import torch

from src.datasets import SyntheticDataset

TIMEDRL_REPO_URL = "blacksnail789521/TimeDRL"
LOCAL_OUT_DIR = Path(__file__).parent.parent / "datasets"


def download_github_folder(repo_url: str, folder_path: str, local_dir):
    local_dir = Path(local_dir)
    api_url = f"https://api.github.com/repos/{repo_url}/contents/{folder_path}"
    response = requests.get(api_url)
    response.raise_for_status()
    contents = response.json()

    for item in contents:
        if item["type"] == "file":
            download_url = item["download_url"]
            file_path = local_dir / str(item["name"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading file: {item['name']} to {file_path}")
            file_response = requests.get(download_url)
            file_response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(file_response.content)
        elif item["type"] == "dir":
            sub_folder_path = item["path"]
            sub_local_dir = local_dir / str(item["name"])
            download_github_folder(repo_url, sub_folder_path, sub_local_dir)


def main():
    print(f"Downloading datasets from: github.com/{TIMEDRL_REPO_URL}")
    download_github_folder(TIMEDRL_REPO_URL, "dataset", LOCAL_OUT_DIR)
    print("Successfully downloaded all datasets!")

    print("Creating snythetic dataset ...")
    out_dir = LOCAL_OUT_DIR / "classification/Synthetic"
    out_dir.mkdir(exist_ok=True, parents=True)
    synthetic = SyntheticDataset(
        n_samples=1400, seq_length=128, n_channels=4, n_classes=4, seed=2954376427
    )
    for name, range in [("train", (0, 1000)), ("val", (1000, 1200)), ("test", (1200, 1400))]:
        torch.save({
            "interference_times": synthetic.interference_times,
            "samples": synthetic.samples[range[0] : range[1]],
            "labels": synthetic.labels[range[0] : range[1]],
        }, out_dir / f"{name}.pt")
    print(f"Saved synthetic dataset to: {out_dir}")
    


if __name__ == "__main__":
    main()
