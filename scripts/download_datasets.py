from pathlib import Path

import requests

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


if __name__ == "__main__":
    main()
