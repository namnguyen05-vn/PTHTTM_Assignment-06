"""Download only the two required public Kaggle CSV files."""
import sys
import time
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT, DATA
from src.io import sha256, write_json
import requests

SOURCES = {
    "retailrocket": ("retailrocket/ecommerce-dataset/events.csv", "events.csv"),
    "sp500": ("camnugent/sandp500/all_stocks_5yr.csv", "all_stocks_5yr.csv"),
}

def fetch(name, reference, member):
    folder = DATA / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (name + ".download")
    url = "https://www.kaggle.com/api/v1/datasets/download/" + reference
    if not destination.exists():
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise RuntimeError("Kaggle returned HTML; download the CSV manually.")
                    with destination.with_suffix(".part").open("wb") as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk)
                destination.with_suffix(".part").replace(destination)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
    if zipfile.is_zipfile(destination):
        with zipfile.ZipFile(destination) as archive:
            matches = [n for n in archive.namelist() if Path(n).name == member]
            if len(matches) != 1:
                raise ValueError((member, archive.namelist()))
            with archive.open(matches[0]) as stream:
                header = stream.readline().decode("utf-8-sig").strip()
    else:
        with destination.open("r", encoding="utf-8-sig") as stream:
            header = stream.readline().strip()
    required = "timestamp,visitorid,event,itemid,transactionid" if name == "retailrocket" else "date,open,high,low,close,volume,Name"
    if header != required:
        raise ValueError(("Unexpected CSV header", name, header))
    record = dict(dataset=name, url=url, file=destination.name, csv_member=member,
                  bytes=destination.stat().st_size, sha256=sha256(destination))
    print(name, record["bytes"], "bytes", flush=True)
    return record

if __name__ == "__main__":
    write_json(ROOT/"results/data_downloads.json",
               [fetch(name, *source) for name, source in SOURCES.items()])
