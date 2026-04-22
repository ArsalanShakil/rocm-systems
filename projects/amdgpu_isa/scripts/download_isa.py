#!/usr/bin/env python3
"""Download the latest machine-readable ISA from GPUOpen and extract XML files."""

import io
import pathlib
import re
import zipfile

import urllib.request

URL = "https://gpuopen.com/download/machine-readable-isa/latest/"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    # Follow the redirect to discover the version, then download the archive.
    req = urllib.request.Request(URL, method="GET")
    with urllib.request.urlopen(req) as resp:
        final_url = resp.url
        archive_bytes = resp.read()

    # Extract version from the redirected URL (e.g. ".../machine_readable_isa_v42.zip")
    match = re.search(r"machine_readable_isa[_-]v?(\d[\w.]*)", final_url, re.IGNORECASE)
    if match:
        version = match.group(1)
    else:
        # Fallback: use the filename stem
        version = pathlib.PurePosixPath(final_url.split("?")[0]).stem

    print(f"Resolved URL : {final_url}")
    print(f"Version      : {version}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Write version file
    version_file = DATA_DIR / "version.txt"
    version_file.write_text(version + "\n", encoding="utf-8")
    print(f"Wrote        : {version_file}")

    # Extract XML files from the archive
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            name = pathlib.PurePosixPath(entry.filename).name
            if not name.lower().endswith(".xml"):
                continue
            dest = DATA_DIR / name
            dest.write_bytes(zf.read(entry.filename))
            print(f"Extracted    : {dest}")

    print("Done.")


if __name__ == "__main__":
    main()
