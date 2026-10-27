"""S3 sync utility — upload/download project artifacts to persistent storage.

Run this at the end of EVERY phase so nothing is lost if the EC2 instance dies.

Usage:
    python s3_sync.py upload                         # push all managed dirs
    python s3_sync.py download                       # pull all managed dirs
    python s3_sync.py upload --dir outputs           # push a single directory
    python s3_sync.py upload --file data/train_data.jsonl   # push one file
    python s3_sync.py download --dir finetuning_adapter
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import boto3
from botocore.exceptions import ClientError

import config


def _client():
    return boto3.client("s3", region_name=config.BEDROCK_REGION)


def _s3_key(local_path: str) -> str:
    """Map a local relative path to an S3 key under the version prefix."""
    rel = os.path.relpath(local_path, config.PROJECT_ROOT)
    rel = rel.replace(os.sep, "/")
    return f"{config.S3_PREFIX}/{rel}"


def _etag_matches(s3, key: str, local_path: str) -> bool:
    """Return True if the S3 object exists and its MD5 ETag matches the local file.

    Only valid for single-part uploads (our files are small), which produce a
    plain MD5 ETag. Multipart ETags contain a '-' and are treated as a mismatch.
    """
    try:
        head = s3.head_object(Bucket=config.S3_BUCKET, Key=key)
    except ClientError:
        return False
    etag = head["ETag"].strip('"')
    if "-" in etag:
        return False
    md5 = hashlib.md5()
    with open(local_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest() == etag


def _iter_local_files(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            yield os.path.join(dirpath, name)


def _human(size: int) -> str:
    val = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024 or unit == "GB":
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} GB"


def upload(dirs: list[str], single_file: str | None) -> None:
    s3 = _client()
    targets: list[str] = []
    if single_file:
        if not os.path.exists(single_file):
            sys.exit(f"File not found: {single_file}")
        targets = [single_file]
    else:
        for d in dirs:
            if os.path.isdir(d):
                targets.extend(_iter_local_files(d))

    uploaded = skipped = 0
    total_bytes = 0
    for path in targets:
        key = _s3_key(path)
        if _etag_matches(s3, key, path):
            skipped += 1
            continue
        s3.upload_file(path, config.S3_BUCKET, key)
        uploaded += 1
        total_bytes += os.path.getsize(path)

    print(
        f"Synced {uploaded} file(s) ({_human(total_bytes)}) to "
        f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}/  "
        f"[{skipped} unchanged, skipped]"
    )


def download(dirs: list[str], single_file: str | None) -> None:
    s3 = _client()
    downloaded = 0
    total_bytes = 0

    if single_file:
        prefixes = [f"{config.S3_PREFIX}/{single_file.replace(os.sep, '/')}"]
    else:
        prefixes = [f"{config.S3_PREFIX}/{d.replace(os.sep, '/')}/" for d in dirs]

    paginator = s3.get_paginator("list_objects_v2")
    for prefix in prefixes:
        for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(config.S3_PREFIX) + 1:]  # strip "v1/"
                local_path = os.path.join(config.PROJECT_ROOT, rel)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if os.path.exists(local_path) and _etag_matches(s3, key, local_path):
                    continue
                s3.download_file(config.S3_BUCKET, key, local_path)
                downloaded += 1
                total_bytes += obj["Size"]

    print(
        f"Downloaded {downloaded} file(s) ({_human(total_bytes)}) from "
        f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}/"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync artifacts to/from S3.")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--dir", help="Limit to a single managed directory.")
    parser.add_argument("--file", help="Limit to a single file (upload only).")
    args = parser.parse_args()

    dirs = [args.dir] if args.dir else config.S3_SYNC_DIRS

    if args.action == "upload":
        upload(dirs, args.file)
    else:
        download(dirs, args.file)


if __name__ == "__main__":
    main()
