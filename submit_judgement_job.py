#!/usr/bin/env python3
import argparse
import os
from glob import glob
from batch_utils import submit_batch_file, create_batch


def main():
    parser = argparse.ArgumentParser(
        description="Submit batch jobs for JUDGEMENT to OpenAI's Batch API."
    )
    parser.add_argument(
        "--jsonl_folder",
        required=True,
        help="Folder containing JSONL files for judgement submission.",
    )
    parser.add_argument(
        "--glob_pattern",
        default="*_corrected_batch.jsonl",
        help="Glob pattern to match JSONL files (default: *_corrected_batch.jsonl).",
    )
    args = parser.parse_args()

    jsonl_files = glob(os.path.join(args.jsonl_folder, args.glob_pattern))
    if not jsonl_files:
        print("No JSONL files found in the specified folder.")
        return

    for jsonl_file in jsonl_files:
        print(f"Submitting JUDGEMENT batch job for {jsonl_file}...")
        file_response = submit_batch_file(jsonl_file)
        batch_response = create_batch(file_response.id, jsonl_file)
        # Write batch job ID to a file
        batch_id_file = f"{jsonl_file}.batch_id"
        with open(batch_id_file, "w") as f:
            f.write(batch_response.id)
        print(f"Batch job ID saved to {batch_id_file}")


if __name__ == "__main__":
    main()
