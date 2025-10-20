#!/usr/bin/env python3
import argparse
import time
import os
from glob import glob
from batch_utils import is_batch_completed, retrieve_save_batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve JUDGEMENT batch job results from OpenAI's Batch API."
    )
    parser.add_argument(
        "--batch_job_id",
        help="The Batch API job ID to poll for results.",
    )
    parser.add_argument(
        "--batch_job_id_folder",
        help="Folder containing batch job ID files to poll.",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30).",
    )
    args = parser.parse_args()

    if not args.batch_job_id and not args.batch_job_id_folder:
        parser.error(
            "At least one of --batch_job_id or --batch_job_id_folder must be provided."
        )

    if args.batch_job_id:
        while not is_batch_completed(args.batch_job_id):
            print("Batch job is still processing. Waiting...")
            time.sleep(args.poll_interval)
        retrieve_save_batch_results(args.batch_job_id)

    if args.batch_job_id_folder:
        batch_id_files = glob(
            os.path.join(args.batch_job_id_folder, "*_corrected_batch.jsonl.batch_id")
        )
        if not batch_id_files:
            print("No batch ID files found in the specified folder.")
            return
        for batch_id_file in batch_id_files:
            with open(batch_id_file, "r") as f:
                batch_id = f.read().strip()
            while not is_batch_completed(batch_id):
                print("Batch job is still processing. Waiting...")
                time.sleep(args.poll_interval)
            retrieve_save_batch_results(batch_id)


if __name__ == "__main__":
    main()
