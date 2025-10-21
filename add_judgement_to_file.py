#!/usr/bin/env python3
"""Merge judgement batch results back into corrected transcription JSON files."""

import argparse
import json
import os
import glob


def load_jsonl_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Combine batch API answers with original JSON segments."
    )
    parser.add_argument("--original_file", help="Original JSON file with segments.")
    parser.add_argument(
        "--folder_original_files",
    help="Folder containing corrected JSON files with segments and the batch jobs with the judgement.",
    )
    args = parser.parse_args()

    if args.folder_original_files:
        # Process all JSON files in the specified folder
        json_transcriptions = glob.glob(
            os.path.join(args.folder_original_files, "*_corrected.json")
        )
    elif args.original_file:
        # Process a single JSON file
        json_transcriptions = [args.original_file]
    else:
        raise ValueError(
            "Either --original_file or --folder_original_files must be provided."
        )

    for json_file in json_transcriptions:
        print(f"Processing file: {json_file}")
        # Load the submitted batch jobs and the batch results.
        # Be careful with the name
        batch_input = json_file.replace("_corrected.json", "_corrected_batch.jsonl")
        batch_results = json_file.replace(
            "_corrected.json", "_corrected_batch_answer.jsonl"
        )
        batch_input = load_jsonl_file(batch_input)
        batch_results = load_jsonl_file(batch_results)

        # Create a mapping from custom_id to the corrected answer from the batch results.
        results_map = {
            job["custom_id"]: job["response"]["body"]["choices"][0]["logprobs"][
                "content"
            ][0]
            for job in batch_results
        }

        # Update each submitted job with its judgement answer.
        for job in batch_input:
            custom_id = job["custom_id"]
            # Drop bytes from each logprobs
            result = results_map[custom_id]
            result_ = []
            for token in result["top_logprobs"]:
                del token["bytes"]
                result_.append(token)
            del result["bytes"]
            result["top_logprobs"] = result_
            job["judgement"] = result

        # Load the original JSON file.
        original_data = load_json_file(json_file)
        segments = original_data["segments"]

        # For each segment, add the judgement from the corresponding batch job.
        for i, seg_id in enumerate(segments.keys()):
            segments[seg_id]["judgement"] = batch_input[i]["judgement"]

        # Determine output file name; do not overwrite the original.
        base, ext = os.path.splitext(json_file)
        output_file = f"{base}_judged{ext}"

        # Save the updated JSON.
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(original_data, f, indent=4, ensure_ascii=False)
        print(f"Combined output saved to {output_file}")


if __name__ == "__main__":
    main()
