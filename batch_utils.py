from openai import OpenAI  # type: ignore

# Expected statuses indicating a successful job (adjust if needed)
SUCCESS_STATUSES = {"completed"}

client = OpenAI()


def is_batch_completed(batch_job_id):
    batch = client.batches.retrieve(batch_job_id)
    print(batch)
    pending = batch.status not in SUCCESS_STATUSES
    if pending:
        print("Batch job is still pending. Current status: " + batch.status)
        return False
    else:
        print("All jobs are completed.")
        return True


def submit_batch_file(file_path):
    print(f"Submitting file {file_path} to OpenAI for batch processing...")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="batch")
    print("File submitted. Response:")
    print(response)
    return response


def create_batch(file_id, file_path):
    response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": file_path},
    )
    print("Batch created. Response:")
    print(response)
    return response


def retrieve_save_batch_results(batch_id):
    batch = client.batches.retrieve(batch_id)
    file_response = client.files.content(batch.output_file_id)
    outfile_name = batch.metadata["description"].replace(".jsonl", "_answer.jsonl")
    with open(outfile_name, "wb") as f:
        f.write(file_response.content)
    print(f"Batch results saved to {outfile_name}")
