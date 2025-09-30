import os
import time
from datasets import load_dataset, Dataset
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
from config import get_ds_config
from tqdm import tqdm


def tokenize_chunk_remote(chunk, ds_config):
    from dataset import build_tokenizer
    tokenizer = build_tokenizer(ds_config)

    #  prompts
    prompts = [
        f"{ds_config['Q_role']}: {ex[ds_config['Q_role']]}\n{ds_config['A_role']}: {ex[ds_config['A_role']]}"
        for ex in chunk
    ]
    # tokenize
    tokenized = tokenizer(
        prompts,
        max_length=ds_config["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = tokenized["input_ids"]
    masks = tokenized["attention_mask"]

    results = []
    for i in range(len(prompts)):
        results.append({
            "input_ids": inputs[i],
            "attention_mask": masks[i],
            "labels": inputs[i].clone()
        })
    return results


def run_tokenization(client, ds_config, job_id):
    print(f"\n=== Running tokenization with {job_id} job(s) ===")

    print("Loading dataset...")
    t0 = time.time()
    dataset = load_dataset(ds_config["dataset_name"], split="train")
    data = list(tqdm(dataset, desc="Loading samples"))
    t1 = time.time()
    print(f"Dataset loaded. Time: {t1 - t0:.2f} sec")

    #  chunk size
    chunk_size = ds_config.get("chunk_size", 10000)
    print("Splitting into chunks...")
    chunks = [
        data[i : i + chunk_size] for i in tqdm(
            range(0, len(data), chunk_size), desc="Splitting chunks"
        )
    ]
    t2 = time.time()
    print(f"Chunks created: {len(chunks)}. Time: {t2 - t1:.2f} sec")

    print("Scattering config...")
    scattered_config = client.scatter(ds_config, broadcast=True)
    t3 = time.time()
    print(f"Config scattered. Time: {t3 - t2:.2f} sec")

    print("Submitting Dask jobs...")
    start_token = time.time()
    futures = []
    for chunk in tqdm(chunks, desc="Submitting jobs"):
        future = client.submit(tokenize_chunk_remote, chunk, scattered_config)
        futures.append(future)

    results = []
    completed = 0
    total = len(futures)
    samples_per_chunk = chunk_size
    last_log = start_token

    print("Running Dask tasks...")
    for future in tqdm(as_completed(futures), total=total, desc="Processing chunks"):
        batch = future.result()
        results.append(batch)
        completed = completed + 1

        now = time.time()
        # speed
        if now - last_log >= 5 or completed == total:
            elapsed = now - start_token
            speed = (completed * samples_per_chunk) / elapsed if elapsed > 0 else 0
            print(f"[{completed}/{total}] chunks completed, approx {speed:.2f} samples/sec")
            last_log = now

    t4 = time.time()
    print(f"All jobs completed. Time: {t4 - t3:.2f} sec")

    print("Gathering and saving results...")
    flat = [
        item for chunk in tqdm(results, desc="Flattening results") for item in chunk
    ]
    processed_dataset = Dataset.from_list(flat)
    save_dir = f"tokenized_output_{job_id}_jobs"
    print(f"Saving dataset to {save_dir}...")
    processed_dataset.save_to_disk(save_dir)
    t5 = time.time()
    print(f"Dataset saved to {save_dir}. Time: {t5 - t4:.2f} sec")

    total_time = t5 - t0
    total_samples = len(flat)
    if total_time > 0:
        samples_per_sec = total_samples / total_time
    else:
        samples_per_sec = 0

    token_time = t4 - start_token
    if token_time > 0:
        token_throughput = total_samples / token_time
    else:
        token_throughput = 0

    print("\n=== Tokenization Summary ===")
    print(f"Jobs: {job_id}")
    print(f"Samples: {total_samples}")
    print(f"Total Time: {total_time:.2f} sec")
    print(f"Overall Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"Tokenization Phase Time: {token_time:.2f} sec")
    print(f"Tokenization Throughput: {token_throughput:.2f} samples/sec")
    print("=============================")


def main():
    ds_config = get_ds_config()

    for job_count in [4]:
        cluster = SLURMCluster(
            queue="courses",
            account="csye7105.202530",
            cores=28,
            memory="128GB",
            walltime="02:00:00",
            interface="internal"
        )
        cluster.scale(jobs=job_count)
        client = Client(cluster)
        client.wait_for_workers(job_count)


        # node info
        info = client.scheduler_info()
        print("[INFO] Dask Scheduler and Workers:")
        print(f"  Scheduler: {client.scheduler.address}")
        for worker_addr, worker_info in info["workers"].items():
            host = worker_info.get("host", "N/A")
            nthreads = worker_info.get("nthreads", "N/A")
            mem = worker_info.get("memory_limit", "N/A")
            print(f"  Worker {worker_addr} -> host={host}, threads={nthreads}, memory_limit={mem/1e9:.2f} GB")

        print(f"\n--- Dask Dashboard (jobs={job_count}): {client.dashboard_link} ---")
        run_tokenization(client, ds_config, job_id=job_count)

        client.close()
        cluster.close()


if __name__ == "__main__":
    main() 