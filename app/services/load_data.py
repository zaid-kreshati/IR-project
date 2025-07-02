from ir_datasets import load
from pymongo import MongoClient
from pymongo.errors import BulkWriteError


def insert_documents(dataset_name: str ,batch_size: int = 1000):
    name=dataset_name.replace("/", "-").replace("\\", "_").strip()

    client = MongoClient("mongodb://localhost:27017/")
    db = client["ir_project"]
    collection = db[name]

    # Ensure index for fast deduplication
    collection.create_index("doc_id", unique=True)

    dataset = load("nano-beir/arguana")
    count_inserted = 0
    batch = []

    print(f"🚀 Starting to load dataset: {name}")

    for i, doc in enumerate(dataset.docs_iter()):
        doc_id = doc.doc_id
        text = doc.text

        batch.append({
            "doc_id": doc_id,
            "body": text
        })

        if len(batch) == batch_size:
            try:
                result = collection.insert_many(batch, ordered=False)
                count_inserted += len(result.inserted_ids)
                print(f"✅ Inserted {count_inserted} documents so far...")
            except BulkWriteError as bwe:
                # Skip duplicates gracefully
                num_errors = len(bwe.details.get("writeErrors", []))
                count_inserted += batch_size - num_errors
                print(f"⚠️ Skipped {num_errors} duplicates, inserted {batch_size - num_errors}")
            batch = []

    # Final batch
    if batch:
        try:
            result = collection.insert_many(batch, ordered=False)
            count_inserted += len(result.inserted_ids)
            print(f"✅ Final batch inserted. Total: {count_inserted}")
        except BulkWriteError as bwe:
            num_errors = len(bwe.details.get("writeErrors", []))
            count_inserted += len(batch) - num_errors
            print(f"⚠️ Final batch: Skipped {num_errors} duplicates, inserted {len(batch) - num_errors}")

    return f"🎉 Done! Inserted {count_inserted} new documents into '{dataset_name}' collection."
