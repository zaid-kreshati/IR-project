from collections import defaultdict

def average_precision(ranked_docs, relevant_docs):
    if not relevant_docs:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant_docs) if relevant_docs else 0.0



def compute_recall(retrieved_doc, qrels_df, query_id):
    # Get the set of relevant documents for the given query
    relevant_docs = set(qrels_df[qrels_df["query_id"] == query_id]["doc_id"])

    # If there are no relevant documents, return 0
    if len(relevant_docs) == 0:
        return 0.0

    print(f"relevant_docs: {len(relevant_docs)}")
        
    # Calculate recall as number of retrieved relevant docs divided by total relevant docs
    return retrieved_doc / len(relevant_docs)

def compute_map(retrieved_doc_ids, qrels_df, query_id):
   
    relevant_docs = set(qrels_df[qrels_df["query_id"] == query_id]["doc_id"])

    if not relevant_docs:
        return 0.0  # No ground truth for this query

    num_relevant = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)

    avg_precision = precision_sum / len(relevant_docs)
    return avg_precision
