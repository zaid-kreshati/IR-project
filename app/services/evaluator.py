from collections import defaultdict
def compute_recall(relevant_docs, retrieved_docs):
    # Get the set of relevant documents for the given query
    relevant_retrieved = [doc_id for doc_id in retrieved_docs if doc_id in relevant_docs]


    # If there are no relevant documents, return 0
    if len(relevant_retrieved) == 0:
        return 0.0

        
    # Calculate recall as number of retrieved relevant docs divided by total relevant docs
    return len(relevant_retrieved) / len(relevant_docs)



def compute_precision(relevant_docs, retrieved_docs):
    if not retrieved_docs:
        return 0.0
    relevant_retrieved = [doc_id for doc_id in retrieved_docs if doc_id in relevant_docs]
    return len(relevant_retrieved) / len(retrieved_docs)


def compute_average_precision(relevant_docs, retrieved_docs):
    if not relevant_docs:
        return 0.0
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant_docs)


def compute_precision_at_k(relevant_docs, retrieved_docs, k=10):
    top_k = retrieved_docs[:k]
    relevant_top_k = [doc_id for doc_id in top_k if doc_id in relevant_docs]
    return len(relevant_top_k) / k



def compute_mrr(all_ranks):
    if not all_ranks:
        return 0.0
    return sum([1.0 / rank if rank > 0 else 0.0 for rank in all_ranks]) / len(all_ranks)
