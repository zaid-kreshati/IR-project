from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import pathlib
from spellchecker import SpellChecker

# Import preprocessing
from app.services.preprocess_service import preprocess_text

spell = SpellChecker()

def correct_query(query: str) -> str:
    words = query.split()
    corrected_words = [
        spell.correction(w) if w not in spell and spell.correction(w) else w
        for w in words
    ]
    return " ".join(corrected_words)

def get_query_suggestions(query: str, limit: int = 3):
    if not query:
        return []

    # 🔍 Correct the query
    corrected_query = correct_query(query)
    print(f"Corrected query: {corrected_query}")

    # ✅ Preprocess and tokenize query
    query_tokens = preprocess_text(corrected_query)
    print(f"Preprocessed tokens: {query_tokens}")

    matching_queries = []

    # Load from JSON file
    with open("query_frequencies.json", "r", encoding="utf-8") as f:
        query_frequencies = json.load(f)

    for stored_query in query_frequencies:
        stored_tokens = preprocess_text(stored_query)

        # Count token overlaps
        match_score = sum(1 for word in query_tokens if word in stored_tokens)

        if match_score > 0:
            # Score based on token match and frequency
            score = (match_score / len(query_tokens)) * query_frequencies[stored_query]
            matching_queries.append((stored_query, score))

    # Sort by score descending
    matching_queries.sort(key=lambda x: x[1], reverse=True)

    # Return top N
    suggestions = [query for query, _ in matching_queries[:limit]]

    return {
        "suggesstions" : suggestions,
        "corrected_query" : corrected_query,
    }
