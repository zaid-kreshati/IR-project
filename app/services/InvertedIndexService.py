from collections import defaultdict
from joblib import dump, load

class InvertedIndexService:
    def __init__(self, tfidf_vectorizer=None, tfidf_matrix=None, tokenized_corpus=None):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.tokenized_corpus = tokenized_corpus
        self.inverted_index = {}
       
    def build_inverted_index_bm25(self):
        
        inverted_index = defaultdict(list)
        if self.tokenized_corpus:
            for doc_idx, tokens in enumerate(self.tokenized_corpus):
                for token in set(tokens):  # Avoid duplicates of the same term in one document
                    inverted_index[token].append(doc_idx)

        self.inverted_index = inverted_index
        return inverted_index

    def build_inverted_index_tfidf(self):
        inverted_index = defaultdict(list)
        # if self.tfidf_vectorizer and self.tfidf_matrix:
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        

        print("⏳ Building inverted index...")

        # Iterate through the TF-IDF matrix and map terms to document IDs
        for doc_idx, doc in enumerate(self.tfidf_matrix):
            for word_idx in doc.nonzero()[1]:  # Get non-zero elements (terms with non-zero weight)
                term = feature_names[word_idx]
                inverted_index[term].append(doc_idx)

        return inverted_index

    def get_inverted_index(self, collection_name):
        return self.inverted_index

    def save(self, collection_name):
        inverted_index_file = f"models/inverted_index_{collection_name}.joblib"
        dump(self.inverted_index, inverted_index_file)
        print(f"✅ Inverted index saved for collection: {collection_name}")

    def load(self, collection_name):
        print(collection_name)
        inverted_index_file = f"models/inverted_index_{collection_name}.joblib"
        self.inverted_index = load(inverted_index_file)
        print(f"🔄 Inverted index loaded for collection: {collection_name}")