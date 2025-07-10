import React, { useState } from "react";
import axios from "axios";

const representationModels = ["TF-IDF", "EMBEDDING", "BM25", "HYBRID"];

interface SearchPanelProps {
  selectedDataset: string;
}

const SearchPanel: React.FC<SearchPanelProps> = ({ selectedDataset }) => {
  const [query, setQuery] = useState("");
  const [model, setModel] = useState(representationModels[0]);
  const [useIndex, setUseIndex] = useState(true);
  const [results, setResults] = useState<any[]>([]);

  const handleSearch = async (query: string, selectedDataset: string, selectedModel: string) => {
    if (!selectedDataset) {
      alert("Please select a dataset first.");
      return;
    }
    if (!query.trim()) {
      alert("Please enter a search query.");
      return;
    }

    try {
      // Construct endpoint based on model and index usage
      // Assuming your backend has routes like:
      // /search/tfidf, /search/tfidf_noindex, etc.
      const baseEndpoint = "/search";
      const modelLower = model.toLowerCase().replace("-", "");
      const indexPart = useIndex ? "" : "_noindex";

      const endpoint = `${baseEndpoint}/${modelLower}${indexPart}`;

      const res = await axios.get(`http://localhost:8000${endpoint}`, {
        params: {
          collection_name: selectedDataset,
          query: query,
        },
      });

    //   setResults(res.data.results);
    } catch (error) {
      console.error("Search error:", error);
      alert("Search failed, check console for details.");
    }
  };

  return (
    <div className="p-4 border rounded shadow-md max-w-xl mx-auto">
      <div className="mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter search query"
          className="w-full p-2 border rounded"
        />
      </div>

      <div className="mb-4 flex space-x-2">
        {representationModels.map((m) => (
          <button
            key={m}
            onClick={() => setModel(m)}
            className={`px-4 py-2 rounded border ${
              model === m ? "bg-black text-white" : "bg-white text-black"
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="mb-4 flex items-center space-x-2">
        <input
          type="checkbox"
          checked={useIndex}
          onChange={(e) => setUseIndex(e.target.checked)}
          id="use-index-checkbox"
        />
        <label htmlFor="use-index-checkbox">Search with index</label>
      </div>

      <button
        onClick={() => handleSearch(query, selectedDataset, model)}
        className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
      >
        Search
      </button>

      <div className="mt-6">
        <h3 className="font-semibold mb-2">Results:</h3>
        {results.length === 0 && <p>No results yet.</p>}
        <ul className="list-disc list-inside">
          {results.map((r, i) => (
            <li key={i}>{JSON.stringify(r)}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default SearchPanel;
