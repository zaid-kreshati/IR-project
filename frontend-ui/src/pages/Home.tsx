import React, { useEffect, useState } from "react";
import DatasetSelector from "../components/DatasetSelector";
import ModelSelector from "../components/ModelSelector";
import axios from "axios";
import SearchBar from "../components/SearchBar";
import { useNavigate } from 'react-router-dom';

interface SearchResultItem {
  doc_id: number;
  score: number;
  body: string;
}

interface SearchResponse {
  results: {
    matched_count: number;
    execution_time: number;
    results: SearchResultItem[];
  };
}

interface SearchData {
  matchedCount: number;
  results: SearchResultItem[];
  executionTime: number;
}

const Home: React.FC = () => {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [document_id, setDocument_id] = useState<number>(0);
  const [message, setMessage] = useState("");
  const [useIndex, setUseIndex] = useState(true);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [executionTime, setExecutionTime] = useState<number>(0);
  const [searchData, setSearchData] = useState<SearchData | null>(null);
  const [topK, setTopK] = useState<number>(10);
  const [threshold, setThreshold] = useState<number>(0.1);
  const navigate = useNavigate();

  const modelOptions = ["TF-IDF", "EMBEDDING", "BM25", "HYBRID"];

  useEffect(() => {
    fetch("http://localhost:8000/Basic/WEB/get-datasets")
      .then((res) => res.json())
      .then((data) => {
        if (data && Array.isArray(data.datasets)) {
          setDatasets(data.datasets);
        } else {
          console.error("Unexpected response data", data);
        }
      })
      .catch((err) => console.error("Failed to fetch datasets:", err));
  }, []);

  const handleBuildVectorizer = async () => {
    if (!selectedDataset || !selectedModel) {
      alert("Please select both a dataset and a representation model.");
      return;
    }
    let endpoint = "";
    switch (selectedModel) {
      case "TF-IDF":
        endpoint = "/Basic/API/build-tfidf-model";
        break;
      case "EMBEDDING":
        endpoint = "/Basic/API/build-embedded";
        break;
      case "BM25":
        endpoint = "/Basic/API/build-bm25";
        break;
      case "HYBRID":
        alert("Hybrid model does not need to be built.");
        break;
      default:
        alert("Invalid model selected.");
        return;
    }

    try {
      const res = await axios.get(`http://localhost:8000${endpoint}`, {
        params: {
          collection_name: selectedDataset,
        },
      });
      setMessage(`${selectedModel} model built successfully!`);
    } catch (err: any) {
      console.error(err);
      setMessage("❌ Error building vectorizer. See console.");
    }
  };

  const handleSearch = async (query: string) => {
    if (!selectedDataset || !selectedModel) {
      alert("Please select both a dataset and a representation model.");
      return;
    }
    const collection_name = selectedDataset;

    try {
      let search_endpoint = "";
      switch (selectedModel) {
        case "TF-IDF":
          search_endpoint = useIndex
            ? "/Basic/API/search/tfidf-inverted"
            : "/Basic/API/search/tfidf";
          break;
        case "EMBEDDING":
          search_endpoint = useIndex
            ? "/Basic/API/search_embedded/vector_index"
            : "/Basic/API/search_embedded";
          break;
        case "BM25":
          search_endpoint = useIndex
            ? "/Basic/API/search/BM25/inverted_index"
            : "/Basic/API/search/BM25";
          break;
        case "HYBRID":
          search_endpoint = useIndex
            ? "/Basic/API/hybrid/rrf/with_Index"
            : "/Basic/API/hybrid/rrf";
          break;
        default:
          alert("Invalid model selected.");
          return;
      }
      const res = await axios.post<SearchResponse>(`http://localhost:8000${search_endpoint}`, {
        query,
        top_k: topK,
        collection_name,
        threshold: threshold,
      });

      const matchedCount = res.data.results.matched_count;
      const results = res.data.results.results;
      const executionTime = res.data.results.execution_time;

      setSearchData({ matchedCount, results, executionTime });
      setMessage(`Found ${res.data.results.matched_count} results for "${query}"`);
    } catch (err: any) {
      console.error(err);
      setMessage("❌ Error during search. See console.");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 p-8">
      <div className="max-w-7xl mx-auto bg-white rounded-xl shadow-lg p-8 flex space-x-12">
        {/* Left side */}
        <div className="w-1/4 space-y-8 bg-gradient-to-r from-blue-50 to-blue-100">
          <h2 className="text-3xl font-bold text-gray-800 mb-8 border-b pb-4">
            Information Retrieval System
          </h2>

          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <label className="block mb-2 text-lg font-semibold text-gray-700">Select Dataset</label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Choose a dataset</option>
                {datasets.map((dataset) => (
                  <option key={dataset} value={dataset}>
                    {dataset}
                  </option>
                ))}
              </select>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm">
              <label className="block mb-2 text-lg font-semibold text-gray-700">
                Select Representation Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Choose a model</option>
                {modelOptions.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm">
              <label className="block mb-2 text-lg font-semibold text-gray-700">
                Number of Results
              </label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                min="1"
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm">
              <label className="block mb-2 text-lg font-semibold text-gray-700">
                Similarity Threshold
              </label>
              <div className="flex items-center space-x-4">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="w-16 text-center font-medium text-gray-700">
                  {threshold.toFixed(2)}
                </span>
              </div>
            </div>

            <button
              onClick={handleBuildVectorizer}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
            >
              Build Vectorizer
            </button>

            <div className="bg-white p-4 rounded-lg shadow-sm">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useIndex}
                  onChange={(e) => setUseIndex(e.target.checked)}
                  className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="text-gray-700 font-medium">Search with index</span>
              </label>
            </div>

            {message && (
              <div className={`p-4 rounded-lg ${message.includes('❌') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                {message}
              </div>
            )}
          </div>
        </div>

        {/* Vertical blue line separator */}
        <div className="w-0.5 bg-blue-500"></div>

        {/* Right side: Search bar and results */}
        <div className="w-3/4 flex flex-col">
          <div className="mb-6">
            <SearchBar onSearch={handleSearch} />
          </div>

          <div className="flex-1 bg-blue-50 rounded-xl p-6 shadow-inner">
            <div className="overflow-auto max-h-[600px]">
              {searchData ? (
                <>
                  <div className="bg-white rounded-lg p-4 mb-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center">
                        <span className="text-gray-600">Matched Results:</span>
                        <span className="ml-2 font-semibold text-blue-600">{searchData.matchedCount}</span>
                      </div>
                      <div className="flex items-center">
                        <span className="text-gray-600">Execution Time:</span>
                        <span className="ml-2 font-semibold text-blue-600">{searchData.executionTime.toFixed(3)} seconds</span>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-6">
                    {searchData.results.map((item, index) => (
                      <div
                        key={item.doc_id}
                        className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-all duration-200"
                      >
                        <div className="flex justify-between items-center mb-4">
                          <div className="flex items-center">
                            <span className="text-xl font-semibold text-blue-600 mr-4">#{index + 1}</span>
                            
                            <a 
                              href={`/Basic/WEB/get-document?doc_id=${item.doc_id}&collection_name=${selectedDataset}`}>
                              View Details
                            </a>

                          </div>
                          <div className="bg-blue-50 px-4 py-2 rounded-full">
                            <span className="text-blue-700 font-semibold">
                              Score: {item.score.toFixed(3)}
                            </span>
                          </div>
                        </div>
                        {item.body && (
                          <div className="mt-3">
                            <h4 className="text-sm font-medium text-gray-500 mb-2">Document Content:</h4>
                            <p className="text-gray-700 leading-relaxed">
                              {item.body.length > 300 ? `${item.body.substring(0, 300)}...` : item.body}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-500 text-lg">No results yet. Try searching above!</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
