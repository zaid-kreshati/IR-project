import React, { useState } from "react";
import { FaSearch } from "react-icons/fa";

interface SearchBarProps {
  onSearch: (
    query: string,
    selectedDataset: string,
    selectedModel: string,
    useIndex: boolean
  ) => void;
}

interface Props {
  onSearch: (
    query: string,
    selectedDataset: string,
    selectedModel: string,
    useIndex: boolean
  ) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
  const [query, setQuery] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("default");
  const [selectedModel, setSelectedModel] = useState("BM25");
  const [useIndex, setUseIndex] = useState(true);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim(), selectedDataset, selectedModel, useIndex);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <FaSearch className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question or type your query..."
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-full shadow-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </form>
    </div>
  );
};

export default SearchBar;
