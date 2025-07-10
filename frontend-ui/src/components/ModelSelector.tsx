import React from "react";

interface ModelSelectorProps {
  models: string[];
  selectedModel: string;
  onSelect: (value: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ models, selectedModel, onSelect }) => {
  return (
    <div className="flex gap-4 flex-wrap">
      {models.map((model) => (
        <button
          key={model}
          onClick={() => onSelect(model)}
          className={`px-4 py-2 rounded border font-semibold transition ${
            selectedModel === model
              ? "bg-blue-600 text-black border-blue-600"
              : "bg-white text-gray-800 border-gray-300 hover:bg-blue-100"
          }`}
        >
          {model}
        </button>
      ))}
    </div>
  );
};

export default ModelSelector;
