import React from "react";

interface DatasetSelectorProps {
  datasets: string[];
  selectedDataset: string;
  onSelect: (value: string) => void;
}

const DatasetSelector: React.FC<DatasetSelectorProps> = ({
  datasets,
  selectedDataset,
  onSelect,
}) => {
  return (
    <div className="flex gap-4 flex-wrap">
      {datasets.map((dataset) => (
        <button
          key={dataset}
          onClick={() => onSelect(dataset)}
          className={`px-4 py-2 rounded border font-medium transition duration-200 ${
            selectedDataset === dataset
              ? "bg-black text-black border-black"
              : "bg-white text-gray-800 border-gray-300 hover:bg-black hover:text-black"
          }`}
        >
          {dataset}
        </button>
      ))}
    </div>
  );
};

export default DatasetSelector;
