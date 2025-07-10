export const fetchDatasets = async (): Promise<string[]> => {
  const res = await fetch("http://localhost:8000/Basic/WEB/get-datasets");
  console.log("fetching datasets");
  console.log(res.status);
  const data = await res.json();
  return data.datasets;
};
