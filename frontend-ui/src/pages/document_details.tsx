import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

const DocumentDetails = () => {
  const [searchParams] = useSearchParams();
  const [document, setDocument] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const docId = searchParams.get("doc_id");
  const collectionName = searchParams.get("collection_name");

  useEffect(() => {
    const fetchDocument = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `http://localhost:8000/Basic/WEB/get-document?doc_id=${docId}&collection_name=${collectionName}`
        );

        const data = await response.json();

        if (response.ok) {
          if (data.document) {
            setDocument(data.document);
          } else {
            setError("Document not found.");
          }
        } else {
          setError(data.error || "Something went wrong.");
        }
      } catch (err) {
        setError("Failed to fetch document.");
      } finally {
        setLoading(false);
      }
    };

    if (docId && collectionName) {
      fetchDocument();
    } else {
      setError("Missing document ID or collection name.");
      setLoading(false);
    }
  }, [docId, collectionName]);

  if (loading) return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="flex justify-center items-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        <span className="ml-3 text-gray-600">Loading document...</span>
      </div>
    </div>
  );

  if (error) return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-red-700">Error: {error}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h1 className="text-2xl font-bold text-gray-900">Document Details</h1>
          </div>
          {document && (
            <div className="p-6 space-y-6">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Document ID</label>
                    <div className="mt-1">
                      <code className="block w-full px-3 py-2 rounded-md border border-gray-300 bg-white text-sm font-mono">
                        {(document as { doc_id: string }).doc_id}
                      </code>
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h2 className="text-lg font-medium text-gray-900 mb-3">Document Body</h2>
                <div className="bg-gray-50 rounded-lg p-4 overflow-auto">
                  <pre className="text-sm font-mono text-gray-700 whitespace-pre-wrap">
                    {(document as { body: string }).body}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentDetails;
