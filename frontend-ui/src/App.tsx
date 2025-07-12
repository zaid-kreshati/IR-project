import React from "react";
import Home from "./pages/Home";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DocumentDetails from './pages/document_details';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/Basic/WEB/get-document" element={<DocumentDetails />} />
        

      </Routes>
    </Router>
  );
}

export default App;


