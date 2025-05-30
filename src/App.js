import React from 'react';
import './App.css';
import FSRCNNTest from './FSRCNNTest';
import FSRCNNVideo from './FSRCNNVideo';
import FSRCNNVideoCompare from './FSRCNNVideoCompare';
function App() {
  return (
    <div className="App">
      {/* <FSRCNNTest /> */}
      {/* <FSRCNNVideo /> */}
      <FSRCNNVideoCompare />
    </div>
  );
}

export default App;