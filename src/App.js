import React from 'react';
import './App.css';
import FSRCNNTest from './FSRCNNTest';
import FSRCNNVideo from './FSRCNNVideo';
import FSRCNNVideoCompare from './FSRCNNVideoCompare';
import VideoSRCompareSync from './VideoSRCompareSync';

function App() {
  return (
    <div className="App">
      {/* <FSRCNNTest /> */}
      {/* <FSRCNNVideo /> */}
      <FSRCNNVideoCompare />
      {/* <VideoSRCompareSync /> */}
    </div>
  );
}

export default App;