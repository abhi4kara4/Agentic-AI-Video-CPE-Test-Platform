import React from 'react';

function App() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>AI Video Test Platform</h1>
      <p>Platform is loading...</p>
      <div style={{ marginTop: '20px' }}>
        <h2>Status</h2>
        <ul>
          <li>Frontend: ✅ Running</li>
          <li>Backend API: Checking...</li>
          <li>WebSocket: Checking...</li>
        </ul>
      </div>
    </div>
  );
}

export default App;