import React from 'react';
import Search from './components/Search';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="container">
          <h1 className="app-title">AI-Powered Image Search</h1>
        </div>
      </header>
      <main className="container app-main">
        <Search />
      </main>
    </div>
  );
}

export default App;
