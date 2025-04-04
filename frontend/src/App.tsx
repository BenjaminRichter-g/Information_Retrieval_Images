// src/App.tsx
import React from 'react';
import Search from './components/Search';

function App() {
  return (
    <div className="App">
      <header className="bg-gray-800 text-white p-4">
        <h1 className="text-xl font-bold">AI-Powered Image Search</h1>
      </header>
      <main className="p-4">
        <Search />
      </main>
    </div>
  );
}

export default App;
