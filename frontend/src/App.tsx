import React, { useState, useEffect } from 'react';
import { Brain, Database, FileText, CheckCircle, AlertTriangle, Key } from 'lucide-react';
import ChatPanel from './components/ChatPanel';
import { checkStatus } from './services/api';
import type { SystemStatus } from './services/api';

function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    // Initial status check
    const check = async () => {
      const s = await checkStatus();
      setStatus(s);
    };
    check();
  }, []);

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg" alt="NVIDIA" style={{ height: '24px' }} />
          <span>Dual-Brain</span>
        </div>

        <div style={{ marginBottom: '8px', fontSize: '12px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 600 }}>
          System Status
        </div>

        <div className={`status-badge ${status?.api_key_set ? 'success' : 'error'}`}>
          <Key size={16} />
          {status?.api_key_set ? 'NVIDIA API Key' : 'No API Key Found'}
        </div>

        <div className={`status-badge ${status?.faiss_index_exists ? 'success' : 'warning'}`}>
          <Database size={16} />
          {status?.faiss_index_exists ? 'Technical Knowledge Base' : 'KB Missing'}
        </div>

        <div className={`status-badge ${status?.resume_index_exists ? 'success' : 'warning'}`}>
          <FileText size={16} />
          {status?.resume_index_exists ? 'Resume Context' : 'Resume Missing'}
        </div>

        <div className="modes-section">
          <div style={{ marginBottom: '16px', fontSize: '12px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 600 }}>
            Available Modes
          </div>
          <div className="mode-card active">
            <div style={{ fontWeight: 600, color: '#fff', marginBottom: '4px' }}>1. NVIDIA Tutor</div>
            Technical deep dives with SA persona
          </div>
          <div className="mode-card">
            <div style={{ fontWeight: 600, color: '#fff', marginBottom: '4px' }}>2. General Tutor</div>
            System design and coding frameworks
          </div>
          <div className="mode-card">
            <div style={{ fontWeight: 600, color: '#fff', marginBottom: '4px' }}>3. Hardware Calculator</div>
            Automatic VRAM & GPU sizing
          </div>
          <div className="mode-card">
            <div style={{ fontWeight: 600, color: '#fff', marginBottom: '4px' }}>4. Deployment Expert</div>
            Triton & TensorRT simulations
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        <header className="top-bar">
          <div style={{ fontSize: '20px', fontWeight: 600 }}>🤖 Solutions Architect Prep</div>
          <div style={{ fontSize: '12px', color: '#888' }}>
            Powered by Llama-3.1-70B, RAG, and Sequential Thinking
          </div>
        </header>

        <ChatPanel />
      </main>
    </div>
  );
}

export default App;
