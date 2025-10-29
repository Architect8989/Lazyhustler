import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

// Pages
import Dashboard from './pages/Dashboard';
import TaskExecutor from './pages/TaskExecutor';
import TaskHistory from './pages/TaskHistory';
import Connectors from './pages/Connectors';
import AGIGrowth from './pages/AGIGrowth';
import Settings from './pages/Settings';

// Components
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import NotificationCenter from './components/NotificationCenter';

interface SystemStatus {
  status: 'online' | 'offline';
  tasks_running: number;
  total_tasks: number;
  agi_intelligence: number;
  uptime: string;
}

function App() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'online',
    tasks_running: 0,
    total_tasks: 0,
    agi_intelligence: 0.35,
    uptime: '2h 15m'
  });

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [notifications, setNotifications] = useState<any[]>([]);

  useEffect(() => {
    // Fetch system status
    const interval = setInterval(() => {
      // In production, fetch from API
      setSystemStatus(prev => ({
        ...prev,
        tasks_running: Math.floor(Math.random() * 5),
        agi_intelligence: Math.min(prev.agi_intelligence + 0.001, 1.0)
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Router>
      <div className="app-container">
        <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
        
        <div className="main-content">
          <Header 
            systemStatus={systemStatus}
            onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          />

          <div className="content-area">
            <Routes>
              <Route path="/" element={<Dashboard systemStatus={systemStatus} />} />
              <Route path="/executor" element={<TaskExecutor />} />
              <Route path="/history" element={<TaskHistory />} />
              <Route path="/connectors" element={<Connectors />} />
              <Route path="/growth" element={<AGIGrowth />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </div>
        </div>

        <NotificationCenter notifications={notifications} />
      </div>
    </Router>
  );
}

export default App;
