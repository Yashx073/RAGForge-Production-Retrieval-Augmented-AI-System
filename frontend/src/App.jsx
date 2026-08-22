import { useState } from 'react'
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import Chat from './pages/Chat'
import Documents from './pages/Documents'
import Evaluation from './pages/Evaluation'
import Performance from './pages/Performance'
import Cost from './pages/Cost'
import Settings from './pages/Settings'

const navigation = [
  { name: 'Chat', href: '/', icon: '💬' },
  { name: 'Documents', href: '/documents', icon: '📚' },
  { name: 'Evaluation', href: '/evaluation', icon: '📊' },
  { name: 'Performance', href: '/performance', icon: '⚡' },
  { name: 'Cost', href: '/cost', icon: '💰' },
  { name: 'Settings', href: '/settings', icon: '⚙️' },
]

function Sidebar() {
  const location = useLocation()
  return (
    <aside className="w-64 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 min-h-screen flex flex-col">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <span>🔮</span> RAGForge
        </h1>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Production RAG System</p>
      </div>
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => (
          <Link
            key={item.name}
            to={item.href}
            className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              location.pathname === item.href
                ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            <span>{item.icon}</span>
            <span>{item.name}</span>
          </Link>
        ))}
      </nav>
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <span className="w-2 h-2 rounded-full bg-green-500"></span>
          <span>Ollama Online</span>
        </div>
      </div>
    </aside>
  )
}

function Layout({ children }) {
  return (
    <div className="flex min-h-screen bg-gray-50 dark:bg-gray-950">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-6xl mx-auto">{children}</div>
      </main>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout><Chat /></Layout>} />
        <Route path="/documents" element={<Layout><Documents /></Layout>} />
        <Route path="/evaluation" element={<Layout><Evaluation /></Layout>} />
        <Route path="/performance" element={<Layout><Performance /></Layout>} />
        <Route path="/cost" element={<Layout><Cost /></Layout>} />
        <Route path="/settings" element={<Layout><Settings /></Layout>} />
      </Routes>
    </BrowserRouter>
  )
}

export default App