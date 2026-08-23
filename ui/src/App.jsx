import { useState, useEffect, useRef } from 'react'
import './index.css'

const API_BASE = '/api'

function App() {
  const [currentPage, setCurrentPage] = useState('chat')
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [expandedSources, setExpandedSources] = useState({})
  const bottomRef = useRef(null)

  const navigate = (page) => setCurrentPage(page)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendQuery = async (override) => {
    const q = (override ?? query).trim()
    if (!q || loading) return
    setLoading(true)
    setMessages((prev) => [...prev, { role: 'user', content: q }])
    setQuery('')

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Query failed')
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || 'No answer returned',
          sources: data.sources || [],
          latency_ms: data.latency_ms,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuery()
    }
  }

  const toggleSource = (key) =>
    setExpandedSources((prev) => ({ ...prev, [key]: !prev[key] }))

  const [documents, setDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [docSearch, setDocSearch] = useState('')

  const [evalData, setEvalData] = useState(null)
  const [evalLoading, setEvalLoading] = useState(false)
  const [evalError, setEvalError] = useState('')
  const [perfData, setPerfData] = useState(null)
  const [costData, setCostData] = useState(null)

  const filteredDocuments = documents.filter((d) =>
    d.name.toLowerCase().includes(docSearch.toLowerCase())
  )

  const loadDocuments = async () => {
    setDocumentsLoading(true)
    try {
      const res = await fetch(`${API_BASE}/documents`)
      const data = await res.json()
      setDocuments(data.documents || [])
    } catch {
      setDocuments([])
    } finally {
      setDocumentsLoading(false)
    }
  }

  useEffect(() => {
    if (currentPage === 'documents') loadDocuments()
    if (currentPage === 'evaluation') {
      setEvalLoading(true)
      setEvalError('')
      fetch(`${API_BASE}/evaluation/summary`)
        .then((r) => (r.ok ? r.json() : r.json().then((d) => { throw new Error(d.detail || 'Failed') })))
        .then(setEvalData)
        .catch((e) => setEvalError(e.message))
        .finally(() => setEvalLoading(false))
    }
    if (currentPage === 'performance') {
      fetch(`${API_BASE}/metrics/performance`).then((r) => r.json()).then(setPerfData).catch(() => {})
    }
    if (currentPage === 'cost') {
      fetch(`${API_BASE}/metrics/cost`).then((r) => r.json()).then(setCostData).catch(() => {})
    }
  }, [currentPage])

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setUploadStatus('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/documents`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setUploadStatus(`Indexed ${file.name} successfully`)
      loadDocuments()
    } catch (err) {
      setUploadStatus(`Error: ${err.message}`)
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  const handleDelete = async (doc) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(doc.id)}`, {
        method: 'DELETE',
      })
      if (!res.ok) throw new Error('Delete failed')
      loadDocuments()
    } catch (err) {
      setUploadStatus(`Error: ${err.message}`)
    }
  }

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-800 h-screen text-white flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold">RAGForge</h2>
          <p className="text-sm text-gray-400">Production RAG System</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          {[
            { id: 'chat', label: '💬 Chat' },
            { id: 'documents', label: '📚 Documents' },
            { id: 'evaluation', label: '📊 Evaluation' },
            { id: 'performance', label: '⚡ Performance' },
            { id: 'cost', label: '💰 Cost' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => navigate(item.id)}
              className={`w-full text-left px-4 py-2 rounded transition-colors ${
                currentPage === item.id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <div className="flex-1 p-6 overflow-y-auto">
        {/* Chat Page */}
        {currentPage === 'chat' && (
          <div className="flex flex-col h-screen max-w-4xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between py-4 border-b border-gray-200 dark:border-gray-700">
              <div>
                <h2 className="text-2xl font-bold">Production RAG Assistant</h2>
                <p className="text-sm text-gray-500">Hybrid retrieval (FAISS + BM25) · Cross-encoder reranking · Grounded answers</p>
              </div>
              {messages.length > 0 && (
                <button
                  onClick={() => { setMessages([]); setExpandedSources({}) }}
                  className="text-sm text-gray-400 hover:text-red-500 transition-colors"
                >
                  Clear chat
                </button>
              )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto py-6 pr-1 space-y-5">
              {messages.length === 0 && !loading && (
                <div className="h-full flex flex-col items-center justify-center text-center px-4">
                  <div className="text-5xl mb-4">💬</div>
                  <h3 className="text-xl font-semibold mb-2">Ask anything about your documents</h3>
                  <p className="text-gray-500 mb-8 max-w-md">
                    Answers are generated only from your indexed knowledge base, with citations to the source chunks.
                  </p>
                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      'What is the architecture of RAGForge?',
                      'What retrieval methods are used?',
                      'How does reranking work?',
                    ].map((s) => (
                      <button
                        key={s}
                        onClick={() => sendQuery(s)}
                        className="px-4 py-2 text-sm rounded-full border border-gray-300 text-gray-600 hover:border-blue-500 hover:text-blue-600 transition-colors"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => {
                const isUser = msg.role === 'user'
                return (
                  <div key={i} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                    <div
                      className={`rounded-2xl px-5 py-4 max-w-[85%] shadow-sm ${
                        isUser ? 'bg-blue-600 text-white' : 'bg-white text-black border border-gray-200'
                      }`}
                    >
                      <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{msg.content}</p>

                      {!isUser && msg.sources?.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-gray-100">
                          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
                            {msg.sources.length} sources
                            {msg.latency_ms != null && (
                              <span className="font-normal normal-case"> · {(msg.latency_ms / 1000).toFixed(1)}s</span>
                            )}
                          </p>
                          <div className="space-y-1.5">
                            {msg.sources.map((src, j) => {
                              const key = `${i}-${j}`
                              const open = expandedSources[key]
                              const name = (src.document_id || 'unknown').split('/').pop()
                              return (
                                <button
                                  key={key}
                                  onClick={() => toggleSource(key)}
                                  className="w-full text-left flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
                                >
                                  <span className="text-xs">{open ? '▾' : '▸'}</span>
                                  <span className="text-xs truncate flex-1">
                                    📄 {name} · chunk {src.chunk_id}
                                  </span>
                                  {src.score != null && (
                                    <span className="text-xs font-mono text-gray-400">
                                      {typeof src.score === 'number' ? src.score.toFixed(3) : src.score}
                                    </span>
                                  )}
                                </button>
                              )
                            })}
                          </div>
                          {msg.sources.some((_, j) => expandedSources[`${i}-${j}`]) && (
                            <div className="mt-2 space-y-2">
                              {msg.sources.map((src, j) =>
                                expandedSources[`${i}-${j}`] ? (
                                  <div key={`body-${i}-${j}`} className="px-3 py-2 rounded-lg bg-gray-900 text-gray-100 text-xs leading-relaxed font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                                    {src.text}
                                  </div>
                                ) : null
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}

              {loading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl px-5 py-4 bg-white border border-gray-200 shadow-sm">
                    <div className="flex gap-1.5">
                      {[0, 150, 300].map((d) => (
                        <span
                          key={d}
                          className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
                          style={{ animationDelay: `${d}ms` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Composer */}
            <form
              onSubmit={(e) => {
                e.preventDefault()
                sendQuery()
              }}
              className="py-4"
            >
              <div className="flex items-end gap-2 bg-white border border-gray-200 rounded-2xl shadow-sm p-2 focus-within:ring-2 focus-within:ring-blue-500 transition-shadow">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask me anything...  (Enter to send, Shift+Enter for new line)"
                  disabled={loading}
                  rows={1}
                  className="flex-1 resize-none px-3 py-2.5 text-[15px] bg-transparent focus:outline-none max-h-32"
                />
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="bg-blue-600 text-white px-5 py-2.5 rounded-xl hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all font-medium"
                >
                  {loading ? '...' : 'Send'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Documents Page */}
        {currentPage === 'documents' && (
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">Documents</h2>
              <label className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm cursor-pointer">
                {uploading ? 'Uploading...' : '+ Upload'}
                <input
                  type="file"
                  accept=".pdf,.txt,.md,.html"
                  className="hidden"
                  onChange={handleUpload}
                  disabled={uploading}
                />
              </label>
            </div>
            {uploadStatus && (
              <p className={`mb-4 text-sm ${uploadStatus.startsWith('Error') ? 'text-red-600' : 'text-green-600'}`}>
                {uploadStatus}
              </p>
            )}

            <div className="bg-white rounded-lg p-6 shadow shadow-gray-200">
              <input
                type="text"
                placeholder="Search documents..."
                value={docSearch}
                onChange={(e) => setDocSearch(e.target.value)}
                className="w-full rounded-lg px-4 py-3 border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <div className="mt-6">
                <h3 className="font-medium mb-4">Documents in Knowledge Base</h3>
                <div className="space-y-3">
                  {documentsLoading && <p className="text-gray-400 text-sm">Loading documents...</p>}
                  {!documentsLoading && filteredDocuments.length === 0 && (
                    <p className="text-gray-400 text-sm">No documents found</p>
                  )}
                  {filteredDocuments.map((doc) => (
                    <div
                      key={doc.id}
                      className="flex items-center justify-between p-4 rounded-lg border border-dashed border-gray-300 hover:border-blue-500 transition-colors"
                    >
                      <div>
                        <div className="font-medium">{doc.name}</div>
                        <div className="text-xs text-gray-500">{(doc.size / 1024).toFixed(1)} KB · {doc.status}</div>
                      </div>
                      <button
                        onClick={() => handleDelete(doc)}
                        className="text-red-500 hover:text-red-700 text-sm font-medium"
                      >
                        Delete
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Evaluation Page */}
        {currentPage === 'evaluation' && (
          <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-lg p-6 shadow shadow-gray-200">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Evaluation Dashboard</h2>
                <button
                  onClick={() => setCurrentPage('evaluation')}
                  className="text-sm text-gray-400 hover:text-blue-600"
                >
                  Refresh
                </button>
              </div>

              {evalLoading && <p className="text-gray-400 text-sm">Running retrieval evaluation...</p>}
              {evalError && (
                <p className="text-red-500 text-sm mb-4">Error: {evalError}</p>
              )}

              {evalData && (
                <>
                  <div className="grid grid-cols-2 gap-6 mb-8">
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold text-blue-600">
                        {((evalData.precision_at_5 ?? 0) * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-500 mt-2">Precision@5</p>
                    </div>
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold text-green-600">
                        {((evalData.recall_at_5 ?? 0) * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-500 mt-2">Hit Rate@5</p>
                    </div>
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold text-purple-600">
                        {(evalData.mrr ?? 0).toFixed(3)}
                      </p>
                      <p className="text-sm text-gray-500 mt-2">MRR</p>
                    </div>
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold text-gray-700">{evalData.total_evaluations}</p>
                      <p className="text-sm text-gray-500 mt-2">Eval Queries</p>
                    </div>
                  </div>

                  <div className="rounded-lg bg-gray-900 text-gray-300 p-4 text-xs font-mono space-y-1">
                    {[
                      ['Precision@5', evalData.precision_at_5],
                      ['Hit Rate@5', evalData.recall_at_5],
                      ['MRR', evalData.mrr],
                    ].map(([label, val]) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="w-24 shrink-0">{label}</span>
                        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full transition-all duration-700"
                            style={{ width: `${(val ?? 0) * 100}%` }}
                          />
                        </div>
                        <span className="w-14 text-right">{((val ?? 0) * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-gray-400 mt-3">
                    Retrieval-only evaluation against a built-in QA set. Faithfulness / hallucination scoring (LLM-judged) coming next.
                  </p>
                </>
              )}
            </div>
          </div>
        )}

        {/* Performance Page */}
        {currentPage === 'performance' && (
          <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-lg p-6 shadow shadow-gray-200">
              <h2 className="text-2xl font-bold mb-6">Performance</h2>

              {!perfData || perfData.total_requests === 0 ? (
                <p className="text-gray-400 text-sm py-8 text-center">
                  No queries recorded yet. Ask something in Chat, then come back.
                </p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-6 mb-8">
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold">{perfData.total_requests.toLocaleString()}</p>
                      <p className="text-sm text-gray-500 mt-2">Requests</p>
                    </div>
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold text-blue-600">
                        {(perfData.percentiles.p95 / 1000).toFixed(2)}s
                      </p>
                      <p className="text-sm text-gray-500 mt-2">P95 Latency</p>
                    </div>
                  </div>

                  <p className="font-medium mb-3">Avg Pipeline Latency Breakdown</p>
                  <div className="space-y-3 mb-8">
                    {(() => {
                      const total = perfData.avg_latency_ms || 1
                      const stages = [
                        { name: 'Retrieval (hybrid + rerank)', ms: perfData.avg_retrieval_ms, color: 'bg-blue-500' },
                        { name: 'LLM generation', ms: perfData.avg_generation_ms, color: 'bg-orange-500' },
                      ]
                      return stages.map((s) => (
                        <div key={s.name}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-600">{s.name}</span>
                            <span className="font-mono text-gray-500">{s.ms.toFixed(0)} ms</span>
                          </div>
                          <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${s.color} rounded-full transition-all duration-700`}
                              style={{ width: `${Math.min((s.ms / total) * 100, 100)}%` }}
                            />
                          </div>
                        </div>
                      ))
                    })()}
                  </div>

                  <div className="grid grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-gray-50 text-center">
                      <p className="text-xl font-bold">{(perfData.avg_latency_ms / 1000).toFixed(2)}s</p>
                      <p className="text-xs text-gray-500">Avg</p>
                    </div>
                    <div className="p-4 rounded-lg bg-gray-50 text-center">
                      <p className="text-xl font-bold">{(perfData.percentiles.p50 / 1000).toFixed(2)}s</p>
                      <p className="text-xs text-gray-500">P50</p>
                    </div>
                    <div className="p-4 rounded-lg bg-blue-50 text-center">
                      <p className="text-xl font-bold text-blue-600">{(perfData.percentiles.p95 / 1000).toFixed(2)}s</p>
                      <p className="text-xs text-gray-500">P95</p>
                    </div>
                    <div className="p-4 rounded-lg bg-gray-50 text-center">
                      <p className="text-xl font-bold">{(perfData.percentiles.p99 / 1000).toFixed(2)}s</p>
                      <p className="text-xs text-gray-500">P99</p>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Cost Page */}
        {currentPage === 'cost' && (
          <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-lg p-6 shadow shadow-gray-200">
              <h2 className="text-2xl font-bold mb-1">Cost Analytics</h2>
              <p className="text-xs text-gray-400 mb-6">
                Cloud-equivalent cost estimate for local inference (ref pricing: $0.15/M input, $0.60/M output tokens)
              </p>

              {!costData || costData.total_queries === 0 ? (
                <p className="text-gray-400 text-sm py-8 text-center">
                  No queries recorded yet. Ask something in Chat, then come back.
                </p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-6 mb-8">
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold">{costData.total_queries.toLocaleString()}</p>
                      <p className="text-sm text-gray-500 mt-2">Queries</p>
                    </div>
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold text-green-600">
                        ${costData.avg_cost_per_query.toFixed(4)}
                      </p>
                      <p className="text-sm text-gray-500 mt-2">Avg Cost / Query</p>
                    </div>
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold">${costData.monthly_equivalent.toFixed(2)}</p>
                      <p className="text-sm text-gray-500 mt-2">Monthly equivalent*</p>
                    </div>
                    <div className="p-6 rounded-lg bg-gray-50">
                      <p className="text-3xl font-bold">{(costData.total_tokens / 1000).toFixed(1)}k</p>
                      <p className="text-sm text-gray-500 mt-2">Total tokens</p>
                    </div>
                  </div>

                  <p className="font-medium mb-3">Cost Distribution</p>
                  {(() => {
                    const bd = costData.cost_breakdown
                    const total = bd.llm + bd.embeddings + bd.reranking || 1
                    const rows = [
                      { name: 'LLM generation', value: bd.llm, color: 'bg-orange-500' },
                      { name: 'Embeddings', value: bd.embeddings, color: 'bg-blue-500' },
                      { name: 'Reranking', value: bd.reranking, color: 'bg-purple-500' },
                    ]
                    return rows.map((r) => (
                      <div key={r.name} className="mb-3">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-600">{r.name}</span>
                          <span className="font-mono text-gray-500">${r.value.toFixed(5)}</span>
                        </div>
                        <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${r.color} rounded-full transition-all duration-700`}
                            style={{ width: `${Math.max((r.value / total) * 100, r.value > 0 ? 2 : 0)}%` }}
                          />
                        </div>
                      </div>
                    ))
                  })()}
                  <p className="text-xs text-gray-400 mt-4">
                    * Extrapolated from avg cost/query assuming ~38,500 queries/month.
                  </p>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App