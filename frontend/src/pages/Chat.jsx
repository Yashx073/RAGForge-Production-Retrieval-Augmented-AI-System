import { useState, useRef, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function Chat() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [latency, setLatency] = useState(null)
  const [tokenUsage, setTokenUsage] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showSources, setShowSources] = useState(true)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [answer, sources])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim() || isLoading) return

    setIsLoading(true)
    setError(null)
    setAnswer('')
    setSources([])
    setLatency(null)
    setTokenUsage(null)

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), top_k: 5 }),
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Query failed')
      }

      const data = await response.json()
      setAnswer(data.answer)
      setSources(data.sources)
      setLatency(data.latency_ms)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const formatLatency = (ms) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  return (
    <div className="max-w-4xl mx-auto">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
          <span className="text-4xl">💬</span>
          Production RAG Assistant
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Ask questions about your documents. Powered by hybrid retrieval + reranking.
        </p>
      </header>

      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden">
        <div className="flex h-[500px] flex-col">
          <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300">
                <strong>Error:</strong> {error}
              </div>
            )}

            {query && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center flex-shrink-0 text-sm font-medium text-gray-600 dark:text-gray-400">
                  U
                </div>
                <div className="bg-gray-100 dark:bg-gray-800 rounded-2xl px-4 py-3 max-w-[85%]">
                  <p className="text-gray-900 dark:text-white whitespace-pre-wrap">{query}</p>
                </div>
              </div>
            )}

            {answer && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center flex-shrink-0 text-sm font-medium text-purple-700 dark:text-purple-300">
                  🤖
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-2xl px-4 py-3 max-w-[85%] border border-purple-100 dark:border-purple-800">
                  <div className="prose dark:prose-invert max-w-none text-gray-900 dark:text-white">
                    {answer.split('\n').map((line, i) => (
                      <p key={i} className="whitespace-pre-wrap">{line}</p>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center flex-shrink-0">
                  <div className="w-5 h-5 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-2xl px-4 py-3 max-w-[85%] border border-purple-100 dark:border-purple-800">
                  <div className="flex gap-1 items-center text-purple-700 dark:text-purple-300">
                    <span className="animate-bounce">●</span>
                    <span className="animate-bounce" style={{ animationDelay: '0.1s' }}>●</span>
                    <span className="animate-bounce" style={{ animationDelay: '0.2s' }}>●</span>
                    <span className="ml-2 text-sm">Generating answer...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-900/50">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question about your documents..."
                className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !query.trim()}
                className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Thinking...
                  </>
                ) : (
                  'Send'
                )}
              </button>
            </form>
          </div>
        </div>

        {showSources && sources.length > 0 && (
          <div className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
            <div className="p-4 flex items-center justify-between">
              <h3 className="font-semibold text-gray-900 dark:text-white">Sources ({sources.length} retrieved)</h3>
              <button
                onClick={() => setShowSources(!showSources)}
                className="text-sm text-purple-600 dark:text-purple-400 hover:underline"
              >
                Hide
              </button>
            </div>
            <div className="px-4 pb-4 space-y-3 max-h-96 overflow-y-auto scrollbar-thin">
              {sources.map((source, index) => (
                <div
                  key={index}
                  className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded">
                        [{index + 1}]
                      </span>
                      <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {source.document_id}
                      </span>
                    </div>
                    {source.score !== null && source.score !== undefined && (
                      <span className="text-xs font-mono text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-900/30 px-2 py-0.5 rounded flex-shrink-0">
                        Score: {source.score.toFixed(3)}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-3 font-mono text-xs">
                    {source.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {latency && (
          <div className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 px-4 py-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500 dark:text-gray-400">Response latency</span>
              <span className="font-mono font-semibold text-gray-900 dark:text-white">
                {formatLatency(latency)}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Latency Breakdown</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total</span>
              <span className="font-mono font-semibold text-gray-900 dark:text-white">{formatLatency(latency || 0)}</span>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Token Usage</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Input tokens</span>
              <span className="font-mono font-semibold text-gray-900 dark:text-white">{tokenUsage?.input || '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Output tokens</span>
              <span className="font-mono font-semibold text-gray-900 dark:text-white">{tokenUsage?.output || '—'}</span>
            </div>
            <div className="flex justify-between border-t border-gray-200 dark:border-gray-700 pt-2">
              <span className="text-gray-600 dark:text-gray-400">Total</span>
              <span className="font-mono font-semibold text-gray-900 dark:text-white">{tokenUsage?.total || '—'}</span>
            </div>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Estimated Cost</h3>
          <div className="text-2xl font-bold text-gray-900 dark:text-white font-mono">
            ${((tokenUsage?.total || 0) / 1_000_000) * 0.0018 || 0.toFixed(6)}
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Local inference equivalent</p>
        </div>
      </div>
    </div>
  )
}