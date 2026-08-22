import { useState, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function Evaluation() {
  const [summary, setSummary] = useState(null)
  const [history, setHistory] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchEvaluation()
  }, [])

  const fetchEvaluation = async () => {
    try {
      const [summaryRes, historyRes] = await Promise.all([
        fetch(`${API_BASE}/evaluation/summary`),
        fetch(`${API_BASE}/evaluation/history`),
      ])

      if (summaryRes.ok) {
        const data = await summaryRes.json()
        setSummary(data)
      }

      if (historyRes.ok) {
        const data = await historyRes.json()
        setHistory(data.evaluations || [])
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const formatPercent = (val) => `${(val * 100).toFixed(1)}%`
  const formatScore = (val) => val.toFixed(2)

  const metricCards = [
    { label: 'Precision@5', value: summary?.precision_at_5, format: formatPercent, color: 'blue' },
    { label: 'Recall@5', value: summary?.recall_at_5, format: formatPercent, color: 'green' },
    { label: 'MRR', value: summary?.mrr, format: formatScore, color: 'purple' },
    { label: 'NDCG@5', value: summary?.ndcg_at_5, format: formatScore, color: 'orange' },
    { label: 'Faithfulness', value: summary?.faithfulness, format: formatScore, color: 'indigo' },
    { label: 'Answer Relevance', value: summary?.answer_relevance, format: formatScore, color: 'teal' },
    { label: 'Context Relevance', value: summary?.context_relevance, format: formatScore, color: 'cyan' },
    { label: 'Hallucination Rate', value: summary?.hallucination_rate, format: formatPercent, color: 'red' },
  ]

  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
            <span className="text-4xl">📊</span>
            Evaluation Dashboard
          </h1>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 animate-pulse">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
              <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
          <span className="text-4xl">📊</span>
          Evaluation Dashboard
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Retrieval and generation quality metrics
        </p>
      </header>

      {error && (
        <div className="mb-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {metricCards.map((metric) => (
          <div
            key={metric.label}
            className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">{metric.label}</h3>
            <div className="text-3xl font-bold text-gray-900 dark:text-white">
              {metric.value !== null && metric.value !== undefined
                ? metric.format(metric.value)
                : '—'}
            </div>
            <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r"
                style={{
                  width: `${metric.value !== null && metric.value !== undefined ? metric.value * 100 : 0}%`,
                  background: metric.color === 'red'
                    ? 'linear-gradient(90deg, #ef4444, #f87171)'
                    : metric.color === 'blue'
                    ? 'linear-gradient(90deg, #3b82f6, #60a5fa)'
                    : metric.color === 'green'
                    ? 'linear-gradient(90deg, #22c55e, #4ade80)'
                    : metric.color === 'purple'
                    ? 'linear-gradient(90deg, #a855f7, #c084fc)'
                    : metric.color === 'orange'
                    ? 'linear-gradient(90deg, #f97316, #fb923c)'
                    : metric.color === 'indigo'
                    ? 'linear-gradient(90deg, #6366f1, #818cf8)'
                    : metric.color === 'teal'
                    ? 'linear-gradient(90deg, #14b8a6, #2dd4bf)'
                    : 'linear-gradient(90deg, #06b6d4, #22d3ee)'
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Evaluation History</h2>
        </div>
        {history.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            No evaluation history available. Run evaluations to populate this dashboard.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-900/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Date</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Query</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Precision@5</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">MRR</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Faithfulness</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Hallucination</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {history.map((eval, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    <td className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                      {new Date(eval.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-white max-w-xs truncate">
                      {eval.query}
                    </td>
                    <td className="px-4 py-3 text-center text-sm font-mono font-medium">
                      {eval.precision_at_5 !== undefined ? formatPercent(eval.precision_at_5) : '—'}
                    </td>
                    <td className="px-4 py-3 text-center text-sm font-mono font-medium">
                      {eval.mrr !== undefined ? formatScore(eval.mrr) : '—'}
                    </td>
                    <td className="px-4 py-3 text-center text-sm font-mono font-medium">
                      {eval.faithfulness !== undefined ? formatScore(eval.faithfulness) : '—'}
                    </td>
                    <td className="px-4 py-3 text-center text-sm font-mono font-medium text-red-600 dark:text-red-400">
                      {eval.hallucination_rate !== undefined ? formatPercent(eval.hallucination_rate) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}