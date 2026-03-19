import { useState } from 'react';
import useHistory from '../hooks/useHistory';
import './HistoryPanel.css';

export default function HistoryPanel({ userId, onSelectSession, onClose }) {
    const { sessions, loading, searchQuery, setSearchQuery, deleteSession } = useHistory(userId);
    const [expandedId, setExpandedId] = useState(null);

    const formatDate = (dateStr) => {
        const d = new Date(dateStr);
        const now = new Date();
        const diffMs = now - d;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    const handleSelect = (session) => {
        if (!session.chat_messages?.length) return;
        const msg = session.chat_messages[0];
        const metrics = msg.session_metrics?.[0] || {};
        onSelectSession({
            sessionId: session.id,
            messageId: msg.id,
            prompt: msg.prompt,
            result: {
                final_steps: msg.final_steps || [],
                strategy_used: msg.strategy_used,
                llm_calls_used: msg.llm_calls_used,
                cepm_score: metrics.cepm_score,
                cnn_score: metrics.cnn_score,
                fused_score: metrics.fused_score,
                confidence: metrics.confidence,
                readiness: metrics.readiness,
                runtime: metrics.runtime,
                formulation_time: metrics.formulation_time,
                total_tokens: metrics.total_tokens,
                features: {
                    prompt_length: metrics.prompt_length,
                    character_count: metrics.character_count,
                    sentence_count: metrics.sentence_count,
                    unique_word_ratio: metrics.unique_word_ratio,
                    concept_density: metrics.concept_density,
                    prompt_quality: metrics.prompt_quality,
                    keystrokes: metrics.keystrokes,
                    prompt_type: metrics.prompt_type,
                },
                feature_count: 8,
                diagnostics: {},
                strengths: [],
                suggestions: [],
                tree: null,
                logs: '',
            },
            config: {
                main_provider: session.main_provider,
                verifier_provider: session.verifier_provider,
                main_model: session.main_model,
                verifier_model: session.verifier_model,
            },
        });
        onClose();
    };

    return (
        <div className="history-panel">
            <div className="history-header">
                <h3>📜 Chat History</h3>
                <button className="history-close" onClick={onClose}>✕</button>
            </div>

            <div className="history-search">
                <input
                    type="text"
                    placeholder="Search prompts..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                />
            </div>

            <div className="history-list">
                {loading && (
                    <div className="history-loading">
                        <div className="spinner-mini" />
                        <span>Loading history...</span>
                    </div>
                )}

                {!loading && sessions.length === 0 && (
                    <div className="history-empty">
                        <span>🕳️</span>
                        <p>No sessions yet. Run your first prompt!</p>
                    </div>
                )}

                {sessions.map(session => {
                    const msg = session.chat_messages?.[0];
                    const metrics = msg?.session_metrics?.[0];
                    const isExpanded = expandedId === session.id;

                    return (
                        <div
                            key={session.id}
                            className={`history-item ${isExpanded ? 'expanded' : ''}`}
                        >
                            <div
                                className="history-item-header"
                                onClick={() => setExpandedId(isExpanded ? null : session.id)}
                            >
                                <div className="history-item-info">
                                    <span className="history-title">
                                        {session.session_title || 'Untitled'}
                                    </span>
                                    <span className="history-meta">
                                        {session.main_provider} • {session.main_model} • {formatDate(session.created_at)}
                                    </span>
                                </div>
                                {metrics?.fused_score != null && (
                                    <div className="history-score">
                                        {(metrics.fused_score * 100).toFixed(0)}%
                                    </div>
                                )}
                            </div>

                            {isExpanded && msg && (
                                <div className="history-item-detail">
                                    <p className="history-prompt">{msg.prompt}</p>
                                    {metrics && (
                                        <div className="history-metrics-row">
                                            <span>CEPM: {(metrics.cepm_score * 100).toFixed(0)}%</span>
                                            <span>CNN: {(metrics.cnn_score * 100).toFixed(0)}%</span>
                                            <span>Tokens: {metrics.total_tokens}</span>
                                            <span>Time: {metrics.runtime?.toFixed(1)}s</span>
                                        </div>
                                    )}
                                    <div className="history-actions">
                                        <button className="history-replay" onClick={() => handleSelect(session)}>
                                            ▶ Replay
                                        </button>
                                        <button className="history-delete" onClick={() => deleteSession(session.id)}>
                                            🗑️ Delete
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
