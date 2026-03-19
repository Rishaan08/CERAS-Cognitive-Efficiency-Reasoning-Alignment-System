import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { getAdaptiveResponse } from '../api';
import ReasoningTree from './ReasoningTree';
import './ResultsPanel.css';

function Expandable({ title, defaultOpen = false, children }) {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <div className="expandable-section">
            <button className="expandable-toggle" onClick={() => setOpen(o => !o)}>
                <span>{title}</span>
                <span className={`chevron ${open ? 'open' : ''}`}>▼</span>
            </button>
            {open && <div className="expandable-content">{children}</div>}
        </div>
    );
}

function ScoreBar({ label, value, className = '' }) {
    const level = value >= 0.75 ? 'high' : value >= 0.5 ? 'medium' : 'low';
    return (
        <div className="score-bar-container">
            <div className="score-bar-label">
                <span>{label}</span>
                <span>{(value * 100).toFixed(0)}%</span>
            </div>
            <div className="score-bar-track">
                <div
                    className={`score-bar-fill ${level}`}
                    style={{ width: `${value * 100}%` }}
                />
            </div>
        </div>
    );
}

export default function ResultsPanel({ result, prompt, config, hasResult }) {
    const [adaptiveRes, setAdaptiveRes] = useState(null);
    const [adaptiveLoading, setAdaptiveLoading] = useState(false);

    // Fetch adaptive response after main results arrive
    useEffect(() => {
        if (!result) return;
        setAdaptiveRes(null);
        setAdaptiveLoading(true);

        getAdaptiveResponse({
            prompt,
            steps: result.final_steps,
            ce_score: result.fused_score,
            diagnostics: result.diagnostics,
            main_provider: config.main_provider,
            main_model: config.main_model,
            groq_api_key: config.groq_api_key,
            gemini_api_key: config.gemini_api_key,
            openai_api_key: config.openai_api_key,
        })
            .then(data => setAdaptiveRes(data.response))
            .catch(() => setAdaptiveRes(null))
            .finally(() => setAdaptiveLoading(false));
    }, [result]);

    if (!hasResult) {
        return (
            <div className="empty-state fade-in">
                <h3>Enter a learning question above and run the session</h3>
                <ul>
                    <li>• Step-by-step reasoning</li>
                    <li>• Cognitive Efficiency score</li>
                    <li>• Diagnostic feedback</li>
                    <li>• Improvement suggestions</li>
                </ul>
            </div>
        );
    }

    if (!result) return null;

    const getScoreLevel = v => v >= 0.75 ? 'high' : v >= 0.5 ? 'medium' : 'low';

    const getReadinessColor = r => {
        if (r?.includes('High')) return 'var(--green)';
        if (r?.includes('Ready')) return 'var(--green)';
        if (r?.includes('Medium') || r?.includes('Needs')) return 'var(--orange)';
        return 'var(--red)';
    };

    const downloadReport = () => {
        const text = `CERAS Session Report

Date: ${new Date().toLocaleString()}
Session ID: session_1

Prompt: ${prompt}

Metrics:
- Formulation Time: ${result.formulation_time?.toFixed(2)}s
- Processing Time: ${result.runtime?.toFixed(2)}s
- Est. Tokens: ${result.total_tokens}

Scores:
- Fused CE Score: ${result.fused_score?.toFixed(2)}
- Structural (CEPM): ${result.cepm_score?.toFixed(2)}
- Semantic (CNN): ${result.cnn_score?.toFixed(2)}
- Readiness: ${result.readiness} (${result.confidence?.toFixed(2)} confidence)

Diagnostics:
Strengths:
${result.strengths?.map(s => `- ${s}`).join('\n')}

Suggestions:
${result.suggestions?.map(s => `- ${s}`).join('\n')}

Learning Response:
${result.final_steps?.map((s, i) => `${i + 1}. ${s}`).join('\n')}
`;
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ceras_report_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="results-panel">
            {/* Learning Response Steps */}
            <div className="results-response">
                <h2>📚 Learning Response</h2>
                <ol className="step-list">
                    {result.final_steps?.map((step, i) => (
                        <li className="step-item" key={i} style={{ animationDelay: `${i * 0.08}s` }}>
                            <span className="step-number">{i + 1}.</span>
                            {step}
                        </li>
                    ))}
                </ol>
            </div>

            {/* Session Metrics */}
            <h3 style={{ marginBottom: 12, fontSize: '1.05rem' }}>⏱️ Session Metrics</h3>
            <div className="metrics-row">
                <div className="metric-card">
                    <div className="metric-label">Formulation Time</div>
                    <div className="metric-value">{result.formulation_time?.toFixed(2)}s</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Processing Time</div>
                    <div className="metric-value">{result.runtime?.toFixed(2)}s</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Est. Tokens</div>
                    <div className="metric-value">{result.total_tokens}</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Features Extracted</div>
                    <div className="metric-value">{result.feature_count}</div>
                </div>
            </div>

            {/* Diagnostic Report */}
            <Expandable title="📋 Cognitive Diagnostic Report" defaultOpen={true}>
                <div className="diagnostic-grid">
                    <div className="diagnostic-col">
                        <h4>✅ Strengths</h4>
                        <ul>
                            {result.strengths?.map((s, i) => <li key={i}>{s}</li>)}
                        </ul>
                    </div>
                    <div className="diagnostic-col">
                        <h4>💡 Suggestions for Improvement</h4>
                        <ul>
                            {result.suggestions?.map((s, i) => <li key={i}>{s}</li>)}
                        </ul>
                    </div>
                </div>
            </Expandable>

            {/* CE Scores */}
            <div className="ce-section">
                <h3>🧠 Cognitive Efficiency Analysis</h3>
                <div className="ce-scores-grid">
                    <div className="ce-score-card">
                        <div className={`ce-score-value ${getScoreLevel(result.fused_score)}`}>
                            {result.fused_score?.toFixed(2)}
                        </div>
                        <div className="ce-score-label">Fused CE Score</div>
                        <div className="ce-score-sub">Target: &gt; 0.7</div>
                    </div>
                    <div className="ce-score-card">
                        <div className={`ce-score-value ${getScoreLevel(result.cepm_score)}`}>
                            {result.cepm_score?.toFixed(2)}
                        </div>
                        <div className="ce-score-label">Structural (CEPM)</div>
                    </div>
                    <div className="ce-score-card">
                        <div className={`ce-score-value ${getScoreLevel(result.cnn_score)}`}>
                            {result.cnn_score?.toFixed(2)}
                        </div>
                        <div className="ce-score-label">Semantic (CNN)</div>
                    </div>
                    <div className="ce-score-card">
                        <div className="ce-score-value" style={{ color: getReadinessColor(result.readiness), fontSize: '1.2rem' }}>
                            {result.readiness}
                        </div>
                        <div className="ce-score-label">Readiness</div>
                        <div className="ce-score-sub">Confidence: {result.confidence?.toFixed(2)}</div>
                    </div>
                </div>

                <ScoreBar label="Fused CE Score" value={result.fused_score || 0} />
                <ScoreBar label="Structural (CEPM)" value={result.cepm_score || 0} />
                <ScoreBar label="Semantic (CNN)" value={result.cnn_score || 0} />
            </div>

            {/* CE Score Explanation */}
            <Expandable title="❓ What is the Fused CE Score?">
                <div className="ce-explanation">
                    <h3>Fused Cognitive Efficiency (CE) Score</h3>
                    <p>
                        The <strong>Fused CE Score</strong> reflects how efficiently you are learning in this session.
                        It combines two independent signals:
                    </p>
                    <p>
                        • <strong>Conceptual Strength (CEPM)</strong> — Depth of understanding<br />
                        • <strong>Behavioral & Reasoning Alignment (CNN)</strong> — Interaction patterns, engagement consistency, and structural reasoning signals
                    </p>
                    <p>These are fused into a single score between <strong>0 and 1</strong>.</p>
                    <h3>What Your Level Means</h3>
                    <p><strong>0.00 – 0.44 → Foundation Building</strong><br />You may need to revisit core concepts and slow down.</p>
                    <p><strong>0.45 – 0.59 → Developing Momentum</strong><br />You're engaging and learning, but some inconsistencies exist.</p>
                    <p><strong>0.60 – 0.74 → Progressing Confidently</strong><br />You demonstrate stable understanding and good engagement.</p>
                    <p><strong>0.75 – 1.00 → Peak Learning State</strong><br />You are operating with strong clarity, alignment, and efficiency.</p>
                    <p><em>This score reflects learning efficiency — not intelligence — and adapts to your behavior in real time.</em></p>
                </div>
            </Expandable>

            {/* Telemetry */}
            <Expandable title="📡 Live Telemetry & Diagnostics">
                <div className="telemetry-grid">
                    <div className="telemetry-col">
                        <h4>Extracted Features</h4>
                        <pre className="telemetry-json">{JSON.stringify(result.features, null, 2)}</pre>
                    </div>
                    <div className="telemetry-col">
                        <h4>System Diagnostics</h4>
                        <pre className="telemetry-json">{JSON.stringify(result.diagnostics, null, 2)}</pre>
                    </div>
                </div>
            </Expandable>

            {/* Adaptive Learning Response */}
            <div className="adaptive-response">
                <h2>🎯 Adaptive Learning Response</h2>
                {adaptiveLoading ? (
                    <div className="adaptive-loading">
                        <div className="spinner-mini" />
                        <span>Generating personalized learning summary...</span>
                    </div>
                ) : adaptiveRes ? (
                    <div className="markdown-content">
                        <ReactMarkdown>{adaptiveRes}</ReactMarkdown>
                    </div>
                ) : (
                    <p style={{ color: 'var(--text-muted)' }}>Adaptive response unavailable.</p>
                )}
            </div>

            {/* Reasoning Trace */}
            <Expandable title="🔍 Reasoning Trace">
                <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 10 }}>
                    Detailed logs of the decomposition and verification process.
                </p>
                <ReasoningTree tree={result.tree} />
                <div className="trace-divider" />
                <pre className="trace-code">{result.logs || 'No logs available.'}</pre>
            </Expandable>

            {/* Download */}
            <button className="download-btn" onClick={downloadReport}>
                📥 Download Session Report
            </button>
        </div>
    );
}
