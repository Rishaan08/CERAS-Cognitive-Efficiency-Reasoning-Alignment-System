import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { getAdaptiveResponse } from '../api';
import { supabase } from '../lib/supabase';
import FollowUpChat from './FollowUpChat';
import ReasoningTree from './ReasoningTree';
import WorkflowModal from './WorkflowModal';
import './Dashboard.css';

/* ---- Collapsible Section ---- */
function Collapse({ title, defaultOpen = false, children }) {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <div className="dash-collapse">
            <button className="dash-collapse-btn" onClick={() => setOpen(o => !o)}>
                <span>{title}</span>
                <span className={`collapse-icon ${open ? 'open' : ''}`}>▼</span>
            </button>
            {open && <div className="dash-collapse-body">{children}</div>}
        </div>
    );
}

/* ---- Score Color Helpers ---- */
const scoreColor = v => {
    if (v >= 0.75) return '#22c55e';
    if (v >= 0.50) return '#60a5fa';
    if (v >= 0.35) return '#f59e0b';
    return '#ef4444';
};
const barClass = v => {
    if (v >= 0.75) return 'green';
    if (v >= 0.50) return 'blue';
    if (v >= 0.35) return 'amber';
    return 'red';
};
const readinessClass = r => {
    if (!r) return 'low';
    if (r.includes('High') || r.includes('Ready')) return 'high';
    if (r.includes('Medium') || r.includes('Needs')) return 'medium';
    return 'low';
};

/* ---- Percentile estimate (simple heuristic) ---- */
const getPercentile = (score) => {
    if (score >= 0.9) return 'Top 5%';
    if (score >= 0.8) return 'Top 10%';
    if (score >= 0.7) return 'Top 20%';
    if (score >= 0.6) return 'Top 35%';
    if (score >= 0.5) return 'Top 50%';
    return 'Below median';
};

export default function Dashboard({ result, prompt, config, hasResult, typingAnalytics, sessionId, messageId, userId }) {
    const [adaptiveRes, setAdaptiveRes] = useState(null);
    const [adaptiveLoading, setAdaptiveLoading] = useState(false);
    const [showWorkflow, setShowWorkflow] = useState(false);
    const [followupCost, setFollowupCost] = useState({ tokens: 0, cost: 0 });
    const [planCost, setPlanCost] = useState({ tokens: 0, cost: 0 });
    // Animated fill: start from 0 and transition to actual score
    const [animatedScore, setAnimatedScore] = useState(0);
    const heroRef = useRef(null);

    useEffect(() => {
        if (!result || result.readiness === 'Error') return;
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
            .then(d => setAdaptiveRes(d.response))
            .catch(() => setAdaptiveRes(null))
            .finally(() => setAdaptiveLoading(false));
    }, [result]);

    // Animate score from 0 when result arrives
    useEffect(() => {
        if (result && result.fused_score != null) {
            setAnimatedScore(0);
            const timer = requestAnimationFrame(() => {
                setTimeout(() => setAnimatedScore(result.fused_score), 50);
            });
            return () => cancelAnimationFrame(timer);
        }
    }, [result]);

    /* ---- Empty State ---- */
    if (!hasResult) {
        return (
            <div className="dash-empty fade-in">
                <div className="dash-empty-icon">🧠</div>
                <h3>Ready to Analyze</h3>
                <p>Enter a learning question above and run the session to see your cognitive efficiency dashboard.</p>
            </div>
        );
    }
    if (!result) return null;

    // Main ring (inner) — fused CE score
    const innerR = 82;
    const innerCirc = 2 * Math.PI * innerR;
    const innerOffset = innerCirc - (animatedScore * innerCirc);

    // Outer ring — structural vs semantic dominance
    const outerR = 92;
    const outerCirc = 2 * Math.PI * outerR;
    const cepmRatio = result.cepm_score / (result.cepm_score + result.cnn_score + 0.001);
    const cepmArc = cepmRatio * outerCirc;
    const cnnArc = (1 - cepmRatio) * outerCirc;

    // Delta from baseline (0.5)
    const baseline = 0.5;
    const delta = result.fused_score - baseline;
    const deltaSign = delta >= 0 ? '+' : '';

    const downloadReport = async () => {
        const lines = [
            `CERAS Session Report`,
            `${'='.repeat(40)}`,
            `Date: ${new Date().toLocaleString()}`,
            ``,
            `Prompt: ${prompt}`,
            ``,
            `Scores:`,
            `  Fused CE:    ${result.fused_score?.toFixed(3)}`,
            `  CEPM:        ${result.cepm_score?.toFixed(3)}`,
            `  CNN:         ${result.cnn_score?.toFixed(3)}`,
            `  Confidence:  ${result.confidence?.toFixed(3)}`,
            `  Readiness:   ${result.readiness}`,
            ``,
            `Metrics:`,
            `  Formulation: ${result.formulation_time?.toFixed(2)}s`,
            `  Processing:  ${result.runtime?.toFixed(2)}s`,
            `  Tokens:      ${result.total_tokens}`,
            `  Features:    ${result.feature_count}`,
            `  LLM Calls:   ${result.llm_calls_used}`,
            ``,
            `Typing Analytics:`,
            `  Keystrokes:  ${typingAnalytics?.totalKeystrokes || 0}`,
            `  WPM:         ${typingAnalytics?.wpm || 0}`,
            `  Hesitations: ${typingAnalytics?.hesitations || 0}`,
            `  Edit Rate:   ${((typingAnalytics?.deletionRatio || 0) * 100).toFixed(0)}%`,
            ``,
            `Strengths:`,
            ...(result.strengths?.map(s => `  • ${s}`) || []),
            ``,
            `Suggestions:`,
            ...(result.suggestions?.map(s => `  • ${s}`) || []),
            ``,
            `Learning Steps:`,
            ...(result.final_steps?.map((s, i) => `  ${i + 1}. ${s}`) || []),
            ``,
            adaptiveRes ? `Adaptive Response:\n${adaptiveRes}` : '',
        ];
        const reportContent = lines.join('\n');
        if (sessionId && messageId && userId) {
            try {
                await supabase.from('session_reports').insert({
                    session_id: sessionId,
                    message_id: messageId,
                    user_id: userId,
                    report_content: reportContent,
                });
            } catch (err) {
                console.error('Failed to save report to Supabase:', err);
            }
        }
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ceras_report_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="dashboard">

            {/* ===== HERO SCORE CARD ===== */}
            <div className="dash-hero" ref={heroRef}>
                <div className="hero-gauge">
                    <svg viewBox="0 0 200 200">
                        {/* Outer ring background */}
                        <circle className="ring-outer-bg" cx="100" cy="100" r={outerR} />
                        {/* Outer ring: CEPM arc (blue) */}
                        <circle
                            className="ring-outer-cepm"
                            cx="100" cy="100" r={outerR}
                            style={{
                                stroke: '#3b82f6',
                                strokeDasharray: `${cepmArc} ${outerCirc - cepmArc}`,
                                strokeDashoffset: 0,
                                transform: 'rotate(-90deg)',
                                transformOrigin: '100px 100px',
                            }}
                        />
                        {/* Outer ring: CNN arc (pink) */}
                        <circle
                            className="ring-outer-cnn"
                            cx="100" cy="100" r={outerR}
                            style={{
                                stroke: '#ec4899',
                                strokeDasharray: `${cnnArc} ${outerCirc - cnnArc}`,
                                strokeDashoffset: -cepmArc,
                                transform: 'rotate(-90deg)',
                                transformOrigin: '100px 100px',
                            }}
                        />

                        {/* Inner ring background */}
                        <circle className="ring-bg" cx="100" cy="100" r={innerR} />
                        {/* Inner ring: fused CE score (animated) */}
                        <circle
                            className="ring-fill"
                            cx="100" cy="100" r={innerR}
                            style={{
                                stroke: scoreColor(result.fused_score),
                                strokeDasharray: innerCirc,
                                strokeDashoffset: innerOffset,
                                transform: 'rotate(-90deg)',
                                transformOrigin: '100px 100px',
                            }}
                        />
                    </svg>
                    <div className="hero-gauge-center">
                        <div className="hero-score-value" style={{ color: scoreColor(result.fused_score) }}>
                            {(result.fused_score * 100).toFixed(0)}
                        </div>
                        <div className="hero-score-label">CE Score</div>
                        <div className={`hero-score-delta ${delta >= 0 ? 'positive' : 'negative'}`}>
                            {deltaSign}{(delta * 100).toFixed(0)} from baseline
                        </div>
                        <div className="hero-score-percentile">
                            {getPercentile(result.fused_score)}
                        </div>
                    </div>
                </div>

                <div className="hero-info">
                    <div className={`hero-readiness ${readinessClass(result.readiness)}`}>
                        {result.readiness}
                    </div>

                    <div className="hero-sub-scores">
                        <div className="hero-sub">
                            <div className="hero-sub-value" style={{ color: scoreColor(result.cepm_score) }}>
                                {(result.cepm_score * 100).toFixed(0)}%
                            </div>
                            <div className="hero-sub-label">Structural (CEPM)</div>
                        </div>
                        <div className="hero-sub">
                            <div className="hero-sub-value" style={{ color: scoreColor(result.cnn_score) }}>
                                {(result.cnn_score * 100).toFixed(0)}%
                            </div>
                            <div className="hero-sub-label">Semantic (CNN)</div>
                        </div>
                        <div className="hero-sub">
                            <div className="hero-sub-value" style={{ color: scoreColor(result.confidence) }}>
                                {(result.confidence * 100).toFixed(0)}%
                            </div>
                            <div className="hero-sub-label">Confidence</div>
                        </div>
                    </div>

                    {/* Score bars */}
                    <div className="ce-bars">
                        {[
                            ['Fused CE', result.fused_score],
                            ['Structural', result.cepm_score],
                            ['Semantic', result.cnn_score],
                        ].map(([label, val]) => (
                            <div className="ce-bar-row" key={label}>
                                <span className="ce-bar-name">{label}</span>
                                <div className="ce-bar-track">
                                    <div className={`ce-bar-fill ${barClass(val)}`} style={{ width: `${val * 100}%` }} />
                                </div>
                                <span className="ce-bar-pct" style={{ color: scoreColor(val) }}>
                                    {(val * 100).toFixed(0)}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* ===== METRICS STRIP ===== */}
            <div className="dash-metrics">
                {[
                    [result.formulation_time?.toFixed(1) + 's', 'Formulation'],
                    [result.runtime?.toFixed(1) + 's', 'Processing'],
                    [result.total_tokens, 'Est. Tokens'],
                    [result.feature_count, 'Features'],
                    [result.llm_calls_used || '—', 'LLM Calls'],
                ].map(([val, label]) => (
                    <div className="dash-metric" key={label}>
                        <div className="dash-metric-value">{val}</div>
                        <div className="dash-metric-label">{label}</div>
                    </div>
                ))}
            </div>

            {/* ===== DIAGNOSTICS ===== */}
            <div className="dash-diagnostics">
                <div className="diag-card">
                    <h3><span className="diag-icon">✅</span> Strengths</h3>
                    <ul>
                        {result.strengths?.map((s, i) => (
                            <li key={i}><span className="diag-icon">•</span> {s}</li>
                        ))}
                    </ul>
                </div>
                <div className="diag-card">
                    <h3><span className="diag-icon">💡</span> Suggestions</h3>
                    <ul>
                        {result.suggestions?.map((s, i) => (
                            <li key={i}><span className="diag-icon">→</span> {s}</li>
                        ))}
                    </ul>
                </div>
            </div>

            {/* ===== LEARNING STEPS ===== */}
            <div className="dash-steps">
                <h2>📚 Learning Response</h2>
                {result.final_steps?.map((step, i) => (
                    <div className="dash-step" key={i} style={{ animationDelay: `${i * 0.06}s` }}>
                        <div className="step-num">{i + 1}</div>
                        <div className="step-text">{step}</div>
                    </div>
                ))}
            </div>

            {/* ===== ADAPTIVE RESPONSE ===== */}
            <div className="dash-adaptive">
                <h2>🎯 Adaptive Learning Summary</h2>
                {adaptiveLoading ? (
                    <div className="adaptive-spinner">
                        <div className="dot-pulse" />
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

            {/* ===== CE EXPLANATION ===== */}
            <Collapse title="❓ What is the Fused CE Score?">
                <div className="ce-explain">
                    <p>The <strong>Fused CE Score</strong> reflects how efficiently you are learning. It combines:</p>
                    <p>• <strong>Structural (CEPM)</strong> — Prompt complexity, depth of constraints<br />
                        • <strong>Semantic (CNN)</strong> — Intent clarity, behavioral alignment</p>
                    <p><strong>0.00–0.44 → Foundation Building</strong> — Revisit core concepts and slow down.</p>
                    <p><strong>0.45–0.59 → Developing Momentum</strong> — Engaging well, some inconsistencies.</p>
                    <p><strong>0.60–0.74 → Progressing Confidently</strong> — Stable understanding and engagement.</p>
                    <p><strong>0.75–1.00 → Peak Learning State</strong> — Strong clarity, alignment, and efficiency.</p>
                    <p><em>This score reflects learning efficiency — not intelligence — and adapts in real time.</em></p>
                </div>
            </Collapse>

            {/* ===== TELEMETRY ===== */}
            <Collapse title="📡 Live Telemetry & Diagnostics">
                <div className="telemetry-split">
                    <div className="telemetry-block">
                        <h4>Extracted Features</h4>
                        <pre className="telemetry-pre">{JSON.stringify(result.features, null, 2)}</pre>
                    </div>
                    <div className="telemetry-block">
                        <h4>System Diagnostics</h4>
                        <pre className="telemetry-pre">{JSON.stringify(result.diagnostics, null, 2)}</pre>
                    </div>
                </div>
            </Collapse>

            {/* ===== REASONING TRACE ===== */}
            <Collapse title="🔍 Reasoning Trace">
                <ReasoningTree tree={result.tree} />
                <div className="trace-divider" />
                <pre className="trace-pre">{result.logs || 'No logs available.'}</pre>
            </Collapse>

            {/* ===== SOCRATIC FOLLOW-UP ===== */}
            <FollowUpChat
                result={result}
                prompt={prompt}
                config={config}
                onCostUpdate={setFollowupCost}
                sessionId={sessionId}
                messageId={messageId}
                userId={userId}
            />

            {/* ===== ACTIONS ===== */}
            <div className="dash-actions">
                <button className="dash-download-btn" onClick={downloadReport}>
                    📥 Download Session Report
                </button>
                <button
                    className="dash-download-btn dash-plan-btn"
                    onClick={() => setShowWorkflow(true)}
                >
                    🗺 Generate Learning Plan
                </button>
            </div>

            {/* Cumulative follow-up cost */}
            {followupCost.tokens > 0 && (
                <div className="dash-followup-cost">
                    Follow-up: {followupCost.tokens.toLocaleString()} tokens · ${followupCost.cost.toFixed(6)}
                </div>
            )}

            {/* ===== WORKFLOW MODAL ===== */}
            {showWorkflow && (
                <WorkflowModal
                    result={result}
                    prompt={prompt}
                    config={config}
                    onClose={() => setShowWorkflow(false)}
                    onCostUpdate={setPlanCost}
                    sessionId={sessionId}
                    messageId={messageId}
                    userId={userId}
                />
            )}
        </div>
    );
}
