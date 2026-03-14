import { useState } from 'react';
import { generatePlan } from '../api';
import './WorkflowModal.css';

export default function WorkflowModal({ result, prompt, config, onClose, onCostUpdate }) {
    const [plan, setPlan] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [copied, setCopied] = useState(false);

    const handleGenerate = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await generatePlan({
                prompt,
                steps: result?.final_steps || [],
                ce_score: result?.fused_score || 0.5,
                diagnostics: result?.diagnostics || {},
                main_provider: config.main_provider,
                main_model: config.main_model,
                groq_api_key: config.groq_api_key,
                gemini_api_key: config.gemini_api_key,
                openai_api_key: config.openai_api_key,
            });
            setPlan(data.plan);
            if (onCostUpdate) {
                onCostUpdate({ tokens: data.total_tokens, cost: data.cost_usd });
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleCopy = () => {
        if (plan) {
            navigator.clipboard.writeText(plan);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const handleDownload = () => {
        if (!plan) return;
        const blob = new Blob([plan], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'learning_plan.md';
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="workflow-overlay" onClick={onClose}>
            <div className="workflow-modal" onClick={e => e.stopPropagation()}>
                <div className="workflow-modal-header">
                    <div className="workflow-modal-title">
                        <span>🗺</span> Learning Action Plan
                    </div>
                    <button className="workflow-close-btn" onClick={onClose}>✕</button>
                </div>

                <div className="workflow-modal-body">
                    {!plan && !loading && !error && (
                        <div className="workflow-intro">
                            <p>Generate a structured 2-week learning plan based on your session results.</p>
                            <p className="workflow-intro-sub">
                                The plan includes phases, tasks, milestones, and resources tailored to your topic.
                            </p>
                            <button
                                className="workflow-generate-btn"
                                onClick={handleGenerate}
                            >
                                ⚡ Generate Plan
                            </button>
                        </div>
                    )}

                    {loading && (
                        <div className="workflow-loading">
                            <div className="workflow-spinner" />
                            <p>Generating your learning plan...</p>
                        </div>
                    )}

                    {error && (
                        <div className="workflow-error">
                            <p>⚠ {error}</p>
                            <button className="workflow-retry-btn" onClick={handleGenerate}>
                                Retry
                            </button>
                        </div>
                    )}

                    {plan && (
                        <div className="workflow-plan">
                            <div className="workflow-plan-actions">
                                <button className="workflow-action-btn" onClick={handleCopy}>
                                    {copied ? '✓ Copied' : '📋 Copy'}
                                </button>
                                <button className="workflow-action-btn" onClick={handleDownload}>
                                    💾 Download .md
                                </button>
                                <button className="workflow-action-btn" onClick={handleGenerate}>
                                    🔄 Regenerate
                                </button>
                            </div>
                            <div className="workflow-plan-content">
                                <pre>{plan}</pre>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
