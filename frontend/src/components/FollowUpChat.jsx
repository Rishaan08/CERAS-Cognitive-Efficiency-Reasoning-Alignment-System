import { useState, useRef, useEffect } from 'react';
import { sendFollowUp } from '../api';
import './FollowUpChat.css';

export default function FollowUpChat({ result, prompt, config, onCostUpdate }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const [collapsed, setCollapsed] = useState(false);
    const [totalCost, setTotalCost] = useState(0);
    const [totalTokens, setTotalTokens] = useState(0);
    const endRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async () => {
        const text = input.trim();
        if (!text || sending) return;

        const userMsg = { role: 'user', content: text };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setSending(true);

        try {
            const data = await sendFollowUp({
                message: text,
                context: {
                    prompt,
                    steps: result?.final_steps || [],
                    ce_score: result?.fused_score || 0.5,
                },
                history: [...messages, userMsg],
                main_provider: config.main_provider,
                main_model: config.main_model,
                groq_api_key: config.groq_api_key,
                gemini_api_key: config.gemini_api_key,
                openai_api_key: config.openai_api_key,
            });

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.response,
                tokens: data.total_tokens,
                cost: data.cost_usd,
            }]);

            const newTotalTokens = totalTokens + data.total_tokens;
            const newTotalCost = totalCost + data.cost_usd;
            setTotalTokens(newTotalTokens);
            setTotalCost(newTotalCost);
            if (onCostUpdate) onCostUpdate({ tokens: newTotalTokens, cost: newTotalCost });
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `⚠ Error: ${err.message}`,
                tokens: 0,
                cost: 0,
            }]);
        } finally {
            setSending(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="followup-chat">
            <div className="followup-header" onClick={() => setCollapsed(!collapsed)}>
                <div className="followup-title">
                    <span className="followup-icon">🧠</span>
                    <span>Socratic Follow-Up</span>
                    <span className="followup-badge">{messages.length} msgs</span>
                </div>
                <div className="followup-meta">
                    {totalTokens > 0 && (
                        <span className="followup-cost">
                            {totalTokens.toLocaleString()} tokens · ${totalCost.toFixed(6)}
                        </span>
                    )}
                    <span className="followup-collapse">{collapsed ? '▸' : '▾'}</span>
                </div>
            </div>

            {!collapsed && (
                <>
                    <div className="followup-messages">
                        {messages.length === 0 && (
                            <div className="followup-empty">
                                Ask a follow-up question — the AI will guide you with probing questions, never giving direct answers.
                            </div>
                        )}
                        {messages.map((msg, i) => (
                            <div key={i} className={`followup-msg followup-msg-${msg.role}`}>
                                <div className="followup-msg-label">
                                    {msg.role === 'user' ? '🧑 You' : '🤖 Mentor'}
                                </div>
                                <div className="followup-msg-text">{msg.content}</div>
                                {msg.tokens > 0 && (
                                    <div className="followup-msg-meta">
                                        {msg.tokens} tokens · ${msg.cost?.toFixed(6)}
                                    </div>
                                )}
                            </div>
                        ))}
                        {sending && (
                            <div className="followup-msg followup-msg-assistant">
                                <div className="followup-msg-label">🤖 Mentor</div>
                                <div className="followup-typing">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        )}
                        <div ref={endRef} />
                    </div>

                    <div className="followup-input-row">
                        <input
                            className="followup-input"
                            type="text"
                            placeholder="Ask a follow-up question..."
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            disabled={sending}
                        />
                        <button
                            className="followup-send-btn"
                            onClick={handleSend}
                            disabled={sending || !input.trim()}
                        >
                            {sending ? '...' : '→'}
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}
