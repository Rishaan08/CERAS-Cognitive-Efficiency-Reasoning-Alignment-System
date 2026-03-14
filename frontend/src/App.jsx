import { useEffect, useRef, useState } from 'react';
import { checkHealth, runSession } from './api';
import './App.css';
import Dashboard from './components/Dashboard';
import ExampleCards from './components/ExampleCards';
import Footer from './components/Footer';
import Header from './components/Header';
import HistoryPanel from './components/HistoryPanel';
import LivePromptScore from './components/LivePromptScore';
import LoadingOverlay from './components/LoadingOverlay';
import PromptGuide from './components/PromptGuide';
import PromptInput from './components/PromptInput';
import Sidebar from './components/Sidebar';
import VaultPage from './components/VaultPage';
import { useAuth } from './context/AuthContext';
import { GROQ_MODELS } from './data/examples';
import useTypingAnalytics from './hooks/useTypingAnalytics';
import useVault from './hooks/useVault';
import { saveSession } from './lib/saveSession';
import LoginPage from './pages/LoginPage';

export default function App() {
    const { user, loading: authLoading } = useAuth();

    const [config, setConfig] = useState({
        main_provider: 'Groq',
        verifier_provider: 'Groq',
        main_model: GROQ_MODELS[0],
        verifier_model: GROQ_MODELS[1],
        groq_api_key: '',
        gemini_api_key: '',
        openai_api_key: '',
    });

    const [prompt, setPrompt] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [hasResult, setHasResult] = useState(false);
    const [modelsLoaded, setModelsLoaded] = useState(false);
    const [modelError, setModelError] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [activePanel, setActivePanel] = useState(null); // 'history' | 'vault' | null
    const startTimeRef = useRef(Date.now());

    const { analytics, onKeyDown, simulateFromPaste, reset: resetAnalytics } = useTypingAnalytics();
    const vault = useVault(user?.id);

    // Load saved API keys from vault on login
    useEffect(() => {
        if (user && vault.keys.length > 0) {
            setConfig(prev => ({
                ...prev,
                groq_api_key: vault.getKeyForProvider('Groq') || prev.groq_api_key,
                gemini_api_key: vault.getKeyForProvider('Gemini') || prev.gemini_api_key,
                openai_api_key: vault.getKeyForProvider('OpenAI') || prev.openai_api_key,
            }));
        }
    }, [user, vault.keys]);

    // Poll model health
    useEffect(() => {
        let alive = true;
        const poll = async () => {
            try {
                const data = await checkHealth();
                if (!alive) return;
                setModelsLoaded(data.models_loaded);
                setModelError(data.model_error || null);
                if (!data.models_loaded) {
                    setTimeout(poll, data.model_error ? 5000 : 2000);
                }
            } catch {
                if (alive) {
                    setModelError('Backend server is not reachable');
                    setTimeout(poll, 3000);
                }
            }
        };
        poll();
        return () => { alive = false; };
    }, []);

    const handleRun = async () => {
        if (!prompt.trim() || loading) return;
        setLoading(true);
        setResult(null);
        setHasResult(true);

        const formulationTime = (Date.now() - startTimeRef.current) / 1000;

        try {
            const data = await runSession({
                prompt: prompt.trim(),
                main_provider: config.main_provider,
                verifier_provider: config.verifier_provider,
                main_model: config.main_model,
                verifier_model: config.verifier_model,
                groq_api_key: config.groq_api_key,
                gemini_api_key: config.gemini_api_key,
                openai_api_key: config.openai_api_key,
                formulation_time: formulationTime,
            });
            setResult(data);

            // Save session to Supabase
            if (user) {
                saveSession({
                    userId: user.id,
                    prompt: prompt.trim(),
                    result: data,
                    config,
                    typingAnalytics: analytics,
                }).catch(err => console.error('Session save failed:', err));
            }
        } catch (err) {
            setResult({
                final_steps: [`Error: ${err.message}`],
                runtime: 0,
                formulation_time: formulationTime,
                fused_score: 0,
                cepm_score: 0,
                cnn_score: 0,
                confidence: 0,
                readiness: 'Error',
                diagnostics: {},
                features: {},
                feature_count: 0,
                total_tokens: 0,
                strengths: [],
                suggestions: ['An error occurred. Please check your API keys and try again.'],
                logs: err.message,
            });
        } finally {
            setLoading(false);
        }
    };

    const handleNewProblem = () => {
        setPrompt('');
        setResult(null);
        setHasResult(false);
        resetAnalytics();
        startTimeRef.current = Date.now();
    };

    const handleSelectExample = (text) => {
        setPrompt(text);
        setResult(null);
        setHasResult(false);
        simulateFromPaste(text);
    };

    const handleSelectSession = ({ prompt: p, result: r, config: c }) => {
        setPrompt(p);
        setResult(r);
        setHasResult(true);
        if (c) {
            setConfig(prev => ({ ...prev, ...c }));
        }
    };

    // Show loading spinner while auth state initializes
    if (authLoading) {
        return (
            <div className="app-auth-loading">
                <div className="auth-spinner" />
                <p>Loading...</p>
            </div>
        );
    }

    // Show login page if not authenticated
    if (!user) {
        return <LoginPage />;
    }

    return (
        <div className="app-layout">
            {/* Mobile hamburger toggle */}
            <button
                className="mobile-menu-btn"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
            >
                ☰
            </button>

            {/* Backdrop overlay for mobile */}
            {sidebarOpen && (
                <div
                    className="sidebar-backdrop"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            <Sidebar
                config={config}
                setConfig={setConfig}
                isOpen={sidebarOpen}
                onClose={() => setSidebarOpen(false)}
                user={user}
                onOpenHistory={() => setActivePanel('history')}
                onOpenVault={() => setActivePanel('vault')}
            />

            <main className="main-content">
                <Header
                    user={user}
                    onOpenHistory={() => setActivePanel('history')}
                    onOpenVault={() => setActivePanel('vault')}
                />
                <PromptGuide />
                <ExampleCards onSelect={handleSelectExample} />

                <PromptInput
                    prompt={prompt}
                    setPrompt={setPrompt}
                    onRun={handleRun}
                    onNewProblem={handleNewProblem}
                    loading={loading}
                    modelsLoaded={modelsLoaded}
                    modelError={modelError}
                    onKeyDown={onKeyDown}
                    onPaste={simulateFromPaste}
                    onFileContent={(text) => {
                        const sep = prompt.trim() ? '\n\n---\n[File Content]:\n' : '';
                        setPrompt(prev => prev + sep + text.slice(0, 4000));
                    }}
                />

                {/* Live Prompt Score — appears below textarea */}
                <LivePromptScore prompt={prompt} analytics={analytics} />

                {/* Dashboard — replaces old ResultsPanel */}
                <Dashboard
                    result={result}
                    prompt={prompt}
                    config={config}
                    hasResult={hasResult}
                    typingAnalytics={analytics}
                />

                <Footer />
            </main>

            {loading && <LoadingOverlay />}

            {/* Slide-over panels */}
            {activePanel === 'history' && (
                <div className="slide-panel-overlay" onClick={() => setActivePanel(null)}>
                    <div className="slide-panel" onClick={(e) => e.stopPropagation()}>
                        <HistoryPanel
                            userId={user.id}
                            onSelectSession={handleSelectSession}
                            onClose={() => setActivePanel(null)}
                        />
                    </div>
                </div>
            )}

            {activePanel === 'vault' && (
                <div className="slide-panel-overlay" onClick={() => setActivePanel(null)}>
                    <div className="slide-panel" onClick={(e) => e.stopPropagation()}>
                        <VaultPage
                            vault={vault}
                            config={config}
                            setConfig={setConfig}
                            onClose={() => setActivePanel(null)}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
