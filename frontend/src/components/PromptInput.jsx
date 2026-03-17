import { useRef, useState } from 'react';
import { parseFile } from '../api';
import './PromptInput.css';

const ALLOWED_TYPES = ['.pdf', '.docx', '.txt', '.csv', '.md'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export default function PromptInput({
    prompt, setPrompt, onRun, onNewProblem, loading,
    modelsLoaded, modelError, onKeyDown, onPaste,
    onFileContent
}) {
    const [attachedFiles, setAttachedFiles] = useState([]);
    const [parsing, setParsing] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const fileRef = useRef(null);

    const getButtonText = () => {
        if (loading) return '⏳ Running Reasoning Engine...';
        if (modelError) return '⚠ Model Loading Error';
        if (!modelsLoaded) return '⏳ Models Loading...';
        return '▶ Run Learning Session';
    };

    const handlePaste = (e) => {
        const pastedText = e.clipboardData?.getData('text') || '';
        if (pastedText && onPaste) {
            setTimeout(() => onPaste(pastedText), 0);
        }
    };

    const processFile = async (file) => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_TYPES.includes(ext)) {
            alert(`Unsupported file type: ${ext}. Allowed: ${ALLOWED_TYPES.join(', ')}`);
            return;
        }
        if (file.size > MAX_FILE_SIZE) {
            alert('File too large. Max 10MB.');
            return;
        }

        setParsing(true);
        try {
            const data = await parseFile(file);
            setAttachedFiles(prev => [...prev, {
                name: data.filename || file.name,
                chars: data.chars,
                text: data.text,
            }]);
            if (onFileContent) onFileContent(data.text);
        } catch (err) {
            alert(`Failed to parse file: ${err.message}`);
        } finally {
            setParsing(false);
        }
    };

    const handleFileSelect = (e) => {
        const files = Array.from(e.target.files || []);
        files.forEach(processFile);
        if (fileRef.current) fileRef.current.value = '';
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        const files = Array.from(e.dataTransfer.files || []);
        files.forEach(processFile);
    };

    const removeFile = (index) => {
        setAttachedFiles(prev => prev.filter((_, i) => i !== index));
    };

    return (
        <div className="prompt-input-section">
            <hr />
            <div className="prompt-actions-top">

                <input
                    ref={fileRef}
                    type="file"
                    accept=".pdf,.docx,.txt,.csv,.md"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                    multiple
                />
                <button className="new-problem-btn" onClick={onNewProblem}>
                    ✦ New Problem
                </button>
            </div>

            {/* File chips */}
            {attachedFiles.length > 0 && (
                <div className="attached-files">
                    {attachedFiles.map((f, i) => (
                        <div key={i} className="file-chip">
                            <span className="file-chip-icon">📄</span>
                            <span className="file-chip-name">{f.name}</span>
                            <span className="file-chip-size">{f.chars.toLocaleString()} chars</span>
                            <button className="file-chip-remove" onClick={() => removeFile(i)}>✕</button>
                        </div>
                    ))}
                </div>
            )}

            <div
                className={`prompt-textarea-wrapper${dragOver ? ' drag-over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
            >
                <textarea
                    className="prompt-textarea"
                    placeholder="Enter your learning question or problem..."
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    onKeyDown={onKeyDown}
                    onPaste={handlePaste}
                    rows={6}
                />
                {dragOver && (
                    <div className="drag-overlay">
                        <span>📎 Drop file here</span>
                    </div>
                )}
            </div>

            <button
                className={`run-btn${modelError ? ' run-btn-error' : ''}`}
                onClick={onRun}
                disabled={loading || !prompt.trim() || !modelsLoaded}
            >
                {getButtonText()}
            </button>
            {modelError && (
                <div className="model-error-msg">
                    <span>⚠</span> {modelError}
                    <span className="error-hint"> — Retrying automatically...</span>
                </div>
            )}
        </div>
    );
}
