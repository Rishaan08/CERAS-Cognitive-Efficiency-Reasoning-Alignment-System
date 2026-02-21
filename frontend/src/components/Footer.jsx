import './Footer.css';

export default function Footer() {
    const year = new Date().getFullYear();
    return (
        <footer className="app-footer">
            <div>© {year} CERAS — Cognitive Efficiency & Reasoning Alignment System</div>
            <div className="footer-names">
                <span>Developed by <a href="https://github.com/Wolfie8935" target="_blank" rel="noopener noreferrer">Aman Goel</a></span>
                <span>&</span>
                <span><a href="https://github.com/Rishaan08" target="_blank" rel="noopener noreferrer">Rishaan Yadav</a></span>
            </div>
        </footer>
    );
}
