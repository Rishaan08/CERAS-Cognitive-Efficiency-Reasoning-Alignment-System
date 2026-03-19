import './ReasoningTree.css';

const ROLE_LABELS = {
    root: 'Prompt',
    strategy: 'Strategy',
    candidate_step: 'Draft Step',
    review: 'Review',
    final_step: 'Final Step',
    subtask: 'Step',
};

function getRoleLabel(role) {
    return ROLE_LABELS[role] || 'Node';
}

function getTone(node) {
    const metadata = node.metadata || {};
    if (metadata.selected) return 'selected';
    if (metadata.verification_status === 'accepted') return 'accepted';
    if (metadata.verification_status === 'rejected') return 'rejected';
    if (node.role === 'review') return 'review';
    if (node.role === 'final_step') return 'final';
    return 'default';
}

function getBadges(node) {
    const metadata = node.metadata || {};
    const badges = [getRoleLabel(node.role)];

    if (typeof metadata.step_number === 'number') {
        badges.push(`Step ${metadata.step_number}`);
    }
    if (metadata.verification_status === 'accepted') {
        badges.push('Accepted');
    } else if (metadata.verification_status === 'rejected') {
        badges.push('Rejected');
    }
    if (metadata.selected) {
        badges.push('Selected Path');
    }
    if (metadata.status === 'final') {
        badges.push('Final');
    }

    return badges;
}

function TreeBranch({ nodeId, nodes }) {
    const node = nodes?.[nodeId];

    if (!node) return null;

    return (
        <div className="reasoning-branch">
            <div className={`reasoning-node ${getTone(node)}`}>
                <div className="reasoning-node-badges">
                    {getBadges(node).map(badge => (
                        <span className="reasoning-badge" key={`${node.id}-${badge}`}>{badge}</span>
                    ))}
                </div>
                <div className="reasoning-node-text">{node.text}</div>
                {node.metadata?.verification_message && (
                    <div className="reasoning-node-meta">{node.metadata.verification_message}</div>
                )}
            </div>

            {node.children?.length > 0 && (
                <div className="reasoning-children">
                    {node.children.map(childId => (
                        <TreeBranch key={childId} nodeId={childId} nodes={nodes} />
                    ))}
                </div>
            )}
        </div>
    );
}

export default function ReasoningTree({ tree }) {
    if (!tree?.root_id || !tree?.nodes?.[tree.root_id]) {
        return <div className="reasoning-tree-empty">Reasoning tree unavailable for this session.</div>;
    }

    return (
        <div className="reasoning-tree-shell">
            <div className="reasoning-tree-caption">
                Candidate strategies branch from the prompt, then the selected path is marked after verification and final review.
            </div>
            <TreeBranch nodeId={tree.root_id} nodes={tree.nodes} />
        </div>
    );
}
