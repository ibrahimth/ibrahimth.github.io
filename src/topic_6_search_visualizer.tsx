import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Play, RotateCcw, Plus, Trash2, Settings,
    MousePointer2, ChevronRight, FastForward,
    Info, CheckCircle, AlertCircle, List, Table, Shuffle
} from 'lucide-react';

// --- Types ---
type Algorithm = 'BFS' | 'DFS' | 'UCS' | 'AStar' | 'MiniMax';
type InteractionMode = 'select' | 'node' | 'edge';
type NodeType = { id: string; x: number; y: number };
type EdgeType = { from: string; to: string; weight: number };
type ToastType = { id: number; message: string; type: 'success' | 'error' | 'info' };

type QueueItem = {
    path: string[];
    cost: number; // g for UCS/A*, 0 for BFS/DFS
    heuristic: number; // h for A*
    total: number; // f for A*
};

type StepLog = {
    step: number;
    queue: QueueItem[];
    extendedNode: string | null;
    extendedList: string[];
    extendedCount: number;
    enqueuedCount: number;
};

// --- Constants ---
const NODE_RADIUS = 24;
const COLORS = {
    bg: '#ffffff',
    node: '#f1f5f9', // Slate 100
    nodeBorder: '#334155', // Slate 700
    nodeSelected: '#2563eb', // Blue 600
    nodeStart: '#059669',    // Emerald 600 (Darker)
    nodeGoal: '#dc2626',     // Red 600 (Darker)
    edge: '#64748b',         // Slate 500
    visited: '#fb923c',      // Orange 400 (Darker for print)
    queued: '#818cf8',       // Indigo 400 (Darker for print)
    path: '#7c3aed',         // Violet 600 (Darker)
    text: '#0f172a',    // Slate 900
    accent: '#06b6d4',
    minimaxValue: '#db2777',
    grid: '#e2e8f0',
    weightBadgeErr: '#ffffff',
    weightBadgeText: '#0f172a'
};

const GraphVisualizer = () => {
    // --- State ---
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Graph Data
    const [nodes, setNodes] = useState<NodeType[]>([]);
    const [edges, setEdges] = useState<EdgeType[]>([]);
    const [startNode, setStartNode] = useState<string | null>(null);
    const [goalNode, setGoalNode] = useState<string | null>(null);

    // UI State
    const [mode, setMode] = useState<InteractionMode>('select');
    const [selectedNode, setSelectedNode] = useState<string | null>(null);
    const [algorithm, setAlgorithm] = useState<Algorithm>('AStar');
    const [speed, setSpeed] = useState<number>(800);
    const [isAnimating, setIsAnimating] = useState(false);
    const [showHeuristics, setShowHeuristics] = useState(true);
    const [showIndirect, setShowIndirect] = useState(false); // Toggle for Dashed Lines
    const [treeSearch, setTreeSearch] = useState(false); // Default to Graph Search (No Duplicates)
    const [manualHeuristics, setManualHeuristics] = useState(false);
    const [isDirected, setIsDirected] = useState(true); // Toggle for Directed/Undirected
    const [rootIsMax, setRootIsMax] = useState(true); // Minimax Root Toggle
    const [showLogs, setShowLogs] = useState(true);
    const [toasts, setToasts] = useState<ToastType[]>([]);

    // Algorithm Visualization State
    const [visitedNodes, setVisitedNodes] = useState<Set<string>>(new Set()); // Closed Set
    const [frontierNodes, setFrontierNodes] = useState<Set<string>>(new Set()); // Open Set
    const [currentPath, setCurrentPath] = useState<string[]>([]);
    const [heuristics, setHeuristics] = useState<Record<string, number>>({});
    const [computedValues, setComputedValues] = useState<Record<string, number>>({}); // For MiniMax
    const [stepLogs, setStepLogs] = useState<StepLog[]>([]);

    // Interaction State
    const [dragState, setDragState] = useState<{
        type: 'node' | 'edge' | null;
        sourceId: string | null;
        currentPos: { x: number; y: number } | null;
    }>({ type: null, sourceId: null, currentPos: null });

    // --- Helpers ---
    const showToast = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
        const id = Date.now();
        setToasts(prev => [...prev, { id, message, type }]);
        setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3000);
    };

    const generateNodeId = () => {
        const existingIds = new Set(nodes.map(n => n.id));
        let id = 'A';
        let charCode = 65;
        while (existingIds.has(id)) {
            charCode++;
            id = String.fromCharCode(charCode);
        }
        return id;
    };

    const getDistance = (n1: NodeType, n2: NodeType) =>
        Math.sqrt(Math.pow(n1.x - n2.x, 2) + Math.pow(n1.y - n2.y, 2));

    // --- Heuristics ---
    useEffect(() => {
        if (!manualHeuristics && goalNode && algorithm !== 'MiniMax') {
            const target = nodes.find(n => n.id === goalNode);
            if (!target) return;

            const h: Record<string, number> = {};
            nodes.forEach(n => {
                // Simple Euclidean distance scaled down
                h[n.id] = Math.round(getDistance(n, target) / 10);
            });
            setHeuristics(h);
        }
    }, [goalNode, nodes, manualHeuristics, algorithm]);

    const updateHeuristic = (id: string, value: string) => {
        const num = parseFloat(value);
        setHeuristics(prev => ({ ...prev, [id]: !isNaN(num) ? num : 0 }));
    };

    const randomizeWeights = () => {
        if (isAnimating) return;
        setEdges(prev => prev.map(e => ({
            ...e,
            weight: Math.floor(Math.random() * 9) + 1 // 1 to 9
        })));
        showToast('Weights randomized', 'info');
    };

    const resetGraph = () => {
        setVisitedNodes(new Set());
        setFrontierNodes(new Set());
        setCurrentPath([]);
        setStepLogs([]);
        setComputedValues({});
        setIsAnimating(false);
    };

    const clearAll = () => {
        setNodes([]);
        setEdges([]);
        setStartNode(null);
        setGoalNode(null);
        setHeuristics({});
        resetGraph();
        showToast('Graph cleared', 'info');
    };

    const loadExample1 = () => {
        // Data from "A* Search - Example 1" slide
        // Nodes: S, A, B, C, D, E, G
        // Positional layout approximating the slide
        const newNodes: NodeType[] = [
            { id: 'S', x: 100, y: 350 },
            { id: 'A', x: 250, y: 350 },
            { id: 'B', x: 250, y: 200 },
            { id: 'C', x: 250, y: 50 },
            { id: 'D', x: 400, y: 350 },
            { id: 'E', x: 550, y: 50 },
            { id: 'G', x: 550, y: 200 },
        ];

        const newEdges: EdgeType[] = [
            { from: 'S', to: 'A', weight: 3 },
            { from: 'S', to: 'B', weight: 5 },
            { from: 'A', to: 'B', weight: 4 },
            { from: 'A', to: 'D', weight: 3 },
            { from: 'B', to: 'C', weight: 4 },
            { from: 'C', to: 'E', weight: 7 },
            { from: 'D', to: 'G', weight: 5 }
        ];

        setNodes(newNodes);
        setEdges(newEdges);
        setStartNode('S');
        setGoalNode('G');

        setManualHeuristics(true);
        setHeuristics({
            'S': 9.8,
            'A': 7.6,
            'B': 6.5,
            'C': 7.6,
            'D': 5.0,
            'E': 4.1,
            'G': 0
        });

        setShowIndirect(true); // Auto-enable indirect view
        setTreeSearch(true); // Ensure Tree Search (duplicates) is ON
        setIsDirected(false); // Example 1 is Undirected
        setAlgorithm('AStar');
        resetGraph();
        showToast('Loaded Example 1', 'success');
    };

    const loadExample3 = () => {
        // Data from "A* Search - Example 3" slide
        // Nodes: S, A, B, C, G
        const newNodes: NodeType[] = [
            { id: 'S', x: 100, y: 200 },
            { id: 'A', x: 250, y: 100 },
            { id: 'B', x: 250, y: 300 },
            { id: 'C', x: 400, y: 200 },
            { id: 'G', x: 550, y: 200 },
        ];

        const newEdges: EdgeType[] = [
            { from: 'S', to: 'A', weight: 1 },
            { from: 'S', to: 'B', weight: 3 },
            { from: 'A', to: 'B', weight: 2 },
            { from: 'A', to: 'C', weight: 4 },
            { from: 'A', to: 'G', weight: 11 },
            { from: 'B', to: 'C', weight: 2 },
            { from: 'C', to: 'G', weight: 3 }
        ];

        setNodes(newNodes);
        setEdges(newEdges);
        setStartNode('S');
        setGoalNode('G');

        setManualHeuristics(true);
        setHeuristics({
            'S': 7,
            'A': 6,
            'B': 2,
            'C': 2,
            'G': 0
        });

        setShowIndirect(false);
        setTreeSearch(false); // Ensure Graph Search (No Duplicates) is ON
        setIsDirected(true); // Example 3 is Directed
        setAlgorithm('AStar');
        resetGraph();
        showToast('Loaded Example 3', 'success');
    };

    // --- Drawing ---
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
        }

        // Background
        ctx.clearRect(0, 0, rect.width, rect.height);

        // Grid
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x < rect.width; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); }
        for (let y = 0; y < rect.height; y += 40) { ctx.moveTo(0, y); ctx.lineTo(rect.width, y); }
        ctx.stroke();

        // Edges
        edges.forEach(edge => {
            const n1 = nodes.find(n => n.id === edge.from);
            const n2 = nodes.find(n => n.id === edge.to);
            if (!n1 || !n2) return;

            const isPathEdge = currentPath.length > 1 &&
                currentPath.some((nodeId, i) => {
                    if (i === currentPath.length - 1) return false;
                    return (nodeId === edge.from && currentPath[i + 1] === edge.to);
                });

            // Line
            ctx.beginPath();
            ctx.moveTo(n1.x, n1.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.strokeStyle = isPathEdge ? COLORS.path : COLORS.edge;
            ctx.lineWidth = isPathEdge ? 4 : 2;
            ctx.lineWidth = isPathEdge ? 4 : 2;
            ctx.stroke();

            // Arrow (Only if Directed)
            if (isDirected) {
                const angle = Math.atan2(n2.y - n1.y, n2.x - n1.x);
                const arrowLength = 12;
                const targetX = n2.x - NODE_RADIUS * Math.cos(angle);
                const targetY = n2.y - NODE_RADIUS * Math.sin(angle);

                ctx.beginPath();
                ctx.moveTo(targetX, targetY);
                ctx.lineTo(
                    targetX - arrowLength * Math.cos(angle - Math.PI / 6),
                    targetY - arrowLength * Math.sin(angle - Math.PI / 6)
                );
                ctx.lineTo(
                    targetX - arrowLength * Math.cos(angle + Math.PI / 6),
                    targetY - arrowLength * Math.sin(angle + Math.PI / 6)
                );
                ctx.fillStyle = isPathEdge ? COLORS.path : COLORS.edge;
                ctx.fill();
            }

            // Weight Badge
            if (algorithm !== 'MiniMax') {
                const midX = (n1.x + n2.x) / 2;
                const midY = (n1.y + n2.y) / 2;

                ctx.beginPath();
                ctx.arc(midX, midY, 14, 0, Math.PI * 2);
                ctx.fillStyle = COLORS.bg;
                ctx.fill();
                ctx.strokeStyle = isPathEdge ? COLORS.path : COLORS.edge;
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.fillStyle = COLORS.text;
                ctx.font = '12px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(edge.weight.toString(), midX, midY);
            }
        });

        // Drag Line
        if (dragState.type === 'edge' && dragState.sourceId && dragState.currentPos) {
            const source = nodes.find(n => n.id === dragState.sourceId);
            if (source) {
                ctx.beginPath();
                ctx.moveTo(source.x, source.y);
                ctx.lineTo(dragState.currentPos.x, dragState.currentPos.y);
                ctx.strokeStyle = COLORS.accent;
                ctx.setLineDash([5, 5]);
                ctx.lineWidth = 2;
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }

        // Nodes
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, NODE_RADIUS, 0, Math.PI * 2);

            let fill = COLORS.node;
            if (node.id === startNode) fill = COLORS.nodeStart;
            else if (node.id === goalNode) fill = COLORS.nodeGoal;
            else if (currentPath.includes(node.id)) fill = COLORS.path;
            else if (visitedNodes.has(node.id)) fill = COLORS.visited; // Extended
            else if (frontierNodes.has(node.id)) fill = COLORS.queued; // Frontier

            ctx.fillStyle = fill;
            ctx.fill();

            // Default Border for all nodes (to stand out against white bg)
            ctx.strokeStyle = COLORS.nodeBorder;
            ctx.lineWidth = 2;
            ctx.stroke();

            if (node.id === selectedNode) {
                ctx.strokeStyle = COLORS.nodeSelected;
                ctx.lineWidth = 3;
                ctx.stroke();
            }

            // Text Color Logic
            // If node is filled with dark color (Start/Goal/Path/Visited/Queued), use White text
            // If node is standard Light (Slate 100), use Dark text
            const isDarkNode = [COLORS.nodeStart, COLORS.nodeGoal, COLORS.path, COLORS.queued, COLORS.visited].includes(fill);

            // Special case for Extended/Frontier which are lightish colors?
            // Visited is Orange 200 (Light), Queued is Indigo 300 (Medium).
            // Let's safe bet: Start, Goal, Path are definitely dark/vibrant.
            // Visited/Queued might need checks. 
            // Actually, let's use dark text for everything except Start/Goal/Path.
            // But Visited/Queued are filled.

            if (node.id === startNode || node.id === goalNode || currentPath.includes(node.id)) {
                ctx.fillStyle = '#ffffff';
            } else {
                ctx.fillStyle = COLORS.text;
            }

            // Adapt font size for City names
            ctx.font = node.id.length > 2 ? 'bold 10px Inter, sans-serif' : 'bold 14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.id, node.x, node.y);

            if (showHeuristics && heuristics[node.id] !== undefined) {
                ctx.fillStyle = '#fbbf24';
                ctx.font = '11px Inter, sans-serif';
                ctx.fillText(`h:${heuristics[node.id]}`, node.x, node.y + NODE_RADIUS + 14);
            }

            // Computed Value (MiniMax)
            if (computedValues[node.id] !== undefined) {
                ctx.fillStyle = COLORS.minimaxValue;
                ctx.font = 'bold 11px Inter, sans-serif';
                ctx.fillText(`Val:${computedValues[node.id]}`, node.x, node.y - NODE_RADIUS - 5);
            }
        });

        // Indirect Costs (Heuristics) - Dashed Lines to Goal
        if (showIndirect && goalNode && heuristics) {
            const target = nodes.find(n => n.id === goalNode);
            if (target) {
                nodes.forEach(node => {
                    if (node.id === goalNode) return;

                    const hVal = heuristics[node.id];
                    if (hVal === undefined) return;

                    const dx = target.x - node.x;
                    const dy = target.y - node.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const midX = (node.x + target.x) / 2;
                    const midY = (node.y + target.y) / 2;

                    // Offset for curve (perpendicular vector)
                    // Keep roughly constant arc height (e.g., 40px) regardless of distance
                    const offset = 40;
                    const normX = -dy / dist;
                    const normY = dx / dist;
                    const cpX = midX + normX * offset;
                    const cpY = midY + normY * offset;

                    ctx.beginPath();
                    ctx.moveTo(node.x, node.y);
                    ctx.quadraticCurveTo(cpX, cpY, target.x, target.y);
                    ctx.strokeStyle = 'rgba(180, 83, 9, 0.8)'; // Dark Amber/Bronze (Printable)
                    ctx.setLineDash([5, 5]);
                    ctx.lineWidth = 1.5; // Slightly thicker
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Label position (t=0.5 on Quadratic Bezier)
                    // P_t = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
                    // t=0.5 => 0.25*P0 + 0.5*P1 + 0.25*P2
                    const labelX = 0.25 * node.x + 0.5 * cpX + 0.25 * target.x;
                    const labelY = 0.25 * node.y + 0.5 * cpY + 0.25 * target.y;

                    // Small background for text
                    ctx.beginPath();
                    ctx.arc(labelX, labelY, 10, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(180, 83, 9, 0.8)'; // Dark Amber border
                    ctx.lineWidth = 1;
                    ctx.stroke();

                    ctx.fillStyle = '#2563eb'; // Blue 600
                    ctx.font = 'italic 10px Inter, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(hVal.toString(), labelX, labelY);
                });
            }
        }
    }, [nodes, edges, startNode, goalNode, visitedNodes, frontierNodes, currentPath, selectedNode, dragState, showHeuristics, heuristics, computedValues]);

    useEffect(() => {
        let handle: number;
        const render = () => { draw(); handle = requestAnimationFrame(render); };
        render();
        return () => cancelAnimationFrame(handle);
    }, [draw]);

    // --- Interaction Handlers ---
    const handleMouseDown = (e: React.MouseEvent) => {
        if (isAnimating) return;
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // 1. Edge Click (Weight Edit)
        const clickedEdgeIndex = edges.findIndex(edge => {
            const n1 = nodes.find(n => n.id === edge.from);
            const n2 = nodes.find(n => n.id === edge.to);
            if (!n1 || !n2) return false;
            const midX = (n1.x + n2.x) / 2;
            const midY = (n1.y + n2.y) / 2;
            const dist = Math.sqrt(Math.pow(x - midX, 2) + Math.pow(y - midY, 2));
            return dist <= 20;
        });

        if (clickedEdgeIndex !== -1) {
            const newWeight = prompt("Enter new edge weight:", edges[clickedEdgeIndex].weight.toString());
            if (newWeight !== null && !isNaN(parseFloat(newWeight))) {
                setEdges(prev => {
                    const next = [...prev];
                    next[clickedEdgeIndex] = { ...next[clickedEdgeIndex], weight: parseFloat(newWeight) };
                    return next;
                });
            }
            return;
        }

        // 2. Node Click
        const clickedNode = nodes.find(n => getDistance(n, { id: 'temp', x, y }) < NODE_RADIUS);

        if (clickedNode) {
            if (mode === 'edge') {
                setDragState({ type: 'edge', sourceId: clickedNode.id, currentPos: { x, y } });
            } else {
                setSelectedNode(clickedNode.id);
                setDragState({ type: 'node', sourceId: clickedNode.id, currentPos: { x, y } });
            }
        } else {
            // Background Click
            setSelectedNode(null);
            if (mode === 'node') {
                setNodes([...nodes, { id: generateNodeId(), x, y }]);
            }
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Cursor Logic
        if (!dragState.type) {
            const isEdge = edges.some(edge => {
                const n1 = nodes.find(n => n.id === edge.from);
                const n2 = nodes.find(n => n.id === edge.to);
                if (!n1 || !n2) return false;
                const midX = (n1.x + n2.x) / 2;
                const midY = (n1.y + n2.y) / 2;
                return Math.sqrt(Math.pow(x - midX, 2) + Math.pow(y - midY, 2)) <= 20;
            });
            canvasRef.current!.style.cursor = isEdge ? 'pointer' : (mode === 'select' ? 'default' : 'crosshair');
        }

        if (!dragState.type) return;
        setDragState(prev => ({ ...prev, currentPos: { x, y } }));

        if (dragState.type === 'node' && dragState.sourceId) {
            setNodes(prev => prev.map(n => n.id === dragState.sourceId ? { ...n, x, y } : n));
        }
    };

    const handleMouseUp = (e: React.MouseEvent) => {
        if (dragState.type === 'edge' && dragState.sourceId) {
            const rect = canvasRef.current!.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const target = nodes.find(n => getDistance(n, { id: 'temp', x, y }) < NODE_RADIUS);

            if (target && target.id !== dragState.sourceId) {
                const exists = edges.some(e => e.from === dragState.sourceId && e.to === target.id);
                if (!exists) {
                    setEdges(prev => [...prev, { from: dragState.sourceId!, to: target.id, weight: 1 }]);
                    showToast('Edge created (weight: 1)', 'success');
                }
            }
        }
        setDragState({ type: null, sourceId: null, currentPos: null });
    };

    const handleDoubleClick = (e: React.MouseEvent) => {
        if (isAnimating) return;
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const clickedNode = nodes.find(n => getDistance(n, { id: 'temp', x, y }) < NODE_RADIUS);

        if (clickedNode) {
            // Minimax Value Editing Shortcut
            if (algorithm === 'MiniMax') {
                const currentVal = heuristics[clickedNode.id] || 0;
                const newVal = prompt(`Enter Value for Node ${clickedNode.id}:`, currentVal.toString());
                if (newVal !== null && !isNaN(parseFloat(newVal))) {
                    setHeuristics(prev => ({ ...prev, [clickedNode.id]: parseFloat(newVal) }));
                    showToast(`Set Node ${clickedNode.id} value to ${newVal}`, 'success');
                }
                return; // Skip renaming if we just set value
            }

            const newId = prompt('Enter new name for node:', clickedNode.id);
            if (newId && newId !== clickedNode.id) {
                // Check uniqueness
                if (nodes.some(n => n.id === newId)) {
                    showToast('Node name must be unique', 'error');
                    return;
                }

                // Update Node ID
                setNodes(prev => prev.map(n => n.id === clickedNode.id ? { ...n, id: newId } : n));

                // Update Edges
                setEdges(prev => prev.map(e => ({
                    ...e,
                    from: e.from === clickedNode.id ? newId : e.from,
                    to: e.to === clickedNode.id ? newId : e.to
                })));

                // Update Start/Goal
                if (startNode === clickedNode.id) setStartNode(newId);
                if (goalNode === clickedNode.id) setGoalNode(newId);

                // Update Heuristics key
                setHeuristics(prev => {
                    const next = { ...prev };
                    if (next[clickedNode.id] !== undefined) {
                        next[newId] = next[clickedNode.id];
                        delete next[clickedNode.id];
                    }
                    return next;
                });

                showToast(`Renamed ${clickedNode.id} to ${newId}`, 'success');
            }
        }
    };

    const handleContextMenu = (e: React.MouseEvent) => {
        e.preventDefault();
        if (isAnimating) return;
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const clicked = nodes.find(n => getDistance(n, { id: 'temp', x, y }) < NODE_RADIUS);

        if (clicked) {
            if (startNode === clicked.id) {
                setStartNode(null);
                setGoalNode(clicked.id);
            } else if (goalNode === clicked.id) {
                setGoalNode(null);
            } else {
                if (startNode) setStartNode(null);
                setStartNode(clicked.id);
            }
        }
    };

    const deleteSelected = () => {
        if (!selectedNode) return;
        setNodes(nodes.filter(n => n.id !== selectedNode));
        setEdges(edges.filter(e => e.from !== selectedNode && e.to !== selectedNode));
        if (startNode === selectedNode) setStartNode(null);
        if (goalNode === selectedNode) setGoalNode(null);
        setSelectedNode(null);
    };
    // --- Algorithm Implementation ---
    const runAlgorithm = async () => {
        if (!startNode) {
            showToast('Set Start node', 'error');
            return;
        }
        if (algorithm !== 'MiniMax' && !goalNode) {
            showToast('Set Goal node', 'error');
            return;
        }

        setIsAnimating(true);
        resetGraph();

        const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

        // --- MiniMax Implementation ---
        if (algorithm === 'MiniMax') {
            // zero-value check
            const allZero = nodes.every(n => (heuristics[n.id] || 0) === 0);
            if (allZero) {
                showToast('Warning: All node values are 0. Set Heuristics (Values) for leaf nodes!', 'error');
            }

            let stepCount = 0;
            const logStep = (nodeId: string, action: string, value?: number) => {
                stepCount++;
                setStepLogs(prev => [...prev, {
                    step: stepCount,
                    queue: value !== undefined ? [{ path: [nodeId], cost: value, heuristic: 0, total: value }] : [],
                    extendedNode: nodeId,
                    extendedList: [action],
                    extendedCount: stepCount,
                    enqueuedCount: stepCount
                }]);
            };

            // Dictionary to store the best move for each node to reconstruct path later if needed, 
            // but simpler is to return the path from the recursive function.

            type MinimaxResult = {
                val: number;
                bestPath: string[]; // Full path from Root to Leaf for this best outcome
            };

            const minimax = async (nodeId: string, depth: number, isMax: boolean, path: string[]): Promise<MinimaxResult> => {
                // Update specific path for visualization during traversal
                setCurrentPath(path);
                setFrontierNodes(prev => new Set(prev).add(nodeId));
                await sleep(speed);

                const neighbors = edges
                    .filter(e => e.from === nodeId)
                    .sort((a, b) => a.to.localeCompare(b.to)) // Lexical Order for MiniMax Children
                    .map(e => e.to);

                // Base Case: Leaf Node
                if (neighbors.length === 0) {
                    const val = heuristics[nodeId] || 0;
                    logStep(nodeId, `Leaf: ${val}`, val);
                    setVisitedNodes(prev => new Set(prev).add(nodeId));
                    setComputedValues(prev => ({ ...prev, [nodeId]: val }));
                    return { val, bestPath: path };
                }

                let bestVal = isMax ? -Infinity : Infinity;
                let bestPath: string[] = [...path]; // Default to current path if no valid children

                // If all children are cycles (shouldn't happen in strict tree, but good safety),
                // we treat this as a leaf with its own heuristic.
                let hasValidChild = false;

                for (const neighborId of neighbors) {
                    if (path.includes(neighborId)) continue; // Avoid cycles

                    hasValidChild = true;

                    const result = await minimax(neighborId, depth + 1, !isMax, [...path, neighborId]);

                    if (isMax) {
                        if (result.val > bestVal) {
                            bestVal = result.val;
                            bestPath = result.bestPath;
                        }
                    } else {
                        if (result.val < bestVal) {
                            bestVal = result.val;
                            bestPath = result.bestPath;
                        }
                    }
                }

                if (!hasValidChild || bestVal === -Infinity || bestVal === Infinity) {
                    const val = heuristics[nodeId] || 0;
                    logStep(nodeId, `Leaf (Cycle/Dead End): ${val}`, val);
                    setComputedValues(prev => ({ ...prev, [nodeId]: val }));
                    return { val, bestPath: path };
                }

                logStep(nodeId, `${isMax ? 'MAX' : 'MIN'} updated: ${bestVal}`, bestVal);
                setVisitedNodes(prev => new Set(prev).add(nodeId));
                setComputedValues(prev => ({ ...prev, [nodeId]: bestVal }));

                return { val: bestVal, bestPath };
            };

            const result = await minimax(startNode, 0, rootIsMax, [startNode]);

            // Final Visualization Update
            setCurrentPath(result.bestPath); // Highlight the OFFICIAL best path
            setComputedValues(prev => ({ ...prev, [startNode]: result.val })); // Ensure root value is set
            showToast(`MiniMax Result: ${result.val}`, 'success');
            setIsAnimating(false);
            return;
        }

        // Data Structures
        const openSet: { id: string; g: number; f: number; path: string[] }[] = [];
        const closedSet = new Map<string, number>(); // ID -> Min Cost (g)

        // Initialize
        const startH = heuristics[startNode] || 0;
        openSet.push({ id: startNode, g: 0, f: startH, path: [startNode] });

        let stepCount = 0;
        let extendedCount = 0;
        let enqueuedCount = 1;

        // --- Main Loop ---
        while (openSet.length > 0) {
            stepCount++;

            // 1. Sort Open Set
            if (algorithm === 'UCS') {
                openSet.sort((a, b) => {
                    if (a.g !== b.g) return a.g - b.g;
                    if (a.id !== b.id) return a.id.localeCompare(b.id);
                    return a.path.join('').localeCompare(b.path.join(''));
                });
            } else if (algorithm === 'AStar') {
                openSet.sort((a, b) => {
                    if (a.f !== b.f) return a.f - b.f;
                    if (a.id !== b.id) return a.id.localeCompare(b.id);
                    return a.path.join('').localeCompare(b.path.join(''));
                });
            }

            // 2. Logging State (Snapshot before pop)
            // For DFS (Stack), we want to show Top-to-Bottom.
            // openSet is [Bottom ... Top].
            // So reverse it for display.
            let queueToLog = openSet;
            if (algorithm === 'DFS') {
                queueToLog = [...openSet].reverse();
            }

            const currentQueueState: QueueItem[] = queueToLog.map(n => ({
                path: n.path,
                cost: n.g,
                heuristic: heuristics[n.id] || 0,
                total: n.f
            }));

            // 3. Pop
            let current;
            if (algorithm === 'DFS') current = openSet.pop()!;
            else current = openSet.shift()!; // BFS, UCS, AStar

            // Lazy Removal (Check if previously extended with lower or equal cost)
            if (!treeSearch && (algorithm === 'AStar' || algorithm === 'UCS')) {
                const previousCost = closedSet.get(current.id);
                // If we found a path that is strictly WORSE or EQUAL to a previous one, skip.
                if (previousCost !== undefined && previousCost <= current.g) {
                    // We still log the step (above) to show it was popped? 
                    // Actually, if we skip it, we shouldn't increment extendedCount or log it as "Extended".
                    // But the flowchart says "Delete Head" and loops back.
                    // The trace usually doesn't show skipped nodes as "Extended Steps".
                    // So we should probably continue immediately without logging this as a full step?
                    // BUT, the existing code logs "Snapshot before pop".
                    // Let's stick to existing pattern: Log it, but don't count it as "Extended Node" or "Expanded".
                    // Wait, the previous code logged it.
                    // Let's CONTINUE here.
                    continue;
                }
            }

            // Increment extended count (Only if we actually process it)
            extendedCount++;

            setStepLogs(prev => [...prev, {
                step: stepCount,
                queue: currentQueueState,
                extendedNode: current.id,
                extendedList: Array.from(closedSet.keys()),
                extendedCount: extendedCount,
                enqueuedCount: enqueuedCount
            }]);

            setFrontierNodes(new Set(openSet.map(n => n.id)));
            setCurrentPath(current.path);


            // 4. Goal Check
            if (current.id === goalNode) {
                // Update visual state one last time
                setVisitedNodes(new Set([...Array.from(closedSet.keys()), current.id]));
                showToast(`Goal Reached! Cost: ${current.g}`, 'success');
                setIsAnimating(false);
                return;
            }

            // 5. Expand (Update Closed Set)
            closedSet.set(current.id, current.g);
            setVisitedNodes(new Set(closedSet.keys()));
            await sleep(speed);

            // 6. Neighbors
            const neighbors = isDirected
                ? edges.filter(e => e.from === current.id)
                : edges.filter(e => e.from === current.id || e.to === current.id).map(e => e.to === current.id ? { ...e, from: e.to, to: e.from } : e);


            // Sort neighbors: 
            // DFS: Push to Stack in Descending Order (so Smallest/A is on Top and popped first). Ignore weights.
            // BFS: Enqueue in Ascending Order. Ignore weights.
            // UCS/A*: Enqueue by Weight then Ascending Order.
            if (algorithm === 'DFS') {
                neighbors.sort((a, b) => b.to.localeCompare(a.to)); // Z-A
            } else if (algorithm === 'BFS') {
                neighbors.sort((a, b) => a.to.localeCompare(b.to)); // A-Z
            } else {
                neighbors.sort((a, b) => {
                    if (a.weight !== b.weight) return a.weight - b.weight; // Low Weight first
                    return a.to.localeCompare(b.to); // A-Z
                });
            }

            for (const edge of neighbors) {
                const neighborId = edge.to;
                const tentativeG = current.g + edge.weight;
                const h = heuristics[neighborId] || 0;
                const f = tentativeG + h;
                const newPath = [...current.path, neighborId];

                // Tree Search vs Graph Search Check
                if (treeSearch) {
                    // Check Cycle
                    if (current.path.includes(neighborId)) continue;
                } else {
                    // Graph Search: Check Closed Set (Extended List)
                    // Flowchart: "Child previously extended with lower or equal cost? -> Yes: Delete child"
                    const prevCost = closedSet.get(neighborId);
                    if (prevCost !== undefined && prevCost <= tentativeG) {
                        continue;
                    }
                }

                // Enqueue
                if (treeSearch || algorithm === 'AStar' || algorithm === 'UCS') {
                    // Always Push (Duplicates Allowed in Queue)
                    openSet.push({ id: neighborId, g: tentativeG, f: f, path: newPath });
                    enqueuedCount++;
                } else {
                    // BFS / DFS Graph Search (Update existing - Standard)
                    const existingNodeIndex = openSet.findIndex(n => n.id === neighborId);
                    if (existingNodeIndex !== -1) {
                        if (tentativeG < openSet[existingNodeIndex].g) {
                            openSet[existingNodeIndex].g = tentativeG;
                            openSet[existingNodeIndex].f = f;
                            openSet[existingNodeIndex].path = newPath;
                        }
                    } else {
                        openSet.push({ id: neighborId, g: tentativeG, f: f, path: newPath });
                        enqueuedCount++;
                    }
                }
            }
        }

        showToast('No path found', 'error');
        setIsAnimating(false);
    };

    return (
        <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">

            {/* Sidebar */}
            <div className="w-80 flex flex-col bg-white border-r border-slate-200 z-10 shadow-xl">
                <div className="p-4 border-b border-slate-200 flex justify-between items-center">
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">
                        PathFinder Pro
                    </h1>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">

                    {/* Algorithms */}
                    <div className="space-y-3">
                        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Algorithm</h2>
                        <div className="grid grid-cols-2 gap-2">
                            {['BFS', 'DFS', 'UCS', 'AStar', 'MiniMax'].map((algo) => (
                                <button
                                    key={algo}
                                    onClick={() => setAlgorithm(algo as Algorithm)}
                                    className={`px-3 py-2 text-xs font-bold rounded-lg transition-all ${algorithm === algo
                                        ? 'bg-blue-600 text-white shadow-lg'
                                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                        }`}
                                >
                                    {algo === 'AStar' ? 'A*' : algo}
                                </button>
                            ))}
                        </div>

                        <button
                            onClick={runAlgorithm}
                            disabled={isAnimating}
                            className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-bold transition-all ${isAnimating
                                ? 'bg-slate-200 cursor-not-allowed opacity-50 text-slate-500'
                                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20'
                                }`}
                        >
                            {isAnimating ? <Settings className="animate-spin w-4 h-4" /> : <Play className="w-4 h-4" />}
                            {isAnimating ? 'Running...' : 'START SEARCH'}
                        </button>
                    </div>

                    {/* Tools */}
                    <div className="space-y-3">
                        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Tools</h2>
                        <div className="flex gap-2">
                            {[
                                { id: 'select', icon: MousePointer2, label: 'Select' },
                                { id: 'node', icon: Plus, label: 'Node' },
                                { id: 'edge', icon: ChevronRight, label: 'Edge' },
                            ].map((tool) => (
                                <button
                                    key={tool.id}
                                    onClick={() => setMode(tool.id as InteractionMode)}
                                    className={`flex-1 p-2 rounded-lg flex justify-center items-center transition-all ${mode === tool.id
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                        }`}
                                    title={tool.label}
                                >
                                    <tool.icon className="w-5 h-5" />
                                </button>
                            ))}
                        </div>
                        <div className="flex gap-2">
                            <button onClick={randomizeWeights} className="flex-1 p-2 bg-slate-100 rounded-lg text-slate-600 hover:bg-slate-200 flex items-center justify-center gap-2 text-xs font-medium">
                                <Shuffle className="w-4 h-4" /> Random
                            </button>
                            <button onClick={loadExample1} className="flex-1 p-2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg hover:bg-indigo-100 flex items-center justify-center gap-2 text-xs font-bold transition-all">
                                <Info className="w-4 h-4" /> Ex 1
                            </button>
                            <button onClick={loadExample3} className="flex-1 p-2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg hover:bg-indigo-100 flex items-center justify-center gap-2 text-xs font-bold transition-all">
                                <Info className="w-4 h-4" /> Ex 3
                            </button>
                        </div>
                        <div className="flex gap-2">
                            <button onClick={deleteSelected} disabled={!selectedNode} className="flex-1 p-2 bg-slate-100 rounded-lg text-red-500 hover:bg-red-50 disabled:opacity-50 flex justify-center">
                                <Trash2 className="w-5 h-5" />
                            </button>
                        </div>
                    </div>

                    {/* Heuristics / Values */}
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                                <Table className="w-3 h-3" /> {algorithm === 'MiniMax' ? 'Node Values' : 'Heuristics'}
                            </h2>
                            <button
                                onClick={() => setManualHeuristics(!manualHeuristics)}
                                className="text-[10px] text-blue-400 hover:underline"
                            >
                                {manualHeuristics ? "Switch to Auto" : "Switch to Manual"}
                            </button>
                        </div>

                        {/* Indirect Cost Toggle */}
                        <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                            <input
                                type="checkbox" // Input type checkbox
                                checked={showIndirect}
                                onChange={(e) => setShowIndirect(e.target.checked)}
                                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
                            />
                            Show Indirect Costs (Heuristics)
                        </label>

                        {/* Tree Search Toggle */}
                        <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={treeSearch}
                                onChange={(e) => setTreeSearch(e.target.checked)}
                                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
                            />
                            Tree Search (Allow Duplicates)
                        </label>

                        {/* Directed/Undirected Toggle */}
                        <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={isDirected}
                                onChange={(e) => setIsDirected(e.target.checked)}
                                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
                            />
                            Directed Graph
                        </label>

                        {/* Minimax specific settings */}
                        {algorithm === 'MiniMax' && (
                            <div className="pt-2 border-t border-slate-200 mt-2">
                                <h3 className="text-[10px] font-bold text-slate-500 uppercase mb-2">Minimax Settings</h3>
                                <div className="flex items-center gap-4">
                                    <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
                                        <input
                                            type="radio"
                                            checked={rootIsMax}
                                            onChange={() => setRootIsMax(true)}
                                            className="text-blue-600 focus:ring-0"
                                        />
                                        Root is MAX
                                    </label>
                                    <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
                                        <input
                                            type="radio"
                                            checked={!rootIsMax}
                                            onChange={() => setRootIsMax(false)}
                                            className="text-pink-600 focus:ring-0"
                                        />
                                        Root is MIN
                                    </label>
                                </div>
                            </div>
                        )}

                        {startNode && (goalNode || algorithm === 'MiniMax') ? (
                            <div className="bg-slate-50 border border-slate-200 rounded-lg p-2 max-h-40 overflow-y-auto custom-scrollbar">
                                <table className="w-full text-xs text-left">
                                    <thead>
                                        <tr className="text-slate-500 border-b border-slate-200">
                                            <th className="pb-1">Node</th>
                                            <th className="pb-1">{algorithm === 'MiniMax' ? 'Value' : 'h(n)'}</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {nodes.map(node => (
                                            <tr key={node.id} className="border-b border-slate-100 last:border-0">
                                                <td className="py-1.5 font-medium text-slate-700">{node.id}</td>
                                                <td className="py-1.5">
                                                    {manualHeuristics || algorithm === 'MiniMax' ? (
                                                        <input
                                                            type="number"
                                                            value={heuristics[node.id] || 0}
                                                            onChange={(e) => updateHeuristic(node.id, e.target.value)}
                                                            className="w-12 bg-white border border-slate-200 rounded px-1 text-slate-900 focus:border-blue-500 outline-none"
                                                        />
                                                    ) : (
                                                        <span className="text-blue-600">{heuristics[node.id] || 0}</span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <div className="text-xs text-slate-500 italic p-2">
                                {algorithm === 'MiniMax' ? 'Set Start node to see values' : 'Set Start & Goal to see heuristics'}
                            </div>
                        )}
                    </div>

                    {/* Speed */}
                    <div className="space-y-2">
                        <div className="flex justify-between text-xs text-slate-500">
                            <span>Speed</span>
                            <span>{speed}ms</span>
                        </div>
                        <input type="range" min="100" max="2000" step="100" value={speed} onChange={(e) => setSpeed(parseInt(e.target.value))} className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" />
                    </div>

                    <div className="flex gap-2 pt-4">
                        <button onClick={resetGraph} className="flex-1 py-2 text-xs font-medium bg-slate-100 text-slate-600 rounded hover:bg-slate-200">Reset Path</button>
                        <button onClick={clearAll} className="flex-1 py-2 text-xs font-medium bg-slate-100 text-red-500 rounded hover:bg-slate-200">Clear All</button>
                    </div>
                </div>
            </div>

            {/* Main Area */}
            <div className="flex-1 relative bg-slate-50">
                <canvas
                    ref={canvasRef}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onDoubleClick={handleDoubleClick}
                    onContextMenu={handleContextMenu}
                    className={`w-full h-full block touch-none ${mode === 'select' ? 'cursor-default' : 'cursor-crosshair'}`}
                />

                {/* Logs Panel - Table Style */}
                {showLogs && stepLogs.length > 0 && (
                    <div className="absolute top-4 right-4 w-[500px] max-h-[calc(100vh-2rem)] flex flex-col bg-white/95 backdrop-blur shadow-2xl border border-slate-200 rounded-xl overflow-hidden animate-in slide-in-from-right-10 duration-300">
                        <div className="p-3 bg-slate-100 border-b border-slate-200 flex justify-between items-center">
                            <h3 className="font-bold text-sm text-slate-800 flex items-center gap-2">
                                <List className="w-4 h-4 text-blue-600" /> Trace
                            </h3>
                            <button onClick={() => setStepLogs([])} className="text-xs text-slate-500 hover:text-slate-800">Clear</button>
                        </div>

                        <div className="overflow-y-auto p-0 custom-scrollbar">
                            <table className="w-full text-xs text-left border-collapse">
                                <thead className="bg-slate-50 text-slate-500 sticky top-0">
                                    <tr>
                                        <th className="p-2 border-b border-slate-200 w-12">Step</th>
                                        <th className="p-2 border-b border-slate-200">Enqueued Paths</th>
                                        <th className="p-2 border-b border-slate-200 w-16 text-center">Extended</th>
                                        <th className="p-2 border-b border-slate-200 w-16 text-center">Enqueued</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {stepLogs.map((log) => (
                                        <tr key={log.step} className="hover:bg-slate-50 transition-colors">
                                            <td className="p-2 font-mono text-slate-500 align-top">{log.step}.</td>
                                            <td className="p-2 align-top">
                                                <div className="flex flex-col gap-1">
                                                    {log.queue.length === 0 ? <span className="text-slate-400 italic">Empty</span> : (
                                                        <div className="flex flex-wrap gap-1">
                                                            {log.queue.map((item, i) => (
                                                                <span key={i} className="inline-flex items-center gap-1 bg-white px-1.5 py-0.5 rounded border border-slate-200 shadow-sm">
                                                                    <span className="font-bold text-indigo-600">{`{${item.path.join('')}}`}</span>
                                                                    {algorithm === 'UCS' && <span className="text-slate-500">[{item.cost}]</span>}
                                                                    {algorithm === 'AStar' && <span className="text-slate-500">[{item.total}]</span>}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                            <td className="p-2 text-center font-mono text-slate-500 align-top">{log.extendedCount}</td>
                                            <td className="p-2 text-center font-mono text-slate-500 align-top">{log.enqueuedCount}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {/* Summary Footer */}
                        {stepLogs.length > 0 && (
                            <div className="p-2 bg-slate-50 border-t border-slate-200 text-[10px] text-slate-500 flex justify-between">
                                <span>Extended: {stepLogs[stepLogs.length - 1].extendedCount}</span>
                                <span>Enqueued: {stepLogs[stepLogs.length - 1].enqueuedCount}</span>
                            </div>
                        )}
                    </div>
                )}

                {/* Instructions Overlay */}
                <div className="absolute bottom-16 left-4 bg-white/90 backdrop-blur border border-slate-200 rounded-xl p-3 shadow-lg pointer-events-none select-none max-w-xs z-10">
                    <div className="text-[11px] text-slate-600 space-y-1">
                        <p><span className="font-bold text-slate-800">Double-click node</span> to rename (or set Value in Minimax).</p>
                        <p><span className="font-bold text-slate-800">Right-click</span> to set Start/Goal.</p>
                        <p><span className="font-bold text-slate-800">Drag</span> nodes to move.</p>
                        <p><span className="font-bold text-slate-800">Click edge</span> to edit weight.</p>
                    </div>
                </div>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur border border-slate-200 rounded-xl p-3 pointer-events-none select-none shadow-lg">
                    <div className="flex gap-4 text-[10px] text-slate-600 font-medium">
                        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Start</div>
                        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-red-500" /> Goal</div>
                        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-orange-200" /> Extended (Closed)</div>
                        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-indigo-300" /> Frontier (Queue)</div>
                        {algorithm === 'MiniMax' && <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-pink-600" /> Value</div>}
                    </div>
                </div>

                {/* Toasts */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col gap-2 z-50 pointer-events-none">
                    {toasts.map(toast => (
                        <div key={toast.id} className="animate-in slide-in-from-bottom-5 fade-in duration-300 flex items-center gap-2 bg-white text-slate-900 px-3 py-2 rounded-lg shadow-xl border border-slate-200 text-sm">
                            {toast.type === 'success' && <CheckCircle className="w-4 h-4 text-green-500" />}
                            {toast.type === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
                            <span>{toast.message}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default GraphVisualizer;