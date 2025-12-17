import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Play, RotateCcw, Plus, Trash2, Settings,
    MousePointer2, ChevronRight, FastForward,
    Info, CheckCircle, AlertCircle, List, Table, Shuffle,
    Menu, X, GripVertical
} from 'lucide-react';

// --- Types ---
type Algorithm = 'BFS' | 'DFS' | 'UCS' | 'AStar' | 'MiniMax';
type InteractionMode = 'select' | 'node' | 'edge';
type NodeType = { id: string; x: number; y: number };
type EdgeType = { from: string; to: string; weight: number };
type ToastType = { id: number; message: string; type: 'success' | 'error' | 'info' };

type QueueItem = {
    path: string[];
    cost: number;      // g for UCS/A*, 0 for BFS/DFS, value for MiniMax
    heuristic: number; // h for A*
    total: number;     // f for A*, g for UCS
};

type StepLog = {
    step: number;
    queue: QueueItem[];
    extendedNode: string | null;
    extendedList: string[];
    extendedListWithCosts: Record<string, number>;  // For A*/UCS: node -> cost mapping
    extendedCount: number;
    enqueuedCount: number;
    events: string[];
};

// --- Constants ---
const NODE_RADIUS = 24;
const COLORS = {
    bg: '#ffffff',
    node: '#f1f5f9',
    nodeBorder: '#334155',
    nodeSelected: '#2563eb',
    nodeStart: '#059669',
    nodeGoal: '#dc2626',
    edge: '#64748b',
    visited: '#fb923c',
    queued: '#818cf8',
    path: '#7c3aed',
    text: '#0f172a',
    accent: '#06b6d4',
    minimaxValue: '#db2777',
    grid: '#e2e8f0',
    weightBadgeErr: '#ffffff',
    weightBadgeText: '#0f172a'
};

// Helper: compute path cost from edges
const computePathCost = (path: string[], edges: EdgeType[], directed: boolean): number => {
    let cost = 0;
    for (let i = 0; i < path.length - 1; i++) {
        const u = path[i];
        const v = path[i + 1];
        let edge = edges.find(e => e.from === u && e.to === v);
        if (!directed && !edge) {
            edge = edges.find(e => e.from === v && e.to === u);
        }
        if (edge) cost += edge.weight;
    }
    return cost;
};

const GraphVisualizer: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Graph data
    const [nodes, setNodes] = useState<NodeType[]>([]);
    const [edges, setEdges] = useState<EdgeType[]>([]);
    const [startNode, setStartNode] = useState<string | null>(null);
    const [goalNode, setGoalNode] = useState<string | null>(null);

    // UI state
    const [mode, setMode] = useState<InteractionMode>('select');
    const [selectedNode, setSelectedNode] = useState<string | null>(null);
    const [selectedEdge, setSelectedEdge] = useState<{ from: string; to: string } | null>(null);
    const [algorithm, setAlgorithm] = useState<Algorithm>('AStar');
    const [speed, setSpeed] = useState<number>(800);
    const [isAnimating, setIsAnimating] = useState(false);
    const [showHeuristics, setShowHeuristics] = useState(true);
    const [showIndirect, setShowIndirect] = useState(false);
    const [treeSearch, setTreeSearch] = useState(true); // false = graph search (extended list)
    const [manualHeuristics, setManualHeuristics] = useState(false);
    const [isDirected, setIsDirected] = useState(true);
    const [rootIsMax, setRootIsMax] = useState(true);
    const [showLogs, setShowLogs] = useState(true);
    const [toasts, setToasts] = useState<ToastType[]>([]);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [tracePanelPos, setTracePanelPos] = useState({ x: 0, y: 0 });
    const [isDraggingPanel, setIsDraggingPanel] = useState(false);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

    // Visualization state
    const [visitedNodes, setVisitedNodes] = useState<Set<string>>(new Set());
    const [frontierNodes, setFrontierNodes] = useState<Set<string>>(new Set());
    const [currentPath, setCurrentPath] = useState<string[]>([]);
    const [heuristics, setHeuristics] = useState<Record<string, number>>({});
    const [computedValues, setComputedValues] = useState<Record<string, number>>({});
    const [stepLogs, setStepLogs] = useState<StepLog[]>([]);

    // Interaction state
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

    // Neighbours (respects directed / undirected toggle)
    const getNeighbors = (nodeId: string): { id: string; weight: number }[] => {
        if (isDirected) {
            return edges
                .filter(e => e.from === nodeId)
                .map(e => ({ id: e.to, weight: e.weight }));
        }
        // undirected: treat edge as bidirectional
        return edges
            .filter(e => e.from === nodeId || e.to === nodeId)
            .map(e =>
                e.from === nodeId
                    ? { id: e.to, weight: e.weight }
                    : { id: e.from, weight: e.weight }
            );
    };

    // --- Heuristics (auto Euclidean when not manual) ---
    useEffect(() => {
        if (!manualHeuristics && goalNode && algorithm !== 'MiniMax') {
            const target = nodes.find(n => n.id === goalNode);
            if (!target) return;
            const h: Record<string, number> = {};
            nodes.forEach(n => {
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
            weight: Math.floor(Math.random() * 9) + 1
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

    // Example 1 (A* slide, Part 2 Example 1)
    const loadExample1 = () => {
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
            { from: 'D', to: 'G', weight: 5 },
        ];
        setNodes(newNodes);
        setEdges(newEdges);
        setStartNode('S');
        setGoalNode('G');

        setManualHeuristics(true);
        setHeuristics({
            S: 9.8,
            A: 7.6,
            B: 6.5,
            C: 7.6,
            D: 5.0,
            E: 4.1,
            G: 0,
        });

        setShowIndirect(true);
        setTreeSearch(false);    // Graph search (with extended list) as shown in slides
        setIsDirected(false);
        setAlgorithm('AStar');
        resetGraph();
        showToast('Loaded Example 1', 'success');
    };

    // Example 2 (A* slide, Part 2 Example 2)
    const loadExample2 = () => {
        const newNodes: NodeType[] = [
            { id: 'S', x: 100, y: 250 },
            { id: 'A', x: 250, y: 150 },
            { id: 'B', x: 250, y: 350 },
            { id: 'C', x: 400, y: 250 },
            { id: 'G', x: 550, y: 250 },
        ];
        const newEdges: EdgeType[] = [
            { from: 'S', to: 'A', weight: 1 },
            { from: 'S', to: 'B', weight: 4 },
            { from: 'A', to: 'B', weight: 2 },
            { from: 'A', to: 'C', weight: 5 },
            { from: 'A', to: 'G', weight: 11 },
            { from: 'B', to: 'C', weight: 2 },
            { from: 'C', to: 'G', weight: 3 },
        ];
        setNodes(newNodes);
        setEdges(newEdges);
        setStartNode('S');
        setGoalNode('G');

        setManualHeuristics(true);
        setHeuristics({
            S: 7,
            A: 6,
            B: 2,
            C: 1,
            G: 0,
        });

        setShowIndirect(false);
        setTreeSearch(false);   // Graph search (with extended list) as shown in slides
        setIsDirected(true);
        setAlgorithm('AStar');
        resetGraph();
        showToast('Loaded Example 2', 'success');
    };


    // Example 3 (A* slide, Part 2 Example 3)
    const loadExample3 = () => {
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
            { from: 'C', to: 'G', weight: 3 },
        ];
        setNodes(newNodes);
        setEdges(newEdges);
        setStartNode('S');
        setGoalNode('G');

        setManualHeuristics(true);
        setHeuristics({
            S: 7,
            A: 6,
            B: 2,
            C: 2,
            G: 0,
        });

        setShowIndirect(false);
        setTreeSearch(false);   // Graph search (with extended list) as shown in slides
        setIsDirected(true);
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

        // Background grid
        ctx.clearRect(0, 0, rect.width, rect.height);
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
                    return (nodeId === edge.from && currentPath[i + 1] === edge.to) ||
                        (!isDirected && nodeId === edge.to && currentPath[i + 1] === edge.from);
                });

            const isSelectedEdge = selectedEdge &&
                ((edge.from === selectedEdge.from && edge.to === selectedEdge.to) ||
                    (!isDirected && edge.from === selectedEdge.to && edge.to === selectedEdge.from));

            ctx.beginPath();
            ctx.moveTo(n1.x, n1.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.strokeStyle = isPathEdge ? COLORS.path : (isSelectedEdge ? COLORS.nodeSelected : COLORS.edge);
            ctx.lineWidth = isPathEdge ? 4 : (isSelectedEdge ? 3 : 2);
            ctx.stroke();

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

            // edge weight badge
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

        // Temporary edge drag line
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
            else if (visitedNodes.has(node.id)) fill = COLORS.visited;
            else if (frontierNodes.has(node.id)) fill = COLORS.queued;

            ctx.fillStyle = fill;
            ctx.fill();

            ctx.strokeStyle = COLORS.nodeBorder;
            ctx.lineWidth = 2;
            ctx.stroke();

            if (node.id === selectedNode) {
                ctx.strokeStyle = COLORS.nodeSelected;
                ctx.lineWidth = 3;
                ctx.stroke();
            }

            // Text
            const isSpecial = node.id === startNode || node.id === goalNode || currentPath.includes(node.id);
            ctx.fillStyle = isSpecial ? '#ffffff' : COLORS.text;
            ctx.font = node.id.length > 2 ? 'bold 10px Inter, sans-serif' : 'bold 14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.id, node.x, node.y);

            // Heuristic display
            if (showHeuristics && algorithm !== 'MiniMax' && heuristics[node.id] !== undefined) {
                ctx.fillStyle = '#fbbf24';
                ctx.font = '11px Inter, sans-serif';
                ctx.fillText(`h:${heuristics[node.id]}`, node.x, node.y + NODE_RADIUS + 14);
            }

            // MiniMax value display
            if (computedValues[node.id] !== undefined) {
                ctx.fillStyle = COLORS.minimaxValue;
                ctx.font = 'bold 11px Inter, sans-serif';
                ctx.fillText(`Val:${computedValues[node.id]}`, node.x, node.y - NODE_RADIUS - 5);
            }
        });

        // Dashed heuristic arcs to goal
        if (showIndirect && goalNode && algorithm === 'AStar') {
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

                    const offset = 40;
                    const normX = -dy / dist;
                    const normY = dx / dist;
                    const cpX = midX + normX * offset;
                    const cpY = midY + normY * offset;

                    ctx.beginPath();
                    ctx.moveTo(node.x, node.y);
                    ctx.quadraticCurveTo(cpX, cpY, target.x, target.y);
                    ctx.strokeStyle = 'rgba(180, 83, 9, 0.8)';
                    ctx.setLineDash([5, 5]);
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    ctx.setLineDash([]);

                    const labelX = 0.25 * node.x + 0.5 * cpX + 0.25 * target.x;
                    const labelY = 0.25 * node.y + 0.5 * cpY + 0.25 * target.y;

                    ctx.beginPath();
                    ctx.arc(labelX, labelY, 10, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(180, 83, 9, 0.8)';
                    ctx.lineWidth = 1;
                    ctx.stroke();

                    ctx.fillStyle = '#2563eb';
                    ctx.font = 'italic 10px Inter, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(hVal.toString(), labelX, labelY);
                });
            }
        }
    }, [nodes, edges, startNode, goalNode, visitedNodes, frontierNodes, currentPath, selectedNode, dragState, showHeuristics, heuristics, computedValues, showIndirect, algorithm, isDirected]);

    useEffect(() => {
        let handle: number;
        const render = () => {
            draw();
            handle = requestAnimationFrame(render);
        };
        render();
        return () => cancelAnimationFrame(handle);
    }, [draw]);

    // --- Interaction handlers ---
    const handleMouseDown = (e: React.MouseEvent) => {
        if (isAnimating) return;
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // click on edge weight to edit
        const clickedEdgeIndex = edges.findIndex(edge => {
            const n1 = nodes.find(n => n.id === edge.from);
            const n2 = nodes.find(n => n.id === edge.to);
            if (!n1 || !n2) return false;
            const midX = (n1.x + n2.x) / 2;
            const midY = (n1.y + n2.y) / 2;
            const dist = Math.hypot(x - midX, y - midY);
            return dist <= 20;
        });

        if (clickedEdgeIndex !== -1) {
            // If already selected, edit weight?
            // Or maybe just click to select, double click to edit?
            // Let's keep click to edit weight for now to minimize disruption, 
            // BUT also select it.
            const edge = edges[clickedEdgeIndex];
            setSelectedEdge({ from: edge.from, to: edge.to });
            setSelectedNode(null);

            // Optional: Only prompt if double click? Or maybe separate logic.
            // Requirement was just "remove link". So selection is key.
            // Let's make weight editing require double click or something?
            // User said: "Click edge to edit weight" in overlay.
            // We can keep that but also select it.

            // Actually, if we want to delete, we need to select it first without prompting immediately every time if we just want to select.
            // Let's modify: Click = Select. Double Click = Edit Weight.
            return;
        }

        const clickedNode = nodes.find(
            n => Math.hypot(n.x - x, n.y - y) < NODE_RADIUS
        );

        if (clickedNode) {
            if (mode === 'edge') {
                setDragState({ type: 'edge', sourceId: clickedNode.id, currentPos: { x, y } });
                setSelectedEdge(null);
            } else {
                setSelectedNode(clickedNode.id);
                setSelectedEdge(null);
                setDragState({ type: 'node', sourceId: clickedNode.id, currentPos: { x, y } });
            }
        } else {
            setSelectedNode(null);
            setSelectedEdge(null);
            if (mode === 'node') {
                setNodes([...nodes, { id: generateNodeId(), x, y }]);
            }
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (!dragState.type) {
            const isEdge = edges.some(edge => {
                const n1 = nodes.find(n => n.id === edge.from);
                const n2 = nodes.find(n => n.id === edge.to);
                if (!n1 || !n2) return false;
                const midX = (n1.x + n2.x) / 2;
                const midY = (n1.y + n2.y) / 2;
                return Math.hypot(x - midX, y - midY) <= 20;
            });
            canvasRef.current!.style.cursor = isEdge
                ? 'pointer'
                : mode === 'select'
                    ? 'default'
                    : 'crosshair';
        }

        if (!dragState.type) return;
        setDragState(prev => ({ ...prev, currentPos: { x, y } }));

        if (dragState.type === 'node' && dragState.sourceId) {
            setNodes(prev =>
                prev.map(n =>
                    n.id === dragState.sourceId ? { ...n, x, y } : n
                )
            );
        }
    };

    const handleMouseUp = (e: React.MouseEvent) => {
        if (dragState.type === 'edge' && dragState.sourceId) {
            const rect = canvasRef.current!.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const target = nodes.find(n => Math.hypot(n.x - x, n.y - y) < NODE_RADIUS);

            if (target && target.id !== dragState.sourceId) {
                const exists = edges.some(
                    edge =>
                        edge.from === dragState.sourceId && edge.to === target.id
                );
                if (!exists) {
                    setEdges(prev => [
                        ...prev,
                        { from: dragState.sourceId!, to: target.id, weight: 1 }
                    ]);
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
        const clickedNode = nodes.find(
            n => Math.hypot(n.x - x, n.y - y) < NODE_RADIUS
        );

        if (!clickedNode) {
            // Check for edge double click (Weight Edit)
            const clickedEdgeIndex = edges.findIndex(edge => {
                const n1 = nodes.find(n => n.id === edge.from);
                const n2 = nodes.find(n => n.id === edge.to);
                if (!n1 || !n2) return false;
                const midX = (n1.x + n2.x) / 2;
                const midY = (n1.y + n2.y) / 2;
                const dist = Math.hypot(x - midX, y - midY);
                return dist <= 20;
            });

            if (clickedEdgeIndex !== -1) {
                const newWeight = prompt(
                    'Enter new edge weight:',
                    edges[clickedEdgeIndex].weight.toString()
                );
                if (newWeight !== null && !isNaN(parseFloat(newWeight))) {
                    setEdges(prev => {
                        const next = [...prev];
                        next[clickedEdgeIndex] = {
                            ...next[clickedEdgeIndex],
                            weight: parseFloat(newWeight)
                        };
                        return next;
                    });
                }
            }
            return;
        }

        // Minimax: double-click to set value
        if (algorithm === 'MiniMax') {
            const currentVal = heuristics[clickedNode.id] || 0;
            const newVal = prompt(
                `Enter Value for Node ${clickedNode.id}:`,
                currentVal.toString()
            );
            if (newVal !== null && !isNaN(parseFloat(newVal))) {
                setHeuristics(prev => ({
                    ...prev,
                    [clickedNode.id]: parseFloat(newVal)
                }));
                showToast(
                    `Set Node ${clickedNode.id} value to ${newVal}`,
                    'success'
                );
            }
            return;
        }

        // rename node
        const newId = prompt('Enter new name for node:', clickedNode.id);
        if (!newId || newId === clickedNode.id) return;

        if (nodes.some(n => n.id === newId)) {
            showToast('Node name must be unique', 'error');
            return;
        }

        setNodes(prev =>
            prev.map(n => (n.id === clickedNode.id ? { ...n, id: newId } : n))
        );
        setEdges(prev =>
            prev.map(e => ({
                ...e,
                from: e.from === clickedNode.id ? newId : e.from,
                to: e.to === clickedNode.id ? newId : e.to
            }))
        );
        if (startNode === clickedNode.id) setStartNode(newId);
        if (goalNode === clickedNode.id) setGoalNode(newId);

        setHeuristics(prev => {
            const next = { ...prev };
            if (next[clickedNode.id] !== undefined) {
                next[newId] = next[clickedNode.id];
                delete next[clickedNode.id];
            }
            return next;
        });

        showToast(`Renamed ${clickedNode.id} to ${newId}`, 'success');
    };

    const handleContextMenu = (e: React.MouseEvent) => {
        e.preventDefault();
        if (isAnimating) return;
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const clicked = nodes.find(
            n => Math.hypot(n.x - x, n.y - y) < NODE_RADIUS
        );
        if (!clicked) return;

        if (startNode === clicked.id) {
            setStartNode(null);
            setGoalNode(clicked.id);
        } else if (goalNode === clicked.id) {
            setGoalNode(null);
        } else {
            if (startNode) setStartNode(null);
            setStartNode(clicked.id);
        }
    };

    const deleteSelected = () => {
        if (selectedNode) {
            setNodes(nodes.filter(n => n.id !== selectedNode));
            setEdges(edges.filter(e => e.from !== selectedNode && e.to !== selectedNode));
            if (startNode === selectedNode) setStartNode(null);
            if (goalNode === selectedNode) setGoalNode(null);
            setSelectedNode(null);
        } else if (selectedEdge) {
            setEdges(edges.filter(e =>
                !((e.from === selectedEdge.from && e.to === selectedEdge.to) ||
                    (!isDirected && e.from === selectedEdge.to && e.to === selectedEdge.from))
            ));
            setSelectedEdge(null);
        }
    };

    // Panel dragging handlers - Mouse
    const handlePanelMouseDown = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setIsDraggingPanel(true);
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        setDragOffset({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        });
    }, []);

    // Panel dragging handlers - Touch
    const handlePanelTouchStart = useCallback((e: React.TouchEvent) => {
        e.preventDefault();
        setIsDraggingPanel(true);
        const touch = e.touches[0];
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        setDragOffset({
            x: touch.clientX - rect.left,
            y: touch.clientY - rect.top
        });
    }, []);

    const handlePanelMouseMove = useCallback((e: MouseEvent) => {
        if (!isDraggingPanel) return;
        const canvasRect = canvasRef.current?.getBoundingClientRect();
        if (!canvasRect) return;

        // Constrain position to stay within canvas bounds
        const maxWidth = window.innerWidth < 640 ? window.innerWidth - 20 : 500;
        const newX = Math.max(0, Math.min(e.clientX - canvasRect.left - dragOffset.x, canvasRect.width - maxWidth));
        const newY = Math.max(0, Math.min(e.clientY - canvasRect.top - dragOffset.y, canvasRect.height - 200));

        setTracePanelPos({ x: newX, y: newY });
    }, [isDraggingPanel, dragOffset]);

    const handlePanelTouchMove = useCallback((e: TouchEvent) => {
        if (!isDraggingPanel) return;
        e.preventDefault();
        const canvasRect = canvasRef.current?.getBoundingClientRect();
        if (!canvasRect) return;

        const touch = e.touches[0];
        // Constrain position to stay within canvas bounds
        const maxWidth = window.innerWidth < 640 ? window.innerWidth - 20 : 500;
        const newX = Math.max(0, Math.min(touch.clientX - canvasRect.left - dragOffset.x, canvasRect.width - maxWidth));
        const newY = Math.max(0, Math.min(touch.clientY - canvasRect.top - dragOffset.y, canvasRect.height - 200));

        setTracePanelPos({ x: newX, y: newY });
    }, [isDraggingPanel, dragOffset]);

    const handlePanelMouseUp = useCallback(() => {
        setIsDraggingPanel(false);
    }, []);

    const handlePanelTouchEnd = useCallback(() => {
        setIsDraggingPanel(false);
    }, []);

    // Add global listeners for panel dragging
    useEffect(() => {
        if (isDraggingPanel) {
            window.addEventListener('mousemove', handlePanelMouseMove);
            window.addEventListener('mouseup', handlePanelMouseUp);
            window.addEventListener('touchmove', handlePanelTouchMove, { passive: false });
            window.addEventListener('touchend', handlePanelTouchEnd);
            return () => {
                window.removeEventListener('mousemove', handlePanelMouseMove);
                window.removeEventListener('mouseup', handlePanelMouseUp);
                window.removeEventListener('touchmove', handlePanelTouchMove);
                window.removeEventListener('touchend', handlePanelTouchEnd);
            };
        }
    }, [isDraggingPanel, handlePanelMouseMove, handlePanelMouseUp, handlePanelTouchMove, handlePanelTouchEnd]);

    // --- Algorithms ---

    const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

    // BFS (uninformed, ignores weights for ordering)  :contentReference[oaicite:4]{index=4}
    const runBFS = async () => {
        const start = startNode!;
        const goal = goalNode!;

        type Path = string[];
        let queue: Path[] = [[start]];
        let extended = new Set<string>();
        let extendedCount = 0;
        let enqueuedCount = 1;

        setVisitedNodes(new Set());
        setFrontierNodes(new Set([start]));
        setCurrentPath([start]);
        setStepLogs([{
            step: 1,
            queue: [{ path: [start], cost: 0, heuristic: 0, total: 0 }],
            extendedNode: null,
            extendedList: [],
            extendedListWithCosts: {},
            extendedCount: 0,
            enqueuedCount: 1,
            events: ['Initial state']
        }]);

        while (queue.length > 0) {
            const path = queue.shift()!;
            const current = path[path.length - 1];

            if (!treeSearch && extended.has(current)) {
                // already processed this state
                continue;
            }

            extended.add(current);
            extendedCount++;
            setVisitedNodes(new Set(extended));
            setCurrentPath(path);
            setFrontierNodes(new Set(queue.map(p => p[p.length - 1])));

            const events: string[] = [];

            if (current === goal) {
                const cost = computePathCost(path, edges, isDirected);
                events.push(`Goal ${goal} reached; path cost = ${cost}`);
                setStepLogs(prev => [
                    ...prev,
                    {
                        step: prev.length + 1,
                        queue: [{ path, cost: 0, heuristic: 0, total: 0 }, ...queue.map(p => ({
                            path: p,
                            cost: 0,
                            heuristic: 0,
                            total: 0
                        }))],
                        extendedNode: current,
                        extendedList: Array.from(extended),
                        extendedListWithCosts: {},
                        extendedCount,
                        enqueuedCount,
                        events
                    }
                ]);
                showToast(`Goal reached (BFS). Path cost = ${cost}`, 'success');
                return;
            }

            const neighbors = getNeighbors(current)
                .map(n => n.id)
                .sort((a, b) => a.localeCompare(b));

            for (const nb of neighbors) {
                if (treeSearch && path.includes(nb)) continue;

                if (!treeSearch) {
                    if (extended.has(nb)) {
                        events.push(`Child ${nb} skipped (already extended)`);
                        continue;
                    }
                    const inQueue = queue.some(p => p[p.length - 1] === nb);
                    if (inQueue) {
                        events.push(`Child ${nb} skipped (already in queue)`);
                        continue;
                    }
                }

                queue.push([...path, nb]);
                enqueuedCount++;
                events.push(`Child ${nb} enqueued`);
            }

            const currentQueueSnapshot = queue.map(p => ({
                path: p,
                cost: 0,
                heuristic: 0,
                total: 0
            }));

            setStepLogs(prev => [
                ...prev,
                {
                    step: prev.length + 1,
                    queue: currentQueueSnapshot,
                    extendedNode: current,
                    extendedList: Array.from(extended),
                    extendedListWithCosts: {},
                    extendedCount,
                    enqueuedCount,
                    events
                }
            ]);

            await delay(speed);
        }

        showToast('No path found (BFS)', 'error');
    };

    // DFS (uninformed, depth-first) :contentReference[oaicite:5]{index=5}
    const runDFS = async () => {
        const start = startNode!;
        const goal = goalNode!;

        type Path = string[];
        let stack: Path[] = [[start]];
        let extended = new Set<string>();
        let extendedCount = 0;
        let enqueuedCount = 1;

        setVisitedNodes(new Set());
        setFrontierNodes(new Set([start]));
        setCurrentPath([start]);
        setStepLogs([{
            step: 1,
            queue: [{ path: [start], cost: 0, heuristic: 0, total: 0 }],
            extendedNode: null,
            extendedList: [],
            extendedListWithCosts: {},
            extendedCount: 0,
            enqueuedCount: 1,
            events: ['Initial state']
        }]);

        while (stack.length > 0) {
            const path = stack.pop()!;
            const current = path[path.length - 1];

            if (!treeSearch && extended.has(current)) {
                continue;
            }

            extended.add(current);
            extendedCount++;
            setVisitedNodes(new Set(extended));
            setCurrentPath(path);
            setFrontierNodes(new Set(stack.map(p => p[p.length - 1])));

            const events: string[] = [];

            if (current === goal) {
                const cost = computePathCost(path, edges, isDirected);
                events.push(`Goal ${goal} reached; path cost = ${cost}`);
                setStepLogs(prev => [
                    ...prev,
                    {
                        step: prev.length + 1,
                        queue: [{ path, cost: 0, heuristic: 0, total: 0 }, ...stack.map(p => ({
                            path: p,
                            cost: 0,
                            heuristic: 0,
                            total: 0
                        }))],
                        extendedNode: current,
                        extendedList: Array.from(extended),
                        extendedListWithCosts: {},
                        extendedCount,
                        enqueuedCount,
                        events
                    }
                ]);
                showToast(`Goal reached (DFS). Path cost = ${cost}`, 'success');
                return;
            }

            // children added in REVERSE lexical order so they are popped in lexical order
            const neighbors = getNeighbors(current)
                .map(n => n.id)
                .sort((a, b) => b.localeCompare(a));

            for (const nb of neighbors) {
                if (treeSearch && path.includes(nb)) continue;

                if (!treeSearch) {
                    if (extended.has(nb)) {
                        events.push(`Child ${nb} skipped (already extended)`);
                        continue;
                    }
                    const inStack = stack.some(p => p[p.length - 1] === nb);
                    if (inStack) {
                        events.push(`Child ${nb} skipped (already in stack)`);
                        continue;
                    }
                }

                stack.push([...path, nb]);
                enqueuedCount++;
                events.push(`Child ${nb} pushed`);
            }

            const currentStackSnapshot = stack.slice().reverse().map(p => ({
                path: p,
                cost: 0,
                heuristic: 0,
                total: 0
            }));

            setStepLogs(prev => [
                ...prev,
                {
                    step: prev.length + 1,
                    queue: currentStackSnapshot,
                    extendedNode: current,
                    extendedList: Array.from(extended),
                    extendedListWithCosts: {},
                    extendedCount,
                    enqueuedCount,
                    events
                }
            ]);

            await delay(speed);
        }

        showToast('No path found (DFS)', 'error');
    };

    // UCS / A* shared cost-based core
    const runCostSearch = async (useHeuristic: boolean) => {
        const start = startNode!;
        const goal = goalNode!;

        type Item = { id: string; g: number; f: number; path: string[] };
        const open: Item[] = [];
        const closed = new Map<string, number>(); // extended list: best g so far for extended states

        const startH = useHeuristic ? (heuristics[start] ?? 0) : 0;
        open.push({ id: start, g: 0, f: useHeuristic ? startH : 0, path: [start] });

        let extendedCount = 0;
        let enqueuedCount = 1;

        const compare = (a: Item, b: Item) => {
            if (useHeuristic) {
                if (a.f !== b.f) return a.f - b.f;   // A*: by f=g+h
            } else {
                if (a.g !== b.g) return a.g - b.g;   // UCS: by g
            }
            // Tie-breaker: Lexical order of the FULL path
            // (Standard requirement: SBE < SBEC, SAB < SZA, etc.)
            return a.path.join('').localeCompare(b.path.join(''));
        };

        const snapshotQueue = (): QueueItem[] => {
            const sorted = [...open].sort(compare);
            return sorted.map(item => ({
                path: item.path,
                cost: item.g,
                heuristic: useHeuristic ? (heuristics[item.id] ?? 0) : 0,
                total: useHeuristic ? item.f : item.g
            }));
        };

        setVisitedNodes(new Set());
        setFrontierNodes(new Set([start]));
        setCurrentPath([start]);
        setStepLogs([{
            step: 1,
            queue: snapshotQueue(),
            extendedNode: null,
            extendedList: [],
            extendedListWithCosts: {},
            extendedCount: 0,
            enqueuedCount: 1,
            events: ['Initial state']
        }]);

        while (open.length > 0) {
            open.sort(compare);
            const current = open.shift()!;

            if (current.id === goal) {
                const events = [`Goal ${goal} reached; path cost = ${current.g}`];
                // Put current back in queue for display
                const finalQueue = snapshotQueue();
                finalQueue.unshift({
                    path: current.path,
                    cost: current.g,
                    heuristic: useHeuristic ? (heuristics[current.id] ?? 0) : 0,
                    total: useHeuristic ? current.f : current.g
                });

                setStepLogs(prev => [
                    ...prev,
                    {
                        step: prev.length + 1,
                        queue: finalQueue,
                        extendedNode: current.id,
                        extendedList: Array.from(closed.keys()),
                        extendedListWithCosts: Object.fromEntries(closed),
                        extendedCount,
                        enqueuedCount,
                        events
                    }
                ]);
                showToast(
                    `Goal reached (${useHeuristic ? 'A*' : 'UCS'}). Path cost = ${current.g}`,
                    'success'
                );
                return;
            }

            // A*: extended-list pruning BEFORE expansion
            if (useHeuristic) {
                const prevCost = closed.get(current.id);
                if (prevCost !== undefined && prevCost <= current.g) {
                    const h = heuristics[current.id] ?? 0;
                    const prevF = prevCost + h;
                    const currentF = current.g + h;
                    const events = [
                        `Node ${current.id} NOT extended (already extended with f=${prevF} ≤ f=${currentF})`
                    ];
                    setStepLogs(prev => [
                        ...prev,
                        {
                            step: prev.length + 1,
                            queue: snapshotQueue(),
                            extendedNode: null,
                            extendedList: Array.from(closed.keys()),
                            extendedListWithCosts: Object.fromEntries(closed),
                            extendedCount,
                            enqueuedCount,
                            events
                        }
                    ]);
                    await delay(speed);
                    continue;
                }
            }

            // Check if we're updating an existing entry in the closed list
            const prevClosedCost = closed.get(current.id);
            const isUpdatingClosedList = prevClosedCost !== undefined && prevClosedCost > current.g;

            closed.set(current.id, current.g);
            extendedCount++;
            setVisitedNodes(new Set(closed.keys()));
            setCurrentPath(current.path);
            setFrontierNodes(new Set(open.map(n => n.id)));

            const events: string[] = [];

            if (isUpdatingClosedList) {
                if (useHeuristic) {
                    // A*: show f-cost update
                    const h = heuristics[current.id] ?? 0;
                    const prevF = prevClosedCost + h;
                    const currentF = current.g + h;
                    events.push(`Extended List updated: ${current.id} (f=${prevF} → f=${currentF})`);
                } else {
                    // UCS: show g-cost update
                    events.push(`Extended List updated: ${current.id} (g=${prevClosedCost} → g=${current.g})`);
                }
            }

            const neighbors = getNeighbors(current.id)
                .slice()
                .sort((a, b) => a.id.localeCompare(b.id));

            for (const nb of neighbors) {
                if (treeSearch && current.path.includes(nb.id)) continue;

                const newG = current.g + nb.weight;
                const h = useHeuristic ? (heuristics[nb.id] ?? 0) : 0;
                const newF = newG + h;
                const newPath = [...current.path, nb.id];

                // Graph search with extended list
                if (!treeSearch) {
                    const closedCost = closed.get(nb.id);

                    // UCS: Prune if in extended list (User said UCS logic was correct)
                    // A*: Do NOT prune here (Generation-time pruning disabled for matches slides)
                    if (!useHeuristic && closedCost !== undefined && closedCost <= newG) {
                        continue; // UCS: Pruned by extended list
                    }

                    // A* Extended List Check:
                    // Technically we shouldn't prune if we want to allow re-opening (handled by Closed Set check on POP).
                    // But if we want to match "Not Extended" on POP, we allow them to go into queue now.

                    const idx = open.findIndex(item => item.id === nb.id);
                    if (idx !== -1) {
                        const existing = open[idx];

                        if (newG < existing.g) {
                            // Better path found
                            if (useHeuristic) {
                                // A*: Add as new entry (slides show this behavior)
                                open.push({ id: nb.id, g: newG, f: newF, path: newPath });
                                enqueuedCount++;
                                events.push(`Child ${nb.id} enqueued (better path: f=${newF} < f=${existing.f})`);
                                continue;
                            } else {
                                // UCS: Update existing entry
                                open[idx] = { id: nb.id, g: newG, f: newF, path: newPath };
                                events.push(`Entry for ${nb.id} updated (g=${existing.g} → g=${newG})`);
                                continue;
                            }
                        } else if (newG === existing.g) {
                            // Equal path found
                            if (useHeuristic) {
                                // A*: Add duplicate (slides show multiple entries with same cost)
                                open.push({ id: nb.id, g: newG, f: newF, path: newPath });
                                enqueuedCount++;
                                events.push(`Child ${nb.id} enqueued (equal cost path)`);
                                continue;
                            } else {
                                // UCS: Ignore equal
                                continue;
                            }
                        } else {
                            // Worse path found - skip for both A* and UCS
                            continue;
                        }
                    }

                    // A*: If in closed set but not in open set?
                    // According to slides: Do NOT enqueue if already extended with lower or equal cost
                    if (useHeuristic && closedCost !== undefined) {
                        if (closedCost <= newG) {
                            const prevF = closedCost + h;
                            events.push(`Child ${nb.id} NOT enqueued (already extended with f=${prevF} ≤ f=${newF})`);
                            continue;
                        } else {
                            // Better path found to an already extended node - reopen it
                            open.push({ id: nb.id, g: newG, f: newF, path: newPath });
                            enqueuedCount++;
                            const prevF = closedCost + h;
                            events.push(`Child ${nb.id} enqueued (better path, reopened: f=${prevF} → f=${newF})`);
                            continue;
                        }
                    }
                }

                // New Node or Tree Search
                open.push({ id: nb.id, g: newG, f: newF, path: newPath });
                enqueuedCount++;
                events.push(`Child ${nb.id} enqueued`);
            }

            // Log Step (After processing all neighbors)
            const currentQueueSnapshot = snapshotQueue();

            setStepLogs(prev => [
                ...prev,
                {
                    step: prev.length + 1,
                    queue: currentQueueSnapshot,
                    extendedNode: current.id,
                    extendedList: Array.from(closed.keys()),
                    extendedListWithCosts: Object.fromEntries(closed),
                    extendedCount,
                    enqueuedCount,
                    events
                }
            ]);

            await delay(speed);
        }

        showToast(`No path found (${useHeuristic ? 'A*' : 'UCS'})`, 'error');
    };

    // MiniMax (adversarial, tree-based) :contentReference[oaicite:7]{index=7}
    const runMiniMax = async () => {
        if (!startNode) {
            showToast('Set Start node (root) for MiniMax', 'error');
            return;
        }
        const start = startNode!;

        const allZero = nodes.every(n => (heuristics[n.id] || 0) === 0);
        if (allZero) {
            showToast(
                'All node values are 0. Set leaf values (double-click) for MiniMax.',
                'error'
            );
        }

        let stepCount = -1;

        const logStep = (nodeId: string, details: string, value?: number) => {
            stepCount++;
            setStepLogs(prev => [
                ...prev,
                {
                    step: stepCount,
                    queue:
                        value !== undefined
                            ? [{
                                path: [nodeId],
                                cost: value,
                                heuristic: 0,
                                total: value
                            }]
                            : [],
                    extendedNode: nodeId,
                    extendedList: [details],
                    extendedListWithCosts: {},
                    extendedCount: stepCount,
                    enqueuedCount: stepCount,
                    events: [details]
                }
            ]);
        };

        type MinimaxResult = {
            val: number;
            bestPath: string[];
        };

        const minimax = async (
            nodeId: string,
            isMax: boolean,
            path: string[]
        ): Promise<MinimaxResult> => {
            setCurrentPath(path);
            setFrontierNodes(prev => new Set(prev).add(nodeId));
            await delay(speed);

            const neighbors = getNeighbors(nodeId)
                .map(n => n.id)
                .sort((a, b) => a.localeCompare(b));

            if (neighbors.length === 0) {
                const val = heuristics[nodeId] || 0;
                logStep(nodeId, `Leaf: ${val}`, val);
                setVisitedNodes(prev => new Set(prev).add(nodeId));
                setComputedValues(prev => ({ ...prev, [nodeId]: val }));
                return { val, bestPath: path };
            }

            let bestVal = isMax ? -Infinity : Infinity;
            let bestPath: string[] = [...path];
            let hasValidChild = false;

            for (const child of neighbors) {
                if (path.includes(child)) continue;
                hasValidChild = true;
                const result = await minimax(child, !isMax, [...path, child]);
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

            if (!hasValidChild || bestVal === Infinity || bestVal === -Infinity) {
                const val = heuristics[nodeId] || 0;
                logStep(nodeId, `Leaf (dead end): ${val}`, val);
                setComputedValues(prev => ({ ...prev, [nodeId]: val }));
                return { val, bestPath: path };
            }

            logStep(nodeId, `${isMax ? 'MAX' : 'MIN'} updated: ${bestVal}`, bestVal);
            setVisitedNodes(prev => new Set(prev).add(nodeId));
            setComputedValues(prev => ({ ...prev, [nodeId]: bestVal }));
            return { val: bestVal, bestPath };
        };

        logStep(start, 'Start MiniMax', 0);
        const result = await minimax(start, rootIsMax, [start]);
        setCurrentPath(result.bestPath);
        setComputedValues(prev => ({ ...prev, [start]: result.val }));
        showToast(`MiniMax Result: ${result.val}`, 'success');
    };

    // Master dispatcher
    const runAlgorithm = async () => {
        if (!startNode) {
            showToast('Set Start node', 'error');
            return;
        }
        if (algorithm !== 'MiniMax' && !goalNode) {
            showToast('Set Goal node', 'error');
            return;
        }
        if (isAnimating) return;

        resetGraph();
        setIsAnimating(true);

        try {
            if (algorithm === 'MiniMax') {
                await runMiniMax();
            } else if (algorithm === 'BFS') {
                await runBFS();
            } else if (algorithm === 'DFS') {
                await runDFS();
            } else if (algorithm === 'UCS') {
                await runCostSearch(false);
            } else if (algorithm === 'AStar') {
                await runCostSearch(true);
            }
        } finally {
            setIsAnimating(false);
        }
    };

    // --- UI ---
    return (
        <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
            {/* Mobile Menu Button */}
            <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="fixed top-4 left-4 z-50 lg:hidden bg-white p-2 rounded-lg shadow-lg border border-slate-200"
            >
                {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>

            {/* Overlay for mobile */}
            {sidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-30 lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            {/* Sidebar */}
            <div className={`
                w-80 flex flex-col bg-white border-r border-slate-200 z-40 shadow-xl
                fixed lg:relative inset-y-0 left-0
                transform transition-transform duration-300 ease-in-out
                ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
            `}>
                <div className="p-4 border-b border-slate-200 flex justify-between items-center">
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">
                        PathFinder Pro
                    </h1>
                    <button
                        onClick={() => setSidebarOpen(false)}
                        className="lg:hidden text-slate-500 hover:text-slate-700"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
                    {/* Algorithm selection */}
                    <div className="space-y-3">
                        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                            Algorithm
                        </h2>
                        <div className="grid grid-cols-2 gap-2">
                            {['BFS', 'DFS', 'UCS', 'AStar', 'MiniMax'].map(algo => (
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
                            {isAnimating ? (
                                <Settings className="animate-spin w-4 h-4" />
                            ) : (
                                <Play className="w-4 h-4" />
                            )}
                            {isAnimating ? 'Running...' : 'START SEARCH'}
                        </button>
                    </div>

                    {/* Tools */}
                    <div className="space-y-3">
                        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                            Tools
                        </h2>
                        <div className="flex gap-2">
                            {[
                                { id: 'select', icon: MousePointer2, label: 'Select' },
                                { id: 'node', icon: Plus, label: 'Node' },
                                { id: 'edge', icon: ChevronRight, label: 'Edge' },
                            ].map(tool => (
                                <button
                                    key={tool.id}
                                    onClick={() =>
                                        setMode(tool.id as InteractionMode)
                                    }
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
                            <button
                                onClick={randomizeWeights}
                                className="flex-1 p-2 bg-slate-100 rounded-lg text-slate-600 hover:bg-slate-200 flex items-center justify-center gap-2 text-xs font-medium"
                            >
                                <Shuffle className="w-4 h-4" /> Random
                            </button>
                            <button
                                onClick={loadExample1}
                                className="flex-1 p-2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg hover:bg-indigo-100 flex items-center justify-center gap-2 text-xs font-bold transition-all"
                            >
                                <Info className="w-4 h-4" /> Ex 1
                            </button>
                            <button
                                onClick={loadExample2}
                                className="flex-1 p-2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg hover:bg-indigo-100 flex items-center justify-center gap-2 text-xs font-bold transition-all"
                            >
                                <Info className="w-4 h-4" /> Ex 2
                            </button>
                            <button
                                onClick={loadExample3}
                                className="flex-1 p-2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg hover:bg-indigo-100 flex items-center justify-center gap-2 text-xs font-bold transition-all"
                            >
                                <Info className="w-4 h-4" /> Ex 3
                            </button>
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={deleteSelected}
                                disabled={!selectedNode && !selectedEdge}
                                className="flex-1 p-2 bg-slate-100 rounded-lg text-red-500 hover:bg-red-50 disabled:opacity-50 flex justify-center"
                            >
                                <Trash2 className="w-5 h-5" />
                            </button>
                        </div>
                    </div>

                    {/* Heuristics / Values */}
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                                <Table className="w-3 h-3" />
                                {algorithm === 'MiniMax'
                                    ? 'Node Values'
                                    : 'Heuristics'}
                            </h2>
                            {algorithm !== 'MiniMax' && (
                                <button
                                    onClick={() =>
                                        setManualHeuristics(!manualHeuristics)
                                    }
                                    className="text-[10px] text-blue-400 hover:underline"
                                >
                                    {manualHeuristics
                                        ? 'Switch to Auto'
                                        : 'Switch to Manual'}
                                </button>
                            )}
                        </div>

                        {/* Show heuristic arcs only for A* */}
                        {algorithm === 'AStar' && (
                            <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={showIndirect}
                                    onChange={e =>
                                        setShowIndirect(e.target.checked)
                                    }
                                    className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
                                />
                                Show indirect costs (heuristics)
                            </label>
                        )}

                        {/* Tree search toggle */}
                        <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={treeSearch}
                                onChange={e => setTreeSearch(e.target.checked)}
                                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
                            />
                            Tree Search (allow duplicates)
                        </label>

                        {/* Directed / undirected */}
                        <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={isDirected}
                                onChange={e => setIsDirected(e.target.checked)}
                                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
                            />
                            Directed Graph
                        </label>

                        {/* Minimax extras */}
                        {algorithm === 'MiniMax' && (
                            <div className="pt-2 border-t border-slate-200 mt-2">
                                <h3 className="text-[10px] font-bold text-slate-500 uppercase mb-2">
                                    Minimax Settings
                                </h3>
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
                                            onChange={() =>
                                                setRootIsMax(false)
                                            }
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
                                            <th className="pb-1">
                                                {algorithm === 'MiniMax'
                                                    ? 'Value'
                                                    : 'h(n)'}
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {nodes.map(node => (
                                            <tr
                                                key={node.id}
                                                className="border-b border-slate-100 last:border-0"
                                            >
                                                <td className="py-1.5 font-medium text-slate-700">
                                                    {node.id}
                                                </td>
                                                <td className="py-1.5">
                                                    {algorithm === 'MiniMax' ||
                                                        manualHeuristics ? (
                                                        <input
                                                            type="number"
                                                            value={
                                                                heuristics[
                                                                node.id
                                                                ] || 0
                                                            }
                                                            onChange={e =>
                                                                updateHeuristic(
                                                                    node.id,
                                                                    e.target
                                                                        .value
                                                                )
                                                            }
                                                            className="w-12 bg-white border border-slate-200 rounded px-1 text-slate-900 focus:border-blue-500 outline-none"
                                                        />
                                                    ) : (
                                                        <span className="text-blue-600">
                                                            {heuristics[
                                                                node.id
                                                            ] || 0}
                                                        </span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <div className="text-xs text-slate-500 italic p-2">
                                {algorithm === 'MiniMax'
                                    ? 'Set Start node to see values'
                                    : 'Set Start & Goal to see heuristics'}
                            </div>
                        )}
                    </div>

                    {/* Speed slider */}
                    <div className="space-y-2">
                        <div className="flex justify-between text-xs text-slate-500">
                            <span>Speed</span>
                            <span>{speed}ms</span>
                        </div>
                        <input
                            type="range"
                            min="100"
                            max="2000"
                            step="100"
                            value={speed}
                            onChange={e => setSpeed(parseInt(e.target.value))}
                            className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        />
                    </div>

                    <div className="flex gap-2 pt-4">
                        <button
                            onClick={resetGraph}
                            className="flex-1 py-2 text-xs font-medium bg-slate-100 text-slate-600 rounded hover:bg-slate-200"
                        >
                            Reset Path
                        </button>
                        <button
                            onClick={clearAll}
                            className="flex-1 py-2 text-xs font-medium bg-slate-100 text-red-500 rounded hover:bg-slate-200"
                        >
                            Clear All
                        </button>
                    </div>
                </div>
            </div>

            {/* Main canvas */}
            <div className="flex-1 relative bg-slate-50">
                <canvas
                    ref={canvasRef}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onDoubleClick={handleDoubleClick}
                    onContextMenu={handleContextMenu}
                    className={`w-full h-full block touch-none ${mode === 'select' ? 'cursor-default' : 'cursor-crosshair'
                        }`}
                />

                {/* Show Logs Button (when hidden) */}
                {!showLogs && stepLogs.length > 0 && (
                    <button
                        onClick={() => setShowLogs(true)}
                        className="absolute top-4 right-4 bg-blue-600 text-white px-3 py-2 rounded-lg shadow-lg hover:bg-blue-700 flex items-center gap-2 text-sm font-medium"
                    >
                        <List className="w-4 h-4" /> Show Trace
                    </button>
                )}

                {/* Logs */}
                {showLogs && stepLogs.length > 0 && (
                    <div
                        className={`absolute w-full max-w-[95vw] sm:max-w-[500px] lg:w-[500px] max-h-[50vh] lg:max-h-[calc(100vh-2rem)] flex flex-col bg-white/95 backdrop-blur shadow-2xl rounded-xl overflow-hidden transition-shadow ${isDraggingPanel ? 'border-4 border-blue-400 shadow-blue-500/50' : 'border-2 border-slate-300'
                            }`}
                        style={{
                            left: tracePanelPos.x === 0 ? (window.innerWidth < 640 ? '0.5rem' : 'auto') : `${tracePanelPos.x}px`,
                            top: tracePanelPos.x === 0 ? '1rem' : `${tracePanelPos.y}px`,
                            right: tracePanelPos.x === 0 ? (window.innerWidth < 640 ? '0.5rem' : '1rem') : 'auto',
                            cursor: isDraggingPanel ? 'grabbing' : 'default',
                            zIndex: 50
                        }}
                    >
                        <div
                            className="p-3 sm:p-3 py-4 sm:py-3 bg-gradient-to-r from-slate-100 to-slate-50 border-b-2 border-slate-300 flex justify-between items-center cursor-grab active:cursor-grabbing select-none"
                            onMouseDown={handlePanelMouseDown}
                            onTouchStart={handlePanelTouchStart}
                            style={{ touchAction: 'none' }}
                        >
                            <h3 className="font-bold text-sm text-slate-800 flex items-center gap-2">
                                <GripVertical className="w-5 h-5 text-slate-400 sm:hidden" />
                                <List className="w-4 h-4 text-blue-600" />
                                <span className="hidden sm:inline">Trace (Drag to Move)</span>
                                <span className="sm:hidden">Trace</span>
                            </h3>
                            <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                                {tracePanelPos.x !== 0 && (
                                    <button
                                        onClick={() => setTracePanelPos({ x: 0, y: 0 })}
                                        className="text-xs text-blue-600 hover:text-blue-800"
                                        title="Reset position"
                                    >
                                        Reset
                                    </button>
                                )}
                                <button
                                    onClick={() => setShowLogs(false)}
                                    className="text-xs text-slate-500 hover:text-slate-800 lg:hidden"
                                >
                                    Hide
                                </button>
                                <button
                                    onClick={() => setStepLogs([])}
                                    className="text-xs text-slate-500 hover:text-slate-800"
                                >
                                    Clear
                                </button>
                            </div>
                        </div>

                        <div className="overflow-y-auto p-0 custom-scrollbar">
                            <table className="w-full text-xs text-left border-collapse">
                                <thead className="bg-slate-50 text-slate-500 sticky top-0">
                                    <tr>
                                        <th className="p-2 border-b border-slate-200 w-12">
                                            Step
                                        </th>
                                        <th className="p-2 border-b border-slate-200 w-1/4">
                                            Enqueued Paths
                                        </th>
                                        <th className="p-2 border-b border-slate-200">
                                            Events
                                        </th>
                                        {(algorithm === 'AStar' || algorithm === 'UCS') && (
                                            <th className="p-2 border-b border-slate-200 w-1/5">
                                                Extended List
                                            </th>
                                        )}
                                        <th className="p-2 border-b border-slate-200 w-16 text-center">
                                            Ext
                                        </th>
                                        <th className="p-2 border-b border-slate-200 w-16 text-center">
                                            Enq
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {stepLogs.map(log => (
                                        <tr
                                            key={log.step}
                                            className="hover:bg-slate-50 transition-colors"
                                        >
                                            <td className="p-2 font-mono text-slate-500 align-top">
                                                {log.step}.
                                            </td>
                                            <td className="p-2 align-top">
                                                <div className="flex flex-col gap-1">
                                                    {log.queue.length === 0 ? (
                                                        <span className="text-slate-400 italic">
                                                            Empty
                                                        </span>
                                                    ) : (
                                                        <div className="flex flex-wrap gap-1">
                                                            {log.queue.map(
                                                                (
                                                                    item,
                                                                    i
                                                                ) => (
                                                                    <span
                                                                        key={i}
                                                                        className="inline-flex items-center gap-1 bg-white px-1.5 py-0.5 rounded border border-slate-200 shadow-sm"
                                                                    >
                                                                        <span className="font-bold text-indigo-600">
                                                                            {`{${item.path.join(
                                                                                ''
                                                                            )}}`}
                                                                        </span>
                                                                        {algorithm ===
                                                                            'UCS' && (
                                                                                <span className="text-slate-500">
                                                                                    [
                                                                                    {
                                                                                        item.cost
                                                                                    }
                                                                                    ]
                                                                                </span>
                                                                            )}
                                                                        {algorithm ===
                                                                            'AStar' && (
                                                                                <span className="text-slate-500">
                                                                                    [
                                                                                    {
                                                                                        item.total
                                                                                    }
                                                                                    ]
                                                                                </span>
                                                                            )}
                                                                    </span>
                                                                )
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                            <td className="p-2 text-xs text-slate-600 align-top">
                                                {log.events &&
                                                    log.events.length > 0 ? (
                                                    <ul className="list-disc list-inside space-y-0.5">
                                                        {log.events.map(
                                                            (e, i) => (
                                                                <li
                                                                    key={i}
                                                                    className={
                                                                        e.includes('better path') ||
                                                                            e.includes('updated') ||
                                                                            e.includes('Extended List updated') ||
                                                                            e.includes('reopened')
                                                                            ? 'text-green-600 font-bold'
                                                                            : e.includes('NOT enqueued') ||
                                                                                e.includes('NOT extended') ||
                                                                                e.includes('pruned')
                                                                                ? 'text-red-600 font-bold'
                                                                                : ''
                                                                    }
                                                                >
                                                                    {e}
                                                                </li>
                                                            )
                                                        )}
                                                    </ul>
                                                ) : (
                                                    <span className="text-slate-300">
                                                        -
                                                    </span>
                                                )}
                                            </td>
                                            {(algorithm === 'AStar' || algorithm === 'UCS') && (
                                                <td className="p-2 align-top">
                                                    {Object.keys(log.extendedListWithCosts).length === 0 ? (
                                                        <span className="text-slate-300 italic">-</span>
                                                    ) : (
                                                        <div className="flex flex-wrap gap-1">
                                                            {Object.entries(log.extendedListWithCosts)
                                                                .sort(([a], [b]) => a.localeCompare(b))
                                                                .map(([node, gCost]) => {
                                                                    // For A*, display f-cost (g+h). For UCS, display g-cost.
                                                                    const h = algorithm === 'AStar' ? (heuristics[node] ?? 0) : 0;
                                                                    const displayCost = algorithm === 'AStar' ? gCost + h : gCost;
                                                                    return (
                                                                        <span
                                                                            key={node}
                                                                            className="inline-flex items-center gap-0.5 bg-orange-50 px-1.5 py-0.5 rounded border border-orange-200 text-orange-700"
                                                                        >
                                                                            <span className="font-bold">{node}</span>
                                                                            <span className="text-slate-400">:</span>
                                                                            <span>{displayCost}</span>
                                                                        </span>
                                                                    );
                                                                })}
                                                        </div>
                                                    )}
                                                </td>
                                            )}
                                            <td className="p-2 text-center font-mono text-slate-500 align-top">
                                                {log.extendedCount}
                                            </td>
                                            <td className="p-2 text-center font-mono text-slate-500 align-top">
                                                {log.enqueuedCount}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {stepLogs.length > 0 && (
                            <div className="p-2 bg-slate-50 border-t border-slate-200 text-[10px] text-slate-500 flex justify-between">
                                <span>
                                    Extended:{' '}
                                    {
                                        stepLogs[stepLogs.length - 1]
                                            .extendedCount
                                    }
                                </span>
                                <span>
                                    Enqueued:{' '}
                                    {
                                        stepLogs[stepLogs.length - 1]
                                            .enqueuedCount
                                    }
                                </span>
                            </div>
                        )}
                    </div>
                )}

                {/* Instructions - Hidden on mobile */}
                <div className="hidden lg:block absolute bottom-16 left-4 bg-white/90 backdrop-blur border border-slate-200 rounded-xl p-3 shadow-lg pointer-events-none select-none max-w-xs z-10">
                    <div className="text-[11px] text-slate-600 space-y-1">
                        <p>
                            <span className="font-bold text-slate-800">
                                Double-click node
                            </span>{' '}
                            to rename (or set value in Minimax).
                        </p>
                        <p>
                            <span className="font-bold text-slate-800">
                                Right-click
                            </span>{' '}
                            to set Start/Goal.
                        </p>
                        <p>
                            <span className="font-bold text-slate-800">
                                Drag
                            </span>{' '}
                            nodes to move.
                        </p>
                        <p>
                            <span className="font-bold text-slate-800">
                                Click edge
                            </span>{' '}
                            to select; Double-click to edit weight.
                        </p>
                    </div>
                </div>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 lg:left-4 bg-white/90 backdrop-blur border border-slate-200 rounded-xl p-2 lg:p-3 pointer-events-none select-none shadow-lg">
                    <div className="flex flex-wrap gap-2 lg:gap-4 text-[9px] lg:text-[10px] text-slate-600 font-medium">
                        <div className="flex items-center gap-1.5">
                            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />{' '}
                            Start
                        </div>
                        <div className="flex items-center gap-1.5">
                            <div className="w-2.5 h-2.5 rounded-full bg-red-500" />{' '}
                            Goal
                        </div>
                        <div className="flex items-center gap-1.5">
                            <div className="w-2.5 h-2.5 rounded-full bg-orange-200" />{' '}
                            Extended
                        </div>
                        <div className="flex items-center gap-1.5">
                            <div className="w-2.5 h-2.5 rounded-full bg-indigo-300" />{' '}
                            Frontier
                        </div>
                        {algorithm === 'MiniMax' && (
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-pink-600" />{' '}
                                Value
                            </div>
                        )}
                    </div>
                </div>

                {/* Toasts */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col gap-2 z-50 pointer-events-none">
                    {toasts.map(toast => (
                        <div
                            key={toast.id}
                            className="animate-in slide-in-from-bottom-5 fade-in duration-300 flex items-center gap-2 bg-white text-slate-900 px-3 py-2 rounded-lg shadow-xl border border-slate-200 text-sm"
                        >
                            {toast.type === 'success' && (
                                <CheckCircle className="w-4 h-4 text-green-500" />
                            )}
                            {toast.type === 'error' && (
                                <AlertCircle className="w-4 h-4 text-red-500" />
                            )}
                            <span>{toast.message}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default GraphVisualizer;
