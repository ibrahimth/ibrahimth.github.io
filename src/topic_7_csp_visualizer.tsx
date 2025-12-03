import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Pause, RotateCcw, StepForward, CheckCircle, XCircle, Settings, Info } from 'lucide-react';
type NodeId = 'V1' | 'V2' | 'V3' | 'V4' | 'V5' | 'V6';
type Color = 'R' | 'G' | 'B' | 'Y' | null;
interface NodeData {
  id: NodeId;
  label: string;
  x: number;
  y: number;
}
interface Edge {
  source: NodeId;
  target: NodeId;
}
const NODES: NodeData[] = [
  { id: 'V1', label: 'Tabuk', x: 100, y: 80 },
  { id: 'V2', label: 'Hail', x: 300, y: 80 },
  { id: 'V6', label: 'Medina', x: 200, y: 220 },
  { id: 'V3', label: 'Qassim', x: 400, y: 220 },
  { id: 'V5', label: 'Makkah', x: 200, y: 380 },
  { id: 'V4', label: 'Riyadh', x: 400, y: 380 },
];
const EDGES: Edge[] = [
  { source: 'V1', target: 'V2' },
  { source: 'V1', target: 'V6' },
  { source: 'V2', target: 'V6' },
  { source: 'V2', target: 'V3' },
  { source: 'V6', target: 'V3' },
  { source: 'V6', target: 'V4' },
  { source: 'V6', target: 'V5' },
  { source: 'V3', target: 'V4' },
  { source: 'V4', target: 'V5' },
];
const COLORS: Color[] = ['R', 'G', 'B', 'Y'];
const COLOR_MAP: Record<string, string> = {
  R: 'bg-red-500 border-red-700',
  G: 'bg-green-500 border-green-700',
  B: 'bg-blue-500 border-blue-700',
  Y: 'bg-yellow-400 border-yellow-600',
  null: 'bg-white border-gray-400',
};
const COLOR_HEX: Record<string, string> = {
  R: '#ef4444',
  G: '#22c55e',
  B: '#3b82f6',
  Y: '#facc15',
  null: '#ffffff',
};
const getNeighbors = (id: NodeId): NodeId[] => {
  const neighbors: NodeId[] = [];
  EDGES.forEach(e => {
    if (e.source === id) neighbors.push(e.target);
    if (e.target === id) neighbors.push(e.source);
  });
  return neighbors;
};
export default function CSPVisualizer() {
  const [assignments, setAssignments] = useState<Record<NodeId, Color>>({ V1: null, V2: null, V3: null, V4: null, V5: null, V6: null });
  const [domains, setDomains] = useState<Record<NodeId, Color[]>>({ V1: [...COLORS], V2: [...COLORS], V3: [...COLORS], V4: [...COLORS], V5: [...COLORS], V6: [...COLORS] });
  const [currentNode, setCurrentNode] = useState<NodeId | null>(null);
  const [solving, setSolving] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [mode, setMode] = useState<'backtracking' | 'forwardChecking'>('backtracking');
  const [useMRV, setUseMRV] = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const [status, setStatus] = useState<'idle' | 'running' | 'solved' | 'failed'>('idle');
  const [conflictEdge, setConflictEdge] = useState<Edge | null>(null);
  const solverRef = useRef<any>(null);
  const isValid = (node: NodeId, color: Color, currentAssignments: Record<NodeId, Color>) => {
    const neighbors = getNeighbors(node);
    for (const neighbor of neighbors) {
      if (currentAssignments[neighbor] === color) return false;
    }
    return true;
  };
  const getUnassignedVar = (currentAssignments: Record<NodeId, Color>, currentDomains: Record<NodeId, Color[]>) => {
    let unassigned = NODES.map(n => n.id).filter(id => currentAssignments[id] === null);
    if (unassigned.length === 0) return null;
    if (useMRV) {
      unassigned.sort((a, b) => {
        const lenA = currentDomains[a].length;
        const lenB = currentDomains[b].length;
        if (lenA !== lenB) return lenA - lenB;
        return a.localeCompare(b);
      });
    } else {
      unassigned.sort((a,b) => a.localeCompare(b));
    }
    return unassigned[0];
  };
  const stepSolver = useCallback(() => {
    if (status === 'solved' || status === 'failed') return;
    setHistory(prevStack => {
      const newStack = [...prevStack];
      if (newStack.length === 0) {
        newStack.push({ assignments: { V1: null, V2: null, V3: null, V4: null, V5: null, V6: null }, domains: { V1: ['R','G','B','Y'], V2: ['R','G','B','Y'], V3: ['R','G','B','Y'], V4: ['R','G','B','Y'], V5: ['R','G','B','Y'], V6: ['R','G','B','Y'] }, triedValues: {}, backtracking: false });
      }
      const currentState = newStack[newStack.length - 1];
      const { assignments, domains, triedValues } = currentState;
      const variable = getUnassignedVar(assignments, domains);
      if (!variable) {
        setStatus('solved');
        setSolving(false);
        return newStack;
      }
      setCurrentNode(variable);
      const currentTried = triedValues[variable] || [];
      const availableValues = domains[variable].filter(c => !currentTried.includes(c));
      if (availableValues.length === 0) {
        if (newStack.length === 1) {
          setStatus('failed');
          setSolving(false);
          return newStack;
        }
        newStack.pop();
        setConflictEdge(null);
        const prev = newStack[newStack.length - 1];
        setAssignments(prev.assignments);
        setDomains(prev.domains);
        return newStack;
      }
      const colorToTry = availableValues[0];
      const valid = isValid(variable, colorToTry, assignments);
      setAssignments({ ...assignments, [variable]: colorToTry });
      currentState.triedValues = { ...triedValues, [variable]: [...currentTried, colorToTry] };
      if (valid) {
        setConflictEdge(null);
        let newDomains = { ...domains };
        if (mode === 'forwardChecking') {
          const neighbors = getNeighbors(variable);
          let emptyDomainFound = false;
          neighbors.forEach(neighbor => {
            if (assignments[neighbor] === null) {
              newDomains[neighbor] = newDomains[neighbor].filter(c => c !== colorToTry);
              if (newDomains[neighbor].length === 0) emptyDomainFound = true;
            }
          });
          if (emptyDomainFound) {
            setDomains(newDomains);
            return newStack;
          }
        }
        newStack.push({ assignments: { ...assignments, [variable]: colorToTry }, domains: newDomains, triedValues: {}, backtracking: false });
        setDomains(newDomains);
      } else {
        const neighbors = getNeighbors(variable);
        const conflict = neighbors.find(n => assignments[n] === colorToTry);
        if (conflict) setConflictEdge({ source: variable, target: conflict });
      }
      return newStack;
    });
  }, [status, mode, useMRV]);
  useEffect(() => {
    let interval: any;
    if (solving && !paused) {
      interval = setInterval(() => {
        stepSolver();
      }, speed);
    }
    return () => clearInterval(interval);
  }, [solving, paused, speed, stepSolver]);
  const handleReset = () => {
    setAssignments({ V1: null, V2: null, V3: null, V4: null, V5: null, V6: null });
    setDomains({ V1: ['R','G','B','Y'], V2: ['R','G','B','Y'], V3: ['R','G','B','Y'], V4: ['R','G','B','Y'], V5: ['R','G','B','Y'], V6: ['R','G','B','Y'] });
    setHistory([]);
    setStatus('idle');
    setSolving(false);
    setPaused(false);
    setCurrentNode(null);
    setConflictEdge(null);
  };
  const handleStart = () => {
    if (status === 'solved' || status === 'failed') handleReset();
    setStatus('running');
    setSolving(true);
    setPaused(false);
  };
  const handleManualColor = (nodeId: NodeId, color: Color) => {
    if (solving) return;
    setAssignments(prev => ({ ...prev, [nodeId]: color }));
    const valid = isValid(nodeId, color, assignments);
    if (!valid && color !== null) {
      const neighbors = getNeighbors(nodeId);
      const conflict = neighbors.find(n => assignments[n] === color);
      if (conflict) setConflictEdge({ source: nodeId, target: conflict });
    } else {
      setConflictEdge(null);
    }
  };
  const renderNode = (node: NodeData) => {
    const color = assignments[node.id];
    const isCurrent = currentNode === node.id && solving;
    const domainSize = domains[node.id].length;
    return (
      <g key={node.id} className="transition-all duration-300" onClick={() => {
        if (!solving) {
          const idx = COLORS.indexOf(color as any);
          const nextColor = COLORS[(idx + 1) % (COLORS.length + 1)] || null;
          handleManualColor(node.id, nextColor);
        }
      }}>
        {isCurrent && (
          <circle cx={node.x} cy={node.y} r="35" className="fill-blue-200 animate-pulse opacity-50" />
        )}
        <circle cx={node.x} cy={node.y} r="25" className={`stroke-2 ${isCurrent ? 'stroke-blue-600' : 'stroke-gray-600'} cursor-pointer transition-colors duration-300`} fill={COLOR_HEX[color as string] || '#fff'} />
        <text x={node.x} y={node.y + 5} textAnchor="middle" className="font-bold pointer-events-none select-none text-sm fill-gray-800">{node.id}</text>
        <text x={node.x} y={node.y - 35} textAnchor="middle" className="text-xs font-semibold fill-gray-600 select-none">{node.label}</text>
        {solving && mode === 'forwardChecking' && !color && (
          <g transform={`translate(${node.x - 20}, ${node.y + 35})`}>
            {COLORS.map((c, i) => (
              <circle key={c} cx={i * 12} cy={0} r={4} fill={COLOR_HEX[c as string]} opacity={domains[node.id].includes(c) ? 1 : 0.1} />
            ))}
          </g>
        )}
        {solving && useMRV && !color && (
          <text x={node.x + 30} y={node.y} className="text-xs fill-purple-600 font-bold">D:{domainSize}</text>
        )}
      </g>
    );
  };
  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto p-4 bg-gray-50 rounded-lg shadow-xl">
      <div className="flex justify-between w-full mb-6 items-center border-b pb-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">CSP Map Coloring</h1>
          <p className="text-sm text-gray-500">Saudi Arabia Region Example (Topic 7 CSP.pdf)</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-sm font-bold ${status === 'solved' ? 'bg-green-100 text-green-700' : status === 'failed' ? 'bg-red-100 text-red-700' : status === 'running' ? 'bg-blue-100 text-blue-700' : 'bg-gray-200 text-gray-700'}`}>{status.toUpperCase()}</span>
        </div>
      </div>
      <div className="flex flex-col md:flex-row gap-6 w-full">
        <div className="w-full md:w-1/3 space-y-6 bg-white p-4 rounded-lg shadow-sm h-fit">
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-700 flex items-center gap-2"><Settings className="w-4 h-4"/> Algorithm Settings </h3>
            <div className="flex flex-col gap-2 p-2 bg-gray-50 rounded border">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input type="radio" name="mode" checked={mode === 'backtracking'} onChange={() => setMode('backtracking')} disabled={solving} className="text-blue-600" />
                <span className="text-sm">Standard Backtracking</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input type="radio" name="mode" checked={mode === 'forwardChecking'} onChange={() => setMode('forwardChecking')} disabled={solving} className="text-blue-600" />
                <span className="text-sm">Forward Checking</span>
              </label>
            </div>
            <div className="flex items-center justify-between p-2 bg-gray-50 rounded border">
              <span className="text-sm">Heuristic: MRV</span>
              <button onClick={() => setUseMRV(!useMRV)} disabled={solving} className={`w-10 h-6 rounded-full p-1 transition-colors ${useMRV ? 'bg-purple-600' : 'bg-gray-300'}`}> <div className={`w-4 h-4 rounded-full bg-white transition-transform ${useMRV ? 'translate-x-4' : ''}`} /></button>
            </div>
          </div>
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-700">Playback</h3>
            <div className="flex gap-2 justify-center">
              {!solving || paused ? (
                <button onClick={handleStart} className="flex items-center gap-1 bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
                  <Play className="w-4 h-4" /> {status === 'running' ? 'Resume' : 'Start'}
                </button>
              ) : (
                <button onClick={() => setPaused(true)} className="flex items-center gap-1 bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600">
                  <Pause className="w-4 h-4" /> Pause
                </button>
              )}
              <button onClick={handleReset} className="flex items-center gap-1 bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700">
                <RotateCcw className="w-4 h-4" /> Reset
              </button>
            </div>
            <div className="flex gap-2 justify-center">
              <button onClick={() => { setPaused(true); stepSolver(); }} disabled={status === 'solved' || status === 'failed'} className="flex items-center gap-1 bg-blue-100 text-blue-700 px-4 py-2 rounded hover:bg-blue-200 w-full justify-center disabled:opacity-50">
                <StepForward className="w-4 h-4" /> Step
              </button>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-gray-500">Speed: {speed}ms</label>
              <input type="range" min="50" max="1500" step="50" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full" />
            </div>
          </div>
          <div className="bg-blue-50 p-3 rounded text-xs text-blue-800 space-y-1">
            <p className="font-bold flex items-center gap-1"><Info className="w-3 h-3"/> Instructions</p>
            <p>1. Select an algorithm mode.</p>
            <p>2. Click <b>Start</b> to watch the CSP solver.</p>
            <p>3. Or click nodes manually to color them.</p>
            <p>4. Enable <b>MRV</b> to pick the hardest variable first.</p>
          </div>
        </div>
        <div className="w-full md:w-2/3 bg-white border rounded-lg shadow-inner relative h-[500px] flex items-center justify-center overflow-hidden">
          <svg width="100%" height="100%" viewBox="0 0 500 500" className="select-none">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#999" />
              </marker>
            </defs>
            {EDGES.map((edge, idx) => {
              const source = NODES.find(n => n.id === edge.source)!;
              const target = NODES.find(n => n.id === edge.target)!;
              const isConflict = conflictEdge && ((conflictEdge.source === edge.source && conflictEdge.target === edge.target) || (conflictEdge.source === edge.target && conflictEdge.target === edge.source));
              return (
                <line key={idx} x1={source.x} y1={source.y} x2={target.x} y2={target.y} stroke={isConflict ? 'red' : '#e5e7eb'} strokeWidth={isConflict ? 4 : 2} className="transition-colors duration-200" />
              );
            })}
            {NODES.map(node => renderNode(node))}
          </svg>
          <div className="absolute top-4 right-4 bg-white/90 p-2 rounded shadow text-xs border">
            <div className="font-bold mb-1">Colors</div>
            {COLORS.map(c => (
              <div key={c} className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 rounded-full border" style={{backgroundColor: COLOR_HEX[c!]}}></div>
                <span>{c}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}