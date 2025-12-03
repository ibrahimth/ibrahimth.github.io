import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Pause, RotateCcw, StepForward, CheckCircle, XCircle, Settings, Info, ArrowRight, List } from 'lucide-react';

// --- Types & Interfaces ---

type Value = string | number;

interface NodeData {
  id: string;
  label: string;
  x: number;
  y: number;
}

interface Edge {
  source: string;
  target: string;
}

interface Constraint {
  id: string;
  scope: [string, string]; // [var1, var2]
  check: (val1: Value, val2: Value) => boolean;
  description: string;
}

interface CSPProblem {
  id: string;
  name: string;
  description: string;
  variables: NodeData[];
  domains: Record<string, Value[]>;
  edges: Edge[];
  constraints: Constraint[];
  isMapColoring?: boolean; // Special rendering for map coloring
}

// --- Problem Definitions ---

// 1. Saudi Map Coloring
const SAUDI_MAP_NODES: NodeData[] = [
  { id: 'V1', label: 'Tabuk', x: 100, y: 80 },
  { id: 'V2', label: 'Hail', x: 300, y: 80 },
  { id: 'V6', label: 'Medina', x: 200, y: 220 },
  { id: 'V3', label: 'Qassim', x: 400, y: 220 },
  { id: 'V5', label: 'Makkah', x: 200, y: 380 },
  { id: 'V4', label: 'Riyadh', x: 400, y: 380 },
];

const SAUDI_MAP_EDGES: Edge[] = [
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

const COLORS = ['R', 'G', 'B', 'Y'];
const COLOR_HEX: Record<string, string> = {
  R: '#ef4444', G: '#22c55e', B: '#3b82f6', Y: '#facc15', null: '#ffffff',
};

// Helper to generate inequality constraints for map coloring
const generateMapConstraints = (edges: Edge[]): Constraint[] => {
  return edges.map((e, idx) => ({
    id: `neq_${idx}`,
    scope: [e.source, e.target],
    check: (v1, v2) => v1 !== v2,
    description: `${e.source} ≠ ${e.target}`
  }));
};

const PROBLEM_SAUDI_MAP: CSPProblem = {
  id: 'saudi_map',
  name: 'Saudi Arabia Map Coloring',
  description: 'Color the regions such that no two adjacent regions have the same color.',
  variables: SAUDI_MAP_NODES,
  domains: SAUDI_MAP_NODES.reduce((acc, node) => ({ ...acc, [node.id]: [...COLORS] }), {}),
  edges: SAUDI_MAP_EDGES,
  constraints: generateMapConstraints(SAUDI_MAP_EDGES),
  isMapColoring: true,
};

// 2. AC-3 Example (X, Y, Z)
const AC3_NODES: NodeData[] = [
  { id: 'X', label: 'Variable X', x: 250, y: 100 },
  { id: 'Y', label: 'Variable Y', x: 100, y: 300 },
  { id: 'Z', label: 'Variable Z', x: 400, y: 300 },
];

const AC3_EDGES: Edge[] = [
  { source: 'X', target: 'Y' },
  { source: 'X', target: 'Z' },
];

const AC3_DOMAIN = [1, 2, 3, 4, 5, 6];

const PROBLEM_AC3_EXAMPLE: CSPProblem = {
  id: 'ac3_example',
  name: 'AC-3 Algorithm Example',
  description: 'Variables X, Y, Z with domains {1..6}. Constraints: X < Y and Z < X - 2.',
  variables: AC3_NODES,
  domains: {
    X: [...AC3_DOMAIN],
    Y: [...AC3_DOMAIN],
    Z: [...AC3_DOMAIN],
  },
  edges: AC3_EDGES,
  constraints: [
    {
      id: 'c1',
      scope: ['X', 'Y'],
      check: (x, y) => (x as number) < (y as number),
      description: 'X < Y'
    },
    {
      id: 'c2',
      scope: ['Z', 'X'],
      check: (z, x) => (z as number) < (x as number) - 2,
      description: 'Z < X - 2'
    }
  ],
  isMapColoring: false,
};

// --- Visualizer Component ---

export default function CSPVisualizer() {
  // Problem State
  const [problem, setProblem] = useState<CSPProblem>(PROBLEM_SAUDI_MAP);

  // Solver State
  const [assignments, setAssignments] = useState<Record<string, Value | null>>({});
  const [domains, setDomains] = useState<Record<string, Value[]>>({});
  const [status, setStatus] = useState<'idle' | 'running' | 'solved' | 'failed'>('idle');
  const [solving, setSolving] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(800);

  // Algorithm Selection
  const [algorithm, setAlgorithm] = useState<'backtracking' | 'forwardChecking' | 'ac3'>('backtracking');
  const [useMRV, setUseMRV] = useState(false);

  // Visualization State
  const [currentNode, setCurrentNode] = useState<string | null>(null);
  const [conflictEdge, setConflictEdge] = useState<Edge | null>(null);
  const [ac3Queue, setAc3Queue] = useState<[string, string][]>([]); // Queue of arcs (Xi, Xj)
  const [currentArc, setCurrentArc] = useState<[string, string] | null>(null);
  const [removedValues, setRemovedValues] = useState<{ node: string, values: Value[] } | null>(null);
  const [history, setHistory] = useState<any[]>([]); // Stack for backtracking
  const [log, setLog] = useState<string[]>([]);

  // Initialize/Reset
  useEffect(() => {
    resetProblem();
  }, [problem]);

  const resetProblem = () => {
    const initialAssignments: Record<string, Value | null> = {};
    const initialDomains: Record<string, Value[]> = {};

    problem.variables.forEach(v => {
      initialAssignments[v.id] = null;
      initialDomains[v.id] = [...problem.domains[v.id]];
    });

    setAssignments(initialAssignments);
    setDomains(initialDomains);
    setStatus('idle');
    setSolving(false);
    setPaused(false);
    setCurrentNode(null);
    setConflictEdge(null);
    setAc3Queue([]);
    setCurrentArc(null);
    setRemovedValues(null);
    setHistory([]);
    setLog([]);
  };

  const addLog = (msg: string) => {
    setLog(prev => [...prev.slice(-4), msg]);
  };

  // --- Helper Functions ---

  const getNeighbors = (nodeId: string): string[] => {
    const neighbors: string[] = [];
    problem.edges.forEach(e => {
      if (e.source === nodeId) neighbors.push(e.target);
      if (e.target === nodeId) neighbors.push(e.source);
    });
    return neighbors;
  };

  const checkConstraint = (var1: string, val1: Value, var2: string, val2: Value): boolean => {
    // Find constraints involving these two variables
    const relevantConstraints = problem.constraints.filter(c =>
      (c.scope[0] === var1 && c.scope[1] === var2) ||
      (c.scope[0] === var2 && c.scope[1] === var1)
    );

    // If no constraint, it's consistent
    if (relevantConstraints.length === 0) return true;

    // Check all relevant constraints
    for (const c of relevantConstraints) {
      const v1 = c.scope[0] === var1 ? val1 : val2;
      const v2 = c.scope[1] === var2 ? val2 : val1;
      if (!c.check(v1, v2)) return false;
    }
    return true;
  };

  const isValid = (node: string, value: Value, currentAssignments: Record<string, Value | null>) => {
    const neighbors = getNeighbors(node);
    for (const neighbor of neighbors) {
      const neighborVal = currentAssignments[neighbor];
      if (neighborVal !== null) {
        if (!checkConstraint(node, value, neighbor, neighborVal)) return false;
      }
    }
    return true;
  };

  // --- AC-3 Logic ---

  const initAC3 = () => {
    // 1. Initialize queue with all arcs
    const queue: [string, string][] = [];
    problem.constraints.forEach(c => {
      queue.push([c.scope[0], c.scope[1]]);
      queue.push([c.scope[1], c.scope[0]]);
    });
    setAc3Queue(queue);
    addLog(`Initialized AC-3 Queue with ${queue.length} arcs.`);
    return queue;
  };

  const revise = (Xi: string, Xj: string, currentDomains: Record<string, Value[]>): { revised: boolean, newDomain: Value[], removed: Value[] } => {
    let revised = false;
    const domainXi = [...currentDomains[Xi]];
    const domainXj = currentDomains[Xj];
    const newDomainXi: Value[] = [];
    const removed: Value[] = [];

    for (const x of domainXi) {
      // Is there some value y in Dj such that constraint is satisfied?
      let consistent = false;
      for (const y of domainXj) {
        if (checkConstraint(Xi, x, Xj, y)) {
          consistent = true;
          break;
        }
      }
      if (consistent) {
        newDomainXi.push(x);
      } else {
        removed.push(x);
        revised = true;
      }
    }

    return { revised, newDomain: newDomainXi, removed };
  };

  // --- Step Logic ---

  const stepSolver = useCallback(() => {
    if (status === 'solved' || status === 'failed') return;

    // === AC-3 Algorithm ===
    if (algorithm === 'ac3') {
      setAc3Queue(prevQueue => {
        const queue = [...prevQueue];

        // If queue empty, we are done
        if (queue.length === 0) {
          setStatus('solved');
          setSolving(false);
          addLog("AC-3 Finished: Arc Consistency enforced.");
          return queue;
        }

        // Pop arc
        const [Xi, Xj] = queue.shift()!;
        setCurrentArc([Xi, Xj]);
        setCurrentNode(Xi); // Highlight source node

        // Revise
        const { revised, newDomain, removed } = revise(Xi, Xj, domains);

        if (revised) {
          setRemovedValues({ node: Xi, values: removed });
          addLog(`Revised ${Xi} against ${Xj}: Removed [${removed.join(', ')}]`);

          const newDomains = { ...domains, [Xi]: newDomain };
          setDomains(newDomains);

          if (newDomain.length === 0) {
            setStatus('failed');
            setSolving(false);
            addLog(`Domain of ${Xi} is empty! Failure.`);
            return [];
          }

          // Add neighbors to queue (except Xj)
          const neighbors = getNeighbors(Xi);
          neighbors.forEach(Xk => {
            if (Xk !== Xj) {
              queue.push([Xk, Xi]);
            }
          });
        } else {
          setRemovedValues(null);
          // addLog(`Checked ${Xi} -> ${Xj}: OK`);
        }

        return queue;
      });
      return;
    }

    // === Backtracking / Forward Checking ===
    setHistory(prevStack => {
      const newStack = [...prevStack];

      // Initialize stack if empty
      if (newStack.length === 0) {
        // Initial state
        newStack.push({
          assignments: { ...assignments },
          domains: JSON.parse(JSON.stringify(domains)),
          triedValues: {},
          backtracking: false
        });
      }

      const currentState = newStack[newStack.length - 1];
      const { assignments: currAssignments, domains: currDomains, triedValues } = currentState;

      // 1. Select Unassigned Variable
      let unassigned = problem.variables.map(n => n.id).filter(id => currAssignments[id] === null);

      if (unassigned.length === 0) {
        setStatus('solved');
        setSolving(false);
        setAssignments(currAssignments);
        addLog("Solution Found!");
        return newStack;
      }

      // MRV Heuristic
      if (useMRV) {
        unassigned.sort((a, b) => {
          const lenA = currDomains[a].length;
          const lenB = currDomains[b].length;
          return lenA - lenB || a.localeCompare(b);
        });
      } else {
        unassigned.sort((a, b) => a.localeCompare(b));
      }

      const variable = unassigned[0];
      setCurrentNode(variable);

      // 2. Select Value
      const currentTried = triedValues[variable] || [];
      const availableValues = currDomains[variable].filter((c: Value) => !currentTried.includes(c));

      if (availableValues.length === 0) {
        // Backtrack
        if (newStack.length === 1) {
          setStatus('failed');
          setSolving(false);
          addLog("No solution found.");
          return newStack;
        }
        addLog(`Backtracking from ${variable}...`);
        newStack.pop();
        setConflictEdge(null);
        const prev = newStack[newStack.length - 1];
        setAssignments(prev.assignments);
        setDomains(prev.domains);
        return newStack;
      }

      const valueToTry = availableValues[0];

      // 3. Check Consistency
      const valid = isValid(variable, valueToTry, currAssignments);

      // Update tried values for this state
      currentState.triedValues = { ...triedValues, [variable]: [...currentTried, valueToTry] };

      if (valid) {
        setConflictEdge(null);
        const newAssignments = { ...currAssignments, [variable]: valueToTry };
        let newDomains = JSON.parse(JSON.stringify(currDomains));

        // Forward Checking
        if (algorithm === 'forwardChecking') {
          const neighbors = getNeighbors(variable);
          let emptyDomainFound = false;

          for (const neighbor of neighbors) {
            if (newAssignments[neighbor] === null) {
              // Remove inconsistent values from neighbor domain
              newDomains[neighbor] = newDomains[neighbor].filter((val: Value) =>
                checkConstraint(neighbor, val, variable, valueToTry)
              );

              if (newDomains[neighbor].length === 0) {
                emptyDomainFound = true;
                addLog(`Forward Check: Domain of ${neighbor} became empty.`);
                break;
              }
            }
          }

          if (emptyDomainFound) {
            // If FC fails, we don't push new state, just continue loop (try next value)
            // But here we return to let the next 'tick' handle the next value
            return newStack;
          }
        }

        // Push new state
        newStack.push({
          assignments: newAssignments,
          domains: newDomains,
          triedValues: {},
          backtracking: false
        });
        setAssignments(newAssignments);
        setDomains(newDomains);
        addLog(`Assigned ${variable} = ${valueToTry}`);
      } else {
        // Visualize conflict
        const neighbors = getNeighbors(variable);
        const conflictNode = neighbors.find(n => currAssignments[n] !== null && !checkConstraint(variable, valueToTry, n, currAssignments[n]!));
        if (conflictNode) {
          setConflictEdge({ source: variable, target: conflictNode });
          addLog(`Conflict: ${variable}=${valueToTry} vs ${conflictNode}`);
        }
      }

      return newStack;
    });

  }, [status, algorithm, useMRV, domains, assignments, problem]);

  // Timer for animation
  useEffect(() => {
    let interval: any;
    if (solving && !paused) {
      interval = setInterval(() => {
        stepSolver();
      }, speed);
    }
    return () => clearInterval(interval);
  }, [solving, paused, speed, stepSolver]);

  // --- Handlers ---

  const handleStart = () => {
    if (status === 'solved' || status === 'failed') resetProblem();

    if (algorithm === 'ac3' && ac3Queue.length === 0) {
      initAC3();
    }

    setStatus('running');
    setSolving(true);
    setPaused(false);
  };

  const handleProblemChange = (p: CSPProblem) => {
    setProblem(p);
  };

  const renderNode = (node: NodeData) => {
    const value = assignments[node.id];
    const isCurrent = currentNode === node.id && solving;
    const domain = domains[node.id] || [];

    // Map Coloring Specific Rendering
    if (problem.isMapColoring) {
      return (
        <g key={node.id} className="transition-all duration-300" onClick={() => { }}>
          {isCurrent && (
            <circle cx={node.x} cy={node.y} r="35" className="fill-blue-200 animate-pulse opacity-50" />
          )}
          <circle
            cx={node.x} cy={node.y} r="25"
            className={`stroke-2 ${isCurrent ? 'stroke-blue-600' : 'stroke-gray-600'} transition-colors duration-300`}
            fill={value ? COLOR_HEX[value as string] : '#fff'}
          />
          <text x={node.x} y={node.y + 5} textAnchor="middle" className="font-bold pointer-events-none select-none text-sm fill-gray-800">
            {node.id}
          </text>
          <text x={node.x} y={node.y - 35} textAnchor="middle" className="text-xs font-semibold fill-gray-600 select-none">
            {node.label}
          </text>

          {/* Domain Dots */}
          {solving && !value && (
            <g transform={`translate(${node.x - 20}, ${node.y + 35})`}>
              {COLORS.map((c, i) => (
                <circle key={c} cx={i * 12} cy={0} r={4} fill={COLOR_HEX[c]} opacity={domain.includes(c) ? 1 : 0.1} />
              ))}
            </g>
          )}
        </g>
      );
    }

    // Generic / AC-3 Rendering
    return (
      <g key={node.id} className="transition-all duration-300">
        {isCurrent && (
          <rect x={node.x - 55} y={node.y - 55} width="110" height="110" rx="10" className="fill-blue-100 animate-pulse opacity-50" />
        )}
        <circle
          cx={node.x} cy={node.y} r="40"
          className={`stroke-2 ${isCurrent ? 'stroke-blue-600' : 'stroke-gray-600'} fill-white transition-colors duration-300`}
        />
        <text x={node.x} y={node.y - 50} textAnchor="middle" className="text-sm font-bold fill-gray-700">{node.label}</text>
        <text x={node.x} y={node.y + 5} textAnchor="middle" className="text-xl font-bold fill-black">
          {value !== null ? value : node.id}
        </text>

        {/* Domain List */}
        <g transform={`translate(${node.x + 50}, ${node.y - 30})`}>
          <rect x="0" y="0" width="80" height={Math.max(20, domain.length * 15 + 5)} rx="4" className="fill-white stroke-gray-300 stroke-1 opacity-90" />
          <text x="5" y="15" className="text-xs font-bold fill-gray-500">Domain:</text>
          {domain.map((d, i) => (
            <text key={d} x="10" y={30 + i * 15} className={`text-xs ${removedValues?.node === node.id && removedValues.values.includes(d) ? 'fill-red-500 line-through' : 'fill-black'}`}>
              {d}
            </text>
          ))}
        </g>
      </g>
    );
  };

  return (
    <div className="flex flex-col items-center w-full max-w-6xl mx-auto p-4 bg-gray-50 rounded-lg shadow-xl font-sans">

      {/* Header */}
      <div className="flex justify-between w-full mb-6 items-center border-b pb-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">CSP Visualizer</h1>
          <p className="text-sm text-gray-500">Constraint Satisfaction Problems: Map Coloring & AC-3</p>
        </div>
        <div className="flex items-center gap-4">
          <select
            className="p-2 border rounded bg-white text-sm"
            value={problem.id}
            onChange={(e) => handleProblemChange(e.target.value === 'saudi_map' ? PROBLEM_SAUDI_MAP : PROBLEM_AC3_EXAMPLE)}
            disabled={solving}
          >
            <option value="saudi_map">Saudi Map Coloring</option>
            <option value="ac3_example">AC-3 Example (X, Y, Z)</option>
          </select>
          <span className={`px-3 py-1 rounded-full text-sm font-bold ${status === 'solved' ? 'bg-green-100 text-green-700' : status === 'failed' ? 'bg-red-100 text-red-700' : status === 'running' ? 'bg-blue-100 text-blue-700' : 'bg-gray-200 text-gray-700'}`}>
            {status.toUpperCase()}
          </span>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6 w-full">

        {/* Controls Panel */}
        <div className="w-full lg:w-1/4 space-y-6 bg-white p-4 rounded-lg shadow-sm h-fit">

          {/* Algorithm Settings */}
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-700 flex items-center gap-2"><Settings className="w-4 h-4" /> Algorithm </h3>
            <div className="flex flex-col gap-2 p-2 bg-gray-50 rounded border">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input type="radio" name="algo" checked={algorithm === 'backtracking'} onChange={() => setAlgorithm('backtracking')} disabled={solving} className="text-blue-600" />
                <span className="text-sm">Backtracking</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input type="radio" name="algo" checked={algorithm === 'forwardChecking'} onChange={() => setAlgorithm('forwardChecking')} disabled={solving} className="text-blue-600" />
                <span className="text-sm">Forward Checking</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input type="radio" name="algo" checked={algorithm === 'ac3'} onChange={() => setAlgorithm('ac3')} disabled={solving} className="text-blue-600" />
                <span className="text-sm">AC-3 (Arc Consistency)</span>
              </label>
            </div>

            {algorithm !== 'ac3' && (
              <div className="flex items-center justify-between p-2 bg-gray-50 rounded border">
                <span className="text-sm">Heuristic: MRV</span>
                <button onClick={() => setUseMRV(!useMRV)} disabled={solving} className={`w-10 h-6 rounded-full p-1 transition-colors ${useMRV ? 'bg-purple-600' : 'bg-gray-300'}`}>
                  <div className={`w-4 h-4 rounded-full bg-white transition-transform ${useMRV ? 'translate-x-4' : ''}`} />
                </button>
              </div>
            )}
          </div>

          {/* Playback Controls */}
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
              <button onClick={resetProblem} className="flex items-center gap-1 bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700">
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

          {/* Log / Info */}
          <div className="bg-slate-800 p-3 rounded text-xs text-green-400 font-mono h-40 overflow-y-auto">
            <div className="font-bold text-white mb-1 border-b border-gray-600 pb-1">Execution Log</div>
            {log.length === 0 && <span className="text-gray-500">Ready...</span>}
            {log.map((l, i) => (
              <div key={i} className="mb-1">&gt; {l}</div>
            ))}
          </div>

        </div>

        {/* Visualization Area */}
        <div className="w-full lg:w-3/4 flex flex-col gap-4">

          {/* Canvas */}
          <div className="bg-white border rounded-lg shadow-inner relative h-[500px] w-full overflow-hidden">
            <svg width="100%" height="100%" viewBox="0 0 600 500" className="select-none">
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#999" />
                </marker>
              </defs>

              {/* Edges */}
              {problem.edges.map((edge, idx) => {
                const source = problem.variables.find(n => n.id === edge.source)!;
                const target = problem.variables.find(n => n.id === edge.target)!;

                // Highlight logic
                let stroke = '#e5e7eb';
                let width = 2;

                // Conflict
                if (conflictEdge && ((conflictEdge.source === edge.source && conflictEdge.target === edge.target) || (conflictEdge.source === edge.target && conflictEdge.target === edge.source))) {
                  stroke = 'red';
                  width = 4;
                }

                // AC-3 Current Arc
                if (algorithm === 'ac3' && currentArc) {
                  if (currentArc[0] === edge.source && currentArc[1] === edge.target) {
                    stroke = '#3b82f6'; // Blue for forward
                    width = 4;
                  } else if (currentArc[0] === edge.target && currentArc[1] === edge.source) {
                    stroke = '#3b82f6'; // Blue for reverse
                    width = 4;
                  }
                }

                return (
                  <line
                    key={idx}
                    x1={source.x} y1={source.y}
                    x2={target.x} y2={target.y}
                    stroke={stroke}
                    strokeWidth={width}
                    className="transition-colors duration-200"
                  />
                );
              })}

              {/* Nodes */}
              {problem.variables.map(node => renderNode(node))}
            </svg>

            {/* AC-3 Queue Visualization Overlay */}
            {algorithm === 'ac3' && (
              <div className="absolute top-4 right-4 bg-white/95 p-3 rounded shadow-lg border w-48 max-h-60 overflow-y-auto">
                <div className="font-bold text-sm mb-2 flex items-center gap-2 border-b pb-1">
                  <List className="w-4 h-4" /> Arc Queue
                </div>
                {ac3Queue.length === 0 && <div className="text-xs text-gray-400 italic">Empty</div>}
                {ac3Queue.map((arc, i) => (
                  <div key={i} className={`text-xs p-1 mb-1 rounded flex items-center gap-1 ${i === 0 ? 'bg-blue-100 font-bold text-blue-700' : 'bg-gray-50 text-gray-600'}`}>
                    {i === 0 && <ArrowRight className="w-3 h-3" />}
                    {arc[0]} → {arc[1]}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Explanation Panel */}
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 text-sm text-blue-900">
            <h4 className="font-bold flex items-center gap-2 mb-1"><Info className="w-4 h-4" /> Algorithm Details</h4>
            {algorithm === 'ac3' ? (
              <p>
                <strong>AC-3 (Arc Consistency Algorithm #3)</strong> iteratively checks arcs (constraints) in the queue.
                If the domain of a variable <em>Xi</em> is revised (values removed to satisfy constraint with <em>Xj</em>),
                all neighbors of <em>Xi</em> are added back to the queue to propagate the change.
              </p>
            ) : algorithm === 'forwardChecking' ? (
              <p>
                <strong>Forward Checking</strong> looks ahead one step. When a variable is assigned, it removes inconsistent values
                from the domains of its unassigned neighbors. If a domain becomes empty, it backtracks immediately.
              </p>
            ) : (
              <p>
                <strong>Backtracking Search</strong> is a depth-first search. It assigns values one by one.
                If a conflict arises, it backtracks to the previous variable and tries a different value.
              </p>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}