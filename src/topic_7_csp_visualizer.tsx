import React, { useState, useEffect, useRef } from 'react';
import { Play, StepForward, RotateCcw, Plus, Trash2, MousePointer2, Link as LinkIcon, Save, Upload, Map as MapIcon, GitBranch, Settings } from 'lucide-react';

// --- Types ---

type InteractionMode = 'move' | 'add-node' | 'add-edge' | 'delete';
type Algorithm = 'AC3' | 'Backtracking';

interface CSPNode {
  id: string; // Variable Name
  x: number;
  y: number;
  domain: number[]; // e.g., [1, 2, 3, 4, 5, 6]
  currentDomain: number[]; // For visualization of reduction
  assignedValue?: number | null; // For Backtracking
}

interface CSPEdge {
  source: string;
  target: string;
  constraint: string; // e.g., '<', '>', '=', '!=', '<=', '>='
  offset: number; // e.g., for X < Y + 2, offset is 2.
  factor: number; // Multiplier for Target logic (default 1)
  sourceMod?: number | null; // Modulo for Source logic
  exponent?: number; // Power for Target logic (default 1)
}

interface LogStep {
  step: number;
  queue?: string[]; // AC-3 (Legacy/Simple)
  queueDetails?: { active: string[], added: string[], removed: string }; // AC-3 (Detailed)
  currentArc?: string | null; // AC-3
  removedValues?: { var: string; val: number }[]; // AC-3 & FC

  assignedVar?: string; // Backtracking
  assignedVal?: number; // Backtracking
  backtrackFrom?: string; // Backtracking

  explanation: string;
}

// --- Default Problem 1: AC-3 Example ---
const NODES_AC3: CSPNode[] = [
  { id: 'X', x: 250, y: 100, domain: [1, 2, 3, 4, 5, 6], currentDomain: [1, 2, 3, 4, 5, 6] },
  { id: 'Y', x: 100, y: 300, domain: [1, 2, 3, 4, 5, 6], currentDomain: [1, 2, 3, 4, 5, 6] },
  { id: 'Z', x: 400, y: 300, domain: [1, 2, 3, 4, 5, 6], currentDomain: [1, 2, 3, 4, 5, 6] },
];

const EDGES_AC3: CSPEdge[] = [
  { source: 'X', target: 'Y', constraint: '<', offset: 0, factor: 1 }, // X < Y
  { source: 'Z', target: 'X', constraint: '<', offset: -2, factor: 1 }, // Z < X - 2
];

// --- Default Problem 2: Saudi Map Coloring ---
// Colors mapped to numbers: 1=Red, 2=Green, 3=Blue, 4=Yellow
const COLORS = [1, 2, 3, 4];
const COLOR_NAMES: Record<number, string> = { 1: 'Red', 2: 'Green', 3: 'Blue', 4: 'Yellow' };
const COLOR_HEX: Record<number, string> = { 1: '#ef4444', 2: '#22c55e', 3: '#3b82f6', 4: '#facc15' };

const NODES_MAP: CSPNode[] = [
  { id: 'Tabuk', x: 100, y: 100, domain: [...COLORS], currentDomain: [...COLORS] },  // V1
  { id: 'Hail', x: 300, y: 100, domain: [...COLORS], currentDomain: [...COLORS] },   // V2
  { id: 'Qassim', x: 400, y: 250, domain: [...COLORS], currentDomain: [...COLORS] }, // V3
  { id: 'Riyadh', x: 300, y: 400, domain: [...COLORS], currentDomain: [...COLORS] }, // V4
  { id: 'Makkah', x: 100, y: 400, domain: [...COLORS], currentDomain: [...COLORS] }, // V5
  { id: 'Medina', x: 200, y: 250, domain: [...COLORS], currentDomain: [...COLORS] }, // V6
];

const EDGES_MAP: CSPEdge[] = [
  // Outer Path: V1-V2-V3-V4-V5
  { source: 'Tabuk', target: 'Hail', constraint: '!=', offset: 0, factor: 1 },
  { source: 'Hail', target: 'Qassim', constraint: '!=', offset: 0, factor: 1 },
  { source: 'Qassim', target: 'Riyadh', constraint: '!=', offset: 0, factor: 1 },
  { source: 'Riyadh', target: 'Makkah', constraint: '!=', offset: 0, factor: 1 },

  // Center Star: V6 connected to all
  { source: 'Medina', target: 'Tabuk', constraint: '!=', offset: 0, factor: 1 },
  { source: 'Medina', target: 'Hail', constraint: '!=', offset: 0, factor: 1 },
  { source: 'Medina', target: 'Qassim', constraint: '!=', offset: 0, factor: 1 },
  { source: 'Medina', target: 'Riyadh', constraint: '!=', offset: 0, factor: 1 },
  { source: 'Medina', target: 'Makkah', constraint: '!=', offset: 0, factor: 1 },
];

// --- Helper Functions ---

const parseDomain = (str: string): number[] => {
  return str.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
};

// Check: Lhs (op) Rhs
// where Lhs = sourceVal (optionally % sourceMod)
//       Rhs = targetVal * factor + offset
const checkConstraint = (valSource: number, valTarget: number, edge: CSPEdge): boolean => {
  let lhs = valSource;
  if (edge.sourceMod !== undefined && edge.sourceMod !== null && !isNaN(edge.sourceMod) && edge.sourceMod !== 0) {
    lhs = lhs % edge.sourceMod;
  }

  const targetValWithExp = edge.exponent !== undefined ? Math.pow(valTarget, edge.exponent) : valTarget;
  const rhs = (targetValWithExp * (edge.factor ?? 1)) + edge.offset;

  // Floating point tolerance check for equality
  const epsilon = 1e-9;

  switch (edge.constraint) {
    case '<': return lhs < rhs - epsilon; // strict less
    case '>': return lhs > rhs + epsilon;
    case '<=': return lhs <= rhs + epsilon;
    case '>=': return lhs >= rhs - epsilon;
    case '=': return Math.abs(lhs - rhs) < epsilon;
    case '!=': return Math.abs(lhs - rhs) >= epsilon;
    default: return true;
  }
};

// --- Algorithms ---

// 1. AC-3
const runAC3 = (nodes: CSPNode[], edges: CSPEdge[]): { steps: LogStep[], finalNodes: CSPNode[] } => {
  let queue: { source: string; target: string }[] = [];
  const steps: LogStep[] = [];

  const currentNodes = nodes.map(n => ({ ...n, currentDomain: [...n.domain] }));
  const getNode = (id: string) => currentNodes.find(n => n.id === id)!;

  edges.forEach(e => {
    queue.push({ source: e.source, target: e.target });
    queue.push({ source: e.target, target: e.source });
  });

  let stepCount = 0;
  steps.push({
    step: 0,
    queue: queue.map(q => `(${q.source}, ${q.target})`),
    explanation: 'Initial Queue populated with all arcs.',
  });

  while (queue.length > 0) {
    stepCount++;
    const { source, target } = queue.shift()!;
    const sourceNode = getNode(source);
    const targetNode = getNode(target);

    // Find the edge connecting these two nodes
    const edge = edges.find(e =>
      (e.source === source && e.target === target) ||
      (e.source === target && e.target === source)
    );

    const removed: number[] = [];
    let addedNeighbors: string[] = [];

    // Capture state before changes
    const queueBefore = queue.map(q => `(${q.source},${q.target})`);

    if (edge) {
      const newDomain = sourceNode.currentDomain.filter(x => {
        // We need to find if there exists ANY value y in targetNode's domain that satisfies the constraint
        const exists = targetNode.currentDomain.some(y => {
          if (edge.source === source) {
            // Standard direction: Check X against Y
            return checkConstraint(x, y, edge);
          } else {
            // Reverse direction: Check Y against X
            // The edge constraint is defined as checkConstraint(sourceVal, targetVal)
            // So we pass (y, x)
            return checkConstraint(y, x, edge);
          }
        });
        if (!exists) removed.push(x);
        return exists;
      });

      if (removed.length > 0) {
        sourceNode.currentDomain = newDomain;
        edges.forEach(e => { // Add neighbors to queue
          const neighbor = (e.source === source) ? e.target : (e.target === source ? e.source : null);
          if (neighbor && neighbor !== target) {
            const arc = { source: neighbor, target: source };
            if (!queue.some(q => q.source === arc.source && q.target === arc.target)) {
              queue.push(arc);
              addedNeighbors.push(`(${arc.source},${arc.target})`);
            }
          }
        });
      }
    }

    steps.push({
      step: stepCount,
      queue: queue.map(q => `(${q.source}, ${q.target})`),
      queueDetails: {
        active: queueBefore, // Queue just before processing this arc
        removed: `(${source},${target})`, // The arc being processed/removed
        added: addedNeighbors
      },
      currentArc: `(${source}, ${target})`,
      removedValues: removed.map(v => ({ var: source, val: v })),
      explanation: removed.length > 0
        ? `Removed {${removed.join(',')}} from ${source} to satisfy constraint. Added neighbors.`
        : `Consistent. No values removed from ${source}.`
    });
  }

  return { steps, finalNodes: currentNodes };
};

// 2. Backtracking (with FC and MRV)
const runBacktracking = (
  nodes: CSPNode[],
  edges: CSPEdge[],
  enableFC: boolean,
  enableMRV: boolean
): { steps: LogStep[], finalNodes: CSPNode[] } => {
  const steps: LogStep[] = [];
  let stepCount = 0;

  // Working copy of assignments and domains
  const assignments: Record<string, number> = {};
  // Map of variable ID to current domain (for FC)
  const domains: Record<string, number[]> = {};
  nodes.forEach(n => domains[n.id] = [...n.domain]);

  const isConsistent = (varId: string, val: number, currentAssignments: Record<string, number>) => {
    for (const edge of edges) {
      if (edge.source === varId) {
        const neighborId = edge.target;
        if (currentAssignments[neighborId] !== undefined) {
          // Checking var (val) against neighbor (assigned)
          // Edge: var -> neighbor
          if (!checkConstraint(val, currentAssignments[neighborId], edge)) return false;
        }
      } else if (edge.target === varId) {
        const neighborId = edge.source;
        if (currentAssignments[neighborId] !== undefined) {
          // Checking var (val) against neighbor (assigned)
          // Edge: neighbor -> var
          // Constraint is checkConstraint(source, target) -> checkConstraint(neighbor, var)
          if (!checkConstraint(currentAssignments[neighborId], val, edge)) return false;
        }
      }
    }
    return true;
  };

  // Forward Checking Helper
  const forwardCheck = (varId: string, val: number, currentDomains: Record<string, number[]>) => {
    const removed: { var: string; val: number }[] = [];
    let failure = false;

    for (const edge of edges) {
      let neighborId: string | null = null;
      let isSource = false; // Is varId the source of this edge?

      if (edge.source === varId) {
        neighborId = edge.target;
        isSource = true;
      } else if (edge.target === varId) {
        neighborId = edge.source;
        isSource = false;
      }

      // We only care about unassigned neighbors
      if (neighborId && assignments[neighborId] === undefined) {
        const neighborDomain = currentDomains[neighborId];

        // Filter neighbor domain
        // If isSource, edge is var -> neighbor. we check checkConstraint(val, nVal, edge)
        // If !isSource, edge is neighbor -> var. we check checkConstraint(nVal, val, edge)

        const newDomain = neighborDomain.filter(nVal => {
          if (isSource) {
            return checkConstraint(val, nVal, edge);
          } else {
            return checkConstraint(nVal, val, edge);
          }
        });

        // Identify removed values
        const removedVals = neighborDomain.filter(nVal => !newDomain.includes(nVal));
        removedVals.forEach(r => removed.push({ var: neighborId!, val: r }));

        currentDomains[neighborId] = newDomain;

        if (newDomain.length === 0) failure = true;
      }
    }
    return { removed, failure };
  };

  const backtrack = (): boolean => {
    if (Object.keys(assignments).length === nodes.length) {
      return true; // Solution found
    }

    // Select unassigned variable
    let varId: string | undefined;
    const unassigned = nodes.filter(n => assignments[n.id] === undefined);

    if (enableMRV) {
      // MRV: Choose variable with smallest remaining domain
      unassigned.sort((a, b) => domains[a.id].length - domains[b.id].length);
      varId = unassigned[0]?.id;
    } else {
      // Default: First unassigned
      varId = unassigned[0]?.id;
    }

    if (!varId) return true;

    const currentDomain = [...domains[varId]]; // Copy domain to iterate

    for (const val of currentDomain) {
      stepCount++;

      // Check consistency with *assignments* (Standard Backtracking check)
      if (isConsistent(varId, val, assignments)) {
        assignments[varId] = val;

        let fcRemoved: { var: string; val: number }[] = [];
        let fcFailed = false;

        // Forward Checking
        if (enableFC) {
          const fcResult = forwardCheck(varId, val, domains);
          fcRemoved = fcResult.removed;
          fcFailed = fcResult.failure;
        }

        steps.push({
          step: stepCount,
          assignedVar: varId,
          assignedVal: val,
          removedValues: fcRemoved.length > 0 ? fcRemoved : undefined,
          explanation: `Assigned ${val} to ${varId}. ${enableFC ? (fcFailed ? 'FC detected empty domain!' : `FC removed ${fcRemoved.length} values.`) : 'Consistent.'}`
        });

        if (!fcFailed) {
          if (backtrack()) return true;
        }

        // Backtrack
        delete assignments[varId];

        // Restore domains (Undo FC)
        if (enableFC) {
          fcRemoved.forEach(r => {
            domains[r.var].push(r.val);
            domains[r.var].sort((a, b) => a - b); // Keep sorted
          });
        }

        steps.push({
          step: stepCount,
          backtrackFrom: varId,
          explanation: `Backtracking from ${varId} (Value ${val} led to failure)`
        });
      } else {
        steps.push({
          step: stepCount,
          explanation: `Value ${val} for ${varId} conflicts with existing assignments.`
        });
      }
    }
    return false;
  };

  backtrack();

  const finalNodes = nodes.map(n => ({ ...n, assignedValue: assignments[n.id] }));
  return { steps, finalNodes };
};


// --- Main Component ---

const parseConstraintInput = (input: string): Partial<CSPEdge> | null => {
  const lower = input.toLowerCase().replace(/\s+/g, ''); // remove spaces

  // Macros
  if (lower.includes('even')) return { sourceMod: 2, constraint: '=', factor: 0, offset: 0 };
  if (lower.includes('odd')) return { sourceMod: 2, constraint: '!=', factor: 0, offset: 0 };

  // Parse Operator
  const ops = ['<=', '>=', '!=', '=', '<', '>'];
  const op = ops.find(o => lower.includes(o));
  if (!op) return null;

  const parts = lower.split(op!);
  const lhs = parts[0];
  let rhs = parts[1];

  let sourceMod: number | undefined = undefined;
  if (lhs.includes('%')) {
    const modPart = lhs.split('%')[1];
    const parsed = parseFloat(modPart);
    if (!isNaN(parsed)) sourceMod = parsed;
  }

  let factor = 0;
  let offset = 0;
  let exponent = 1;

  // Sqrt Macro
  if (rhs.includes('sqrt')) {
    exponent = 0.5;
    rhs = rhs.replace('sqrt', '').replace(/\(/g, '').replace(/\)/g, '');
  }

  // Heuristic for RHS: Check for variable presence (any letter) implies dependency on Target
  const hasTarget = /[a-z]/.test(rhs);

  if (hasTarget) {
    // try to split by target char
    const splitRhs = rhs.split(/[a-z]/);
    const before = splitRhs[0];
    let after = splitRhs[1];

    if (before === '' || before === '*') factor = 1;
    else if (before === '-') factor = -1;
    else {
      const f = parseFloat(before);
      factor = isNaN(f) ? 1 : f;
    }

    if (after) {
      if (after.startsWith('^')) {
        // Parse exponent
        const expParts = after.substring(1).split(/[+\-]/); // Split by next sign to get just the exponent
        const expVal = parseFloat(expParts[0]);
        if (!isNaN(expVal)) exponent = expVal;

        // Remove exponent part from 'after' to parse offset
        // after was ^2+1 -> offset starts at +
        // const powerStrLength = 1 + expParts[0].length;
        // after = after.substring(powerStrLength);
        // Simplified: just re-parse offset from remainder?
        // Safer: split RHS by ^ and re-evaluate logic? 
        // Let's stick to simple: if ^ exists, take everything after ^ as power? 
        // No, ^2+1. 

        // Re-slice check for offset
        const signIndex = after.substring(1).search(/[+\-]/);
        if (signIndex !== -1) {
          // There is an offset
          const expStr = after.substring(1, 1 + signIndex);
          const expParsed = parseFloat(expStr);
          if (!isNaN(expParsed)) exponent = expParsed; // Overwrite if successful

          const offsetStr = after.substring(1 + signIndex);
          const o = parseFloat(offsetStr);
          offset = isNaN(o) ? 0 : o;
        } else {
          // Just power
          const expParsed = parseFloat(after.substring(1));
          if (!isNaN(expParsed)) exponent = expParsed;
        }
      } else {
        const o = parseFloat(after);
        offset = isNaN(o) ? 0 : o;
      }
    }
  } else {
    // Pure constant
    const o = parseFloat(rhs);
    offset = isNaN(o) ? 0 : o;
  }

  return { constraint: op, sourceMod, factor, offset, exponent };
};

export default function CSPVisualizer() {
  const [nodes, setNodes] = useState<CSPNode[]>([]);
  const [edges, setEdges] = useState<CSPEdge[]>([]);
  const [algo, setAlgo] = useState<Algorithm>('AC3');
  const [problemType, setProblemType] = useState<'numerical' | 'coloring'>('numerical');

  // Backtracking Options
  const [enableFC, setEnableFC] = useState(false);
  const [enableMRV, setEnableMRV] = useState(false);

  // Solver State
  const [steps, setSteps] = useState<LogStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isSolving, setIsSolving] = useState(false);

  // Interaction State
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('move');
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const [connectStart, setConnectStart] = useState<string | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Selection
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeIndex, setSelectedEdgeIndex] = useState<number | null>(null);
  const [editDomainVal, setEditDomainVal] = useState('');
  const [editOffsetVal, setEditOffsetVal] = useState('');

  // Sync edit value when selection changes
  useEffect(() => {
    if (selectedNodeId) {
      const node = nodes.find(n => n.id === selectedNodeId);
      if (node) setEditDomainVal(node.domain.join(', '));
    }
    if (selectedEdgeIndex !== null) {
      const edge = edges[selectedEdgeIndex];
      if (edge) setEditOffsetVal(edge.offset.toString());
    }
  }, [selectedNodeId, selectedEdgeIndex]);

  // --- Handlers ---

  const loadPreset = (type: 'AC3' | 'Map') => {
    if (type === 'AC3') {
      setNodes(NODES_AC3);
      setEdges(EDGES_AC3);
      setAlgo('AC3');
      setProblemType('numerical');
    } else {
      setNodes(NODES_MAP);
      setEdges(EDGES_MAP);
      setAlgo('Backtracking');
      setProblemType('coloring');
      setEnableFC(true); // Usually good for map coloring
      setEnableMRV(true);
    }
    setSteps([]);
    setCurrentStepIndex(0);
    setIsSolving(false);
    setSelectedNodeId(null);
    setSelectedEdgeIndex(null);
  };

  const handleSolve = () => {
    let result;
    if (algo === 'AC3') {
      result = runAC3(nodes, edges);
    } else {
      result = runBacktracking(nodes, edges, enableFC, enableMRV);
    }
    setSteps(result.steps);
    setCurrentStepIndex(0);
    setIsSolving(true);
  };

  const handleReset = () => {
    setSteps([]);
    setCurrentStepIndex(0);
    setIsSolving(false);
  };

  const handleStepForward = () => {
    if (currentStepIndex < steps.length - 1) {
      setCurrentStepIndex(prev => prev + 1);
    }
  };

  const handleStepBack = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(prev => prev - 1);
    }
  };

  // --- Graph Interaction Handlers ---

  const getNextNodeId = () => {
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const used = new Set(nodes.map(n => n.id));
    for (const char of letters) {
      if (!used.has(char)) return char;
    }
    return `V${nodes.length + 1}`;
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (interactionMode === 'add-node') {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const id = getNextNodeId();
      setNodes(prev => [...prev, { id, x, y, domain: [1, 2, 3, 4], currentDomain: [1, 2, 3, 4] }]);
      setSelectedNodeId(id);
      setSelectedEdgeIndex(null);
    } else {
      setSelectedNodeId(null);
      setSelectedEdgeIndex(null);
    }
  };

  const handleNodeMouseDown = (e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    if (interactionMode === 'delete') {
      setNodes(prev => prev.filter(n => n.id !== nodeId));
      setEdges(prev => prev.filter(edge => edge.source !== nodeId && edge.target !== nodeId));
      setSelectedNodeId(null);
      return;
    }
    if (interactionMode === 'add-edge') {
      setConnectStart(nodeId);
      return;
    }
    if (interactionMode === 'move') {
      setDraggedNode(nodeId);
      setSelectedNodeId(nodeId);
      setSelectedEdgeIndex(null);
    }
  };

  const handleNodeMouseUp = (e: React.MouseEvent, targetId: string) => {
    e.stopPropagation();
    if (interactionMode === 'add-edge' && connectStart && connectStart !== targetId) {
      const exists = edges.some(e => (e.source === connectStart && e.target === targetId) || (e.source === targetId && e.target === connectStart));
      if (!exists) {
        setEdges(prev => [...prev, { source: connectStart, target: targetId, constraint: '!=', offset: 0, factor: 1 }]);
      }
      setConnectStart(null);
    }
    setDraggedNode(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });
    if (interactionMode === 'move' && draggedNode) {
      setNodes(prev => prev.map(n => n.id === draggedNode ? { ...n, x, y } : n));
    }
  };

  // --- Property Editing ---

  const handleNodeDomainChange = (val: string) => {
    setEditDomainVal(val); // Update local input immediately
    if (!selectedNodeId) return;
    const newDomain = parseDomain(val);
    setNodes(prev => prev.map(n => n.id === selectedNodeId ? { ...n, domain: newDomain, currentDomain: newDomain } : n));
    handleReset(); // Clear solution when editing
  };

  const handleEdgeConstraintChange = (val: string) => {
    if (selectedEdgeIndex === null) return;
    setEdges(prev => prev.map((e, i) => i === selectedEdgeIndex ? { ...e, constraint: val } : e));
    handleReset(); // Clear solution
  };

  const handleEdgeOffsetChange = (val: string) => {
    if (selectedEdgeIndex === null) return;
    setEditOffsetVal(val); // Update local input
    const num = parseInt(val);
    if (!isNaN(num)) {
      setEdges(prev => prev.map((e, i) => i === selectedEdgeIndex ? { ...e, offset: num } : e));
      handleReset(); // Clear solution
    }
  };

  const selectedNode = selectedNodeId ? nodes.find(n => n.id === selectedNodeId) : null;
  const selectedEdge = selectedEdgeIndex !== null ? edges[selectedEdgeIndex] : null;

  // --- Rendering Helpers ---

  // Get current state based on step
  const getVisualStateAtStep = (nodeId: string) => {
    if (!isSolving || steps.length === 0) {
      return {
        domain: nodes.find(n => n.id === nodeId)?.domain || [],
        assignedValue: null
      };
    }

    // Replay steps
    const initialDomain = nodes.find(n => n.id === nodeId)?.domain || [];
    const activeAssignments: Record<string, number> = {};
    const activeRemovals: { var: string; val: number, causeStep: number }[] = [];

    for (let i = 0; i <= currentStepIndex; i++) {
      const s = steps[i];

      if (s.assignedVar) {
        activeAssignments[s.assignedVar] = s.assignedVal!;
      }

      if (s.removedValues) {
        s.removedValues.forEach(r => {
          activeRemovals.push({ var: r.var, val: r.val, causeStep: i });
        });
      }

      if (s.backtrackFrom) {
        // Remove assignment
        delete activeAssignments[s.backtrackFrom];

        // Find the step index of the assignment we are undoing.
        let assignStepIndex = -1;
        for (let j = i - 1; j >= 0; j--) {
          if (steps[j].assignedVar === s.backtrackFrom && activeAssignments[s.backtrackFrom] === steps[j].assignedVal) {
            assignStepIndex = j;
            break;
          }
        }

        if (assignStepIndex !== -1) {
          // Remove removals caused by this step
          const toKeep = activeRemovals.filter(r => r.causeStep !== assignStepIndex);
          activeRemovals.length = 0;
          activeRemovals.push(...toKeep);
        }
      }
    }

    const currentRemovedSet = new Set<number>();
    activeRemovals.forEach(r => {
      if (r.var === nodeId) currentRemovedSet.add(r.val);
    });

    return {
      domain: initialDomain.filter(v => !currentRemovedSet.has(v)),
      assignedValue: activeAssignments[nodeId] || null
    };
  };

  const currentStep = steps[currentStepIndex];

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans text-gray-800">
      {/* Header */}
      <div className="bg-white p-4 shadow-sm border-b flex flex-col md:flex-row justify-between items-center gap-4 z-10">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
            CSP Studio
          </h1>
          <div className="flex bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => { setAlgo('AC3'); handleReset(); }}
              className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${algo === 'AC3' ? 'bg-white text-green-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              AC-3
            </button>
            <button
              onClick={() => { setAlgo('Backtracking'); handleReset(); }}
              className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${algo === 'Backtracking' ? 'bg-white text-green-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
            >
              Backtracking
            </button>
          </div>
        </div>
        <div className="flex gap-2 items-center">
          <button onClick={() => loadPreset('AC3')} className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-600">
            Load AC-3 Example
          </button>
          <button onClick={() => loadPreset('Map')} className="px-3 py-1 text-xs bg-blue-100 hover:bg-blue-200 rounded text-blue-700 flex items-center gap-1">
            <MapIcon size={12} /> Load Map Coloring
          </button>
          <div className="w-px h-6 bg-gray-300 mx-2" />
          <button onClick={handleReset} className="flex items-center gap-1 px-3 py-2 text-gray-600 hover:bg-gray-100 rounded-md">
            <RotateCcw size={16} /> Reset
          </button>
          {!isSolving ? (
            <button onClick={handleSolve} className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md shadow hover:bg-green-700">
              <Play size={16} /> Run {algo}
            </button>
          ) : (
            <div className="flex gap-2">
              <button onClick={handleStepBack} disabled={currentStepIndex === 0} className="p-2 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50">
                &lt; Prev
              </button>
              <span className="flex items-center text-sm font-mono">
                Step {currentStepIndex} / {steps.length - 1}
              </span>
              <button onClick={handleStepForward} disabled={currentStepIndex === steps.length - 1} className="p-2 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50">
                Next &gt;
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: Editor & Graph */}
        <div className="w-2/3 border-r bg-white flex flex-col select-none">
          {/* Toolbar */}
          <div className="flex-1 relative overflow-hidden">
            <div className="absolute top-4 left-4 flex flex-col bg-white shadow-md border rounded-lg z-20">
              <button onClick={() => setInteractionMode('move')} className={`p-2 rounded-t-lg hover:bg-gray-50 ${interactionMode === 'move' ? 'bg-blue-50 text-blue-600' : 'text-gray-600'}`}>
                <MousePointer2 size={20} />
              </button>
              <button onClick={() => setInteractionMode('add-node')} className={`p-2 hover:bg-gray-50 ${interactionMode === 'add-node' ? 'bg-blue-50 text-blue-600' : 'text-gray-600'}`}>
                <Plus size={20} />
              </button>
              <button onClick={() => setInteractionMode('add-edge')} className={`p-2 hover:bg-gray-50 ${interactionMode === 'add-edge' ? 'bg-blue-50 text-blue-600' : 'text-gray-600'}`}>
                <LinkIcon size={20} />
              </button>
              <button onClick={() => setInteractionMode('delete')} className={`p-2 rounded-b-lg hover:bg-red-50 ${interactionMode === 'delete' ? 'bg-red-50 text-red-600' : 'text-gray-600'}`}>
                <Trash2 size={20} />
              </button>
            </div>

            {/* Backtracking Options Overlay */}
            {algo === 'Backtracking' && !isSolving && (
              <div className="absolute top-4 right-4 bg-white shadow-md border rounded-lg p-3 z-20 space-y-2">
                <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Heuristics</div>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" checked={enableFC} onChange={e => setEnableFC(e.target.checked)} className="rounded text-blue-600" />
                  Forward Checking (FC)
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" checked={enableMRV} onChange={e => setEnableMRV(e.target.checked)} className="rounded text-blue-600" />
                  Min Remaining Values (MRV)
                </label>
              </div>
            )}

            {/* Quick Instructions */}
            <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur border border-slate-200 rounded-lg p-3 shadow-md border-l-4 border-l-blue-500 max-w-xs select-none pointer-events-none z-10">
              <div className="font-bold text-xs text-gray-700 mb-1">Quick Guide</div>
              <ul className="text-[10px] text-gray-600 space-y-1 list-disc list-inside">
                <li><strong>Double-click</strong> Domain/Constraint to edit.</li>
                <li><strong>Toolbar</strong> (top-left) to Add/Remove Nodes.</li>
                <li><strong>Drag</strong> nodes to rearrange.</li>
              </ul>
            </div>

            <svg
              className="flex-1 w-full h-full"
              onMouseMove={handleMouseMove}
              onClick={handleCanvasClick}
              onMouseUp={() => { setDraggedNode(null); setConnectStart(null); }}
            >
              <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="22" refY="3" orient="auto" markerUnits="strokeWidth">
                  <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
                </marker>
              </defs>

              {/* Connecting Line */}
              {interactionMode === 'add-edge' && connectStart && (
                <line x1={nodes.find(n => n.id === connectStart)?.x} y1={nodes.find(n => n.id === connectStart)?.y} x2={mousePos.x} y2={mousePos.y} stroke="#3b82f6" strokeWidth="2" strokeDasharray="5,5" pointerEvents="none" />
              )}

              {/* Edges */}
              {edges.map((edge, i) => {
                const s = nodes.find(n => n.id === edge.source);
                const t = nodes.find(n => n.id === edge.target);
                if (!s || !t) return null;
                const isSelected = selectedEdgeIndex === i;

                // Highlight if this edge is being checked in AC-3
                const isCurrentArc = algo === 'AC3' && currentStep?.currentArc === `(${edge.source}, ${edge.target})`;
                const isReverseArc = algo === 'AC3' && currentStep?.currentArc === `(${edge.target}, ${edge.source})`;

                let stroke = '#94a3b8';
                if (isSelected) stroke = '#3b82f6';
                if (isCurrentArc || isReverseArc) stroke = '#f59e0b'; // Orange for active

                return (
                  <g key={i}
                    onClick={(e) => { e.stopPropagation(); setSelectedEdgeIndex(i); setSelectedNodeId(null); }}
                    onDoubleClick={(e) => { e.stopPropagation(); setSelectedEdgeIndex(i); setSelectedNodeId(null); }}
                    className="cursor-pointer">
                    <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke={stroke} strokeWidth={isSelected || isCurrentArc ? 3 : 2} markerEnd="url(#arrow)" />

                    {/* Constraint Box */}
                    <rect
                      x={(s.x + t.x) / 2 - 25}
                      y={(s.y + t.y) / 2 - 12}
                      width="50"
                      height="24"
                      fill="white"
                      stroke={stroke}
                      strokeWidth="1"
                      rx="4"
                    />
                    <text
                      x={(s.x + t.x) / 2}
                      y={(s.y + t.y) / 2 + 5}
                      textAnchor="middle"
                      fontSize="10"
                      fontWeight="bold"
                      fill={stroke}
                    >
                      {(() => {
                        // Smart Label
                        if (edge.sourceMod === 2 && edge.factor === 0 && edge.offset === 0) {
                          if (edge.constraint === '=') return 'Even';
                          if (edge.constraint === '!=') return 'Odd';
                        }

                        let label = '';
                        if (edge.sourceMod) label += `%${edge.sourceMod} `;
                        label += `${edge.constraint} `;

                        if (edge.factor !== 0) {
                          if (edge.factor !== 1) label += `${edge.factor}`;
                          label += edge.target;
                          if (edge.exponent !== undefined && edge.exponent !== 1) {
                            label += `^${edge.exponent}`;
                          }
                        }

                        if (edge.offset !== 0) {
                          label += edge.offset > 0 ? ` + ${edge.offset}` : ` - ${Math.abs(edge.offset)}`;
                        }

                        if (edge.factor === 0 && edge.offset === 0 && !edge.sourceMod) return edge.constraint; // Just constraint char if nothing else? Unlikely case.

                        return label;
                      })()}
                    </text>
                    {/* Invisible Hit Area for Double Click */}
                    <rect
                      x={(s.x + t.x) / 2 - 25}
                      y={(s.y + t.y) / 2 - 12}
                      width="50"
                      height="24"
                      fill="transparent"
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        // Prompt
                        const current = `${edge.constraint} ${edge.offset}`;
                        const input = window.prompt(
                          "Enter Constraint Equation.\nExamples:\n'X < Y'\n'X > 0.5Y + 1'\n'X = Y^2'\n'X = sqrt(Y)'\n'X even'",
                          `${edge.constraint} ${edge.factor !== 1 ? `${edge.factor}${edge.target}` : edge.target}${edge.exponent && edge.exponent !== 1 ? `^${edge.exponent}` : ''} ${edge.offset !== 0 ? `+ ${edge.offset}` : ''}`
                        );

                        if (input !== null) {
                          const parsed = parseConstraintInput(input);
                          if (parsed && parsed.constraint) {
                            setEdges(prev => prev.map((ed, idx) => idx === i ? {
                              ...ed,
                              constraint: parsed.constraint!,
                              offset: parsed.offset ?? 0,
                              factor: parsed.factor ?? 0,
                              sourceMod: parsed.sourceMod,
                              exponent: parsed.exponent ?? 1
                            } : ed));
                            handleReset();
                          }
                        }
                      }}
                    />
                  </g>
                );
              })}

              {/* Nodes */}
              {nodes.map(node => {
                const isSelected = selectedNodeId === node.id;
                const { domain, assignedValue } = getVisualStateAtStep(node.id);

                // Check if values removed in current step (AC-3 or FC)
                const removedInStep = (currentStep?.removedValues?.filter(r => r.var === node.id).map(r => r.val) || []);

                // Backtracking Visuals
                let fillColor = 'white';
                if (assignedValue) {
                  if (problemType === 'coloring') {
                    fillColor = COLOR_HEX[assignedValue] || '#dcfce7';
                  } else {
                    fillColor = '#dcfce7'; // Simple green for assigned numerical
                  }
                }

                return (
                  <g
                    key={node.id}
                    onMouseDown={(e) => handleNodeMouseDown(e, node.id)}
                    onMouseUp={(e) => handleNodeMouseUp(e, node.id)}
                    onDoubleClick={(e) => { e.stopPropagation(); setSelectedNodeId(node.id); setSelectedEdgeIndex(null); }}
                    className="cursor-pointer"
                  >
                    <circle cx={node.x} cy={node.y} r="25" fill={fillColor} stroke={isSelected ? '#3b82f6' : '#475569'} strokeWidth={isSelected ? 3 : 2} />
                    <text x={node.x} y={node.y + 5} textAnchor="middle" fontWeight="bold" fontSize="14" fill={assignedValue ? 'white' : '#1e293b'} className="pointer-events-none" style={{ textShadow: assignedValue ? '0px 1px 2px rgba(0,0,0,0.5)' : 'none' }}>{node.id}</text>

                    {/* Domain Display (Always visible) */}
                    {(algo === 'AC3' || algo === 'Backtracking') && (
                      <foreignObject
                        x={node.x - 60}
                        y={node.y - 85}
                        width="120"
                        height="100"
                        onDoubleClick={(e) => {
                          e.stopPropagation();
                          const input = window.prompt("Edit Domain (comma separated):", node.domain.join(", "));
                          if (input !== null) {
                            // Re-use logic
                            const newDomain = parseDomain(input);
                            setNodes(prev => prev.map(n => n.id === node.id ? { ...n, domain: newDomain, currentDomain: newDomain } : n));
                            handleReset();
                          }
                        }}
                      >
                        <div className="flex flex-wrap justify-center gap-1">
                          {domain.map(val => (
                            <span
                              key={val}
                              className={`text-[10px] flex items-center justify-center font-bold rounded shadow-sm border border-gray-200 ${problemType === 'coloring' ? 'w-5 h-5' : 'px-1 py-0.5 bg-white'}`}
                              style={problemType === 'coloring' ? {
                                backgroundColor: COLOR_HEX[val] ? COLOR_HEX[val] : '#f3f4f6',
                                color: COLOR_HEX[val] ? 'white' : 'black'
                              } : {}}
                            >
                              {problemType === 'coloring' && COLOR_NAMES[val] ? COLOR_NAMES[val][0] : val}
                            </span>
                          ))}
                          {removedInStep.map(val => (
                            <span
                              key={val}
                              className={`text-[10px] flex items-center justify-center font-bold rounded opacity-40 border border-gray-300 bg-gray-100 text-gray-400 line-through ${problemType === 'coloring' ? 'w-5 h-5' : 'px-1 py-0.5'}`}
                            >
                              {problemType === 'coloring' && COLOR_NAMES[val] ? COLOR_NAMES[val][0] : val}
                            </span>
                          ))}
                        </div>
                      </foreignObject>
                    )}

                    {/* Assigned Value Label (Backtracking) */}
                    {algo === 'Backtracking' && assignedValue && (
                      <text
                        x={node.x}
                        y={node.y - 45}
                        textAnchor="middle"
                        fontSize="14"
                        fontWeight="bold"
                        fill={problemType === 'coloring' ? (COLOR_HEX[assignedValue] || '#166534') : '#166534'}
                        style={problemType === 'coloring' ? { textShadow: '0px 1px 2px rgba(255,255,255,0.8)' } : {}}
                      >
                        {problemType === 'coloring' && COLOR_NAMES[assignedValue] ? COLOR_NAMES[assignedValue][0] : assignedValue}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Property Editor */}
          <div className="border-t bg-gray-50 p-4 text-sm space-y-4 border-b-2 border-gray-100">
            <div className="font-bold text-gray-700 flex items-center justify-between">
              <span>Properties</span>
              {(selectedNode || selectedEdge) && <span className="text-xs font-normal text-gray-400">Editing will reset progress</span>}
            </div>
            {selectedNode && (
              <div className="space-y-2">
                <div>
                  <label className="block text-gray-500 mb-1">Variable Name</label>
                  <input className="border rounded px-2 py-1 w-full" value={selectedNode.id} readOnly />
                </div>
                <div>
                  <label className="block text-gray-500 mb-1">Domain (comma separated)</label>
                  <input
                    className="border rounded px-2 py-1 w-full"
                    value={editDomainVal}
                    onChange={(e) => handleNodeDomainChange(e.target.value)}
                  />
                </div>
              </div>
            )}
            {selectedEdge && (
              <div className="space-y-2">
                <div className="flex gap-2">
                  <div className="flex-1">
                    <label className="block text-gray-500 mb-1">Constraint</label>
                    <select
                      className="border rounded px-2 py-1 w-full"
                      value={selectedEdge.constraint}
                      onChange={(e) => handleEdgeConstraintChange(e.target.value)}
                    >
                      <option value="<">&lt;</option>
                      <option value=">">&gt;</option>
                      <option value="=">=</option>
                      <option value="!=">!=</option>
                      <option value="<=">&lt;=</option>
                      <option value=">=">&gt;=</option>
                    </select>
                  </div>
                  <div className="flex-1">
                    <label className="block text-gray-500 mb-1">Offset</label>
                    <input
                      type="number"
                      step="0.1"
                      className="border rounded px-2 py-1 w-full"
                      value={editOffsetVal}
                      onChange={(e) => handleEdgeOffsetChange(e.target.value)}
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <div className="flex-1">
                    <label className="block text-gray-500 mb-1">Factor (Mult)</label>
                    <input
                      type="number"
                      step="0.1"
                      className="border rounded px-2 py-1 w-full"
                      value={selectedEdge.factor ?? 1}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                          setEdges(prev => prev.map((ed, i) => i === selectedEdgeIndex ? { ...ed, factor: val } : ed));
                          handleReset();
                        }
                      }}
                    />
                  </div>
                  <div className="flex-1">
                    <label className="block text-gray-500 mb-1">Source Mod</label>
                    <input
                      type="number"
                      placeholder="None"
                      className="border rounded px-2 py-1 w-full"
                      value={selectedEdge.sourceMod ?? ''}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        setEdges(prev => prev.map((ed, i) => i === selectedEdgeIndex ? { ...ed, sourceMod: !isNaN(val) ? val : undefined } : ed));
                        handleReset();
                      }}
                    />
                  </div>
                </div>
                <p className="text-gray-400 italic text-xs mt-2">
                  Format: {`${selectedEdge.source} ${selectedEdge.sourceMod ? `% ${selectedEdge.sourceMod}` : ''} ${selectedEdge.constraint} ${selectedEdge.factor !== 1 ? `${selectedEdge.factor} * ` : ''}${selectedEdge.target} ${selectedEdge.offset !== 0 ? (selectedEdge.offset > 0 ? `+ ${selectedEdge.offset}` : `${selectedEdge.offset}`) : ''}`}
                </p>
              </div>
            )}
            {!selectedNode && !selectedEdge && <div className="text-gray-400">Select a node or edge to edit.</div>}
          </div>
        </div>

        {/* Right: Algorithm Log */}
        <div className="w-1/3 flex flex-col bg-gray-50 border-l">
          <div className="p-4 bg-white border-b font-bold text-gray-700 flex justify-between items-center">
            <span>Algorithm Log</span>
            <span className="text-xs font-normal text-gray-500 bg-gray-100 px-2 py-1 rounded">{algo}</span>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {steps.slice(0, currentStepIndex + 1).reverse().map((step, i) => (
              <div key={step.step} className={`p-3 rounded border ${i === 0 ? 'bg-blue-50 border-blue-200 shadow-sm' : 'bg-white border-gray-200 opacity-70'}`}>
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span className="font-bold">Step {step.step}</span>
                  {algo === 'AC3' && <span>Queue: {step.queue?.length}</span>}
                </div>

                {/* AC-3 Specifics */}
                {algo === 'AC3' && step.currentArc && (
                  <div className="text-sm font-medium text-gray-800 mb-1">
                    Checking Arc: <span className="text-blue-600">{step.currentArc}</span>
                  </div>
                )}

                {/* Backtracking Specifics */}
                {algo === 'Backtracking' && step.assignedVar && (
                  <div className="text-sm font-medium text-green-700 mb-1">
                    Assign {step.assignedVar} = {problemType === 'coloring' && COLOR_NAMES[step.assignedVal!] ? COLOR_NAMES[step.assignedVal!] : step.assignedVal}
                  </div>
                )}
                {algo === 'Backtracking' && step.backtrackFrom && (
                  <div className="text-sm font-medium text-red-700 mb-1">
                    Backtrack from {step.backtrackFrom}
                  </div>
                )}

                <div className="text-xs text-gray-600">{step.explanation}</div>

                {/* Removed Values (AC-3 or FC) */}
                {step.removedValues && step.removedValues.length > 0 && (
                  <div className="mt-2 text-xs bg-red-50 text-red-700 p-1 rounded border border-red-100">
                    Removed: {step.removedValues.map(r => `${r.val} from ${r.var}`).join(', ')}
                  </div>
                )}

                {/* Queue Table for AC-3 */}
                {algo === 'AC3' && step.queueDetails && (
                  <div className="mt-3">
                    <div className="text-xs font-bold text-gray-600 mb-1">Arc_Queue</div>
                    <div className="bg-slate-800 text-white p-2 rounded text-xs font-mono overflow-x-auto">
                      <div className="flex gap-2 items-center flex-wrap">
                        {/* Removed Arc (Strikethrough) */}
                        <span className="text-red-400 line-through decoration-2 opacity-70">
                          {step.queueDetails.removed}
                        </span>

                        {/* Active Queue Arcs */}
                        {step.queueDetails.active.map((arc, idx) => (
                          <span key={idx} className="text-slate-300">
                            {arc},
                          </span>
                        ))}

                        {/* Added Arcs (Highlight) */}
                        {step.queueDetails.added.map((arc, idx) => (
                          <span key={`added-${idx}`} className="text-green-400 font-bold animate-pulse">
                            {arc}
                          </span>
                        ))}
                      </div>
                      <div className="mt-1 pt-1 border-t border-slate-600 text-[10px] text-slate-400 flex justify-between">
                        <span>Processing: {step.queueDetails.removed}</span>
                        <span>Next: {step.queueDetails.active[0] || (step.queueDetails.added[0] ? step.queueDetails.added[0] : 'Empty')}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
            {steps.length === 0 && <div className="text-gray-400 text-center mt-10">Click "Run {algo}" to start.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}