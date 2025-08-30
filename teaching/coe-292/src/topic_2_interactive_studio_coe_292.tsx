import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Play, Pause, RotateCcw, StepForward, CheckCircle2, CircleHelp, Brain, Hammer, Trees, GitBranch } from "lucide-react";

// ===============================
// Topic 2 Playground – Goal Trees & Problem Solving
// COE 292 – Introduction to AI
// ===============================
// Single-file React app with mini-demos for Topic 2:
// - Goal representation & goal trees (AND/OR)
// - Tower of Hanoi (problem reduction + recursion)
// - Recursion sandbox (factorial)
// - Generate & Test demo (naïve vs pruned search)
// - Self-check quiz (on its own tab)
// Styling: TailwindCSS; Components: shadcn/ui; Animations: Framer Motion

// ---------- Utility ----------
const cn = (...cls: Array<string | false | null | undefined>) => cls.filter(Boolean).join(" ");

function useInterval(callback: () => void, delay: number | null) {
  const savedRef = useRef(callback);
  useEffect(() => { savedRef.current = callback; }, [callback]);
  useEffect(() => {
    if (delay === null) return;
    const id = setInterval(() => savedRef.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

// Fisher–Yates shuffle (with optional deterministic RNG for tests)
function shuffle<T>(arr: T[], rng?: () => number): T[] {
  const a = [...arr];
  const R = rng ?? Math.random;
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(R() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Evaluate a generic AND/OR tree node with optional `done` flags on leaves
function evaluate(node: any): boolean {
  if (!node || !node.children || node.children.length === 0) return !!node?.done;
  if (node.type === "AND") return node.children.every((c: any) => evaluate(c));
  return node.children.some((c: any) => evaluate(c));
}

// ---------- Components ----------
function Header() {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <motion.div initial={{ rotate: -10, scale: 0.9 }} animate={{ rotate: 0, scale: 1 }} className="p-2 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow">
          <Brain className="w-6 h-6" />
        </motion.div>
        <div>
          <h1 className="text-2xl md:text-3xl font-bold leading-tight">Topic 2 Playground</h1>
          <p className="text-sm text-muted-foreground">Goal Trees & Problem Solving — COE 292</p>
        </div>
      </div>
      <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground">
        <span className="px-2 py-1 rounded-full bg-muted">Interactive</span>
        <span className="px-2 py-1 rounded-full bg-muted">No math panic 😊</span>
      </div>
    </div>
  );
}

function LearningGoals() {
  const items = [
    { icon: <Trees className="w-4 h-4" />, text: "Read and build AND/OR goal trees" },
    { icon: <GitBranch className="w-4 h-4" />, text: "Reduce a problem into subgoals (problem reduction)" },
    { icon: <Hammer className="w-4 h-4" />, text: "Apply recursion with base/recursive cases" },
    { icon: <CircleHelp className="w-4 h-4" />, text: "Contrast naïve Generate & Test vs. simple pruning" },
  ];
  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">What you will practice</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="grid sm:grid-cols-2 gap-2">
          {items.map((it, i) => (
            <li key={i} className="flex items-center gap-2 p-2 rounded-xl bg-muted">
              <span className="text-muted-foreground">{it.icon}</span>
              <span>{it.text}</span>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-sm text-muted-foreground">
          This playground follows the structure and vocabulary used in the course slides for Topic 2 (Goal Trees & Problem Solving).
        </p>
      </CardContent>
    </Card>
  );
}

function Overview() {
  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Cheat‑Sheet (Topic 2)</CardTitle>
      </CardHeader>
      <CardContent className="text-sm space-y-2">
        <ul className="list-disc ml-5">
          <li><strong>State-space problem:</strong> specify initial state, goal state, all possible states, and constraints before search.</li>
          <li><strong>Goal tree:</strong> hierarchical breakdown of a goal into subgoals; leaves are terminal subgoals. <em>Note:</em> some leaves may be unused in the final solution.</li>
          <li><strong>AND node vs OR node:</strong> AND = all children must be achieved; OR = any one child suffices.</li>
          <li><strong>Hanoi reduction:</strong> <code>nAC → (n−1)AB</code>, <code>1AC</code>, <code>(n−1)BC</code> (AND).</li>
          <li><strong>Recursion:</strong> base case stops; recursive case reduces to a smaller instance.</li>
          <li><strong>Generate & Test:</strong> generate candidates then test; pruning improves efficiency.</li>
        </ul>
        <p className="text-muted-foreground">Tip: When stuck, sketch an AND/OR structure and mark which leaves are already satisfied.</p>
      </CardContent>
    </Card>
  );
}

// ---------- Tower of Hanoi ----------
function hanoiMoves(n: number, from: string, to: string, aux: string, acc: Array<[string, string]> = []) {
  if (n <= 0) return acc;
  hanoiMoves(n - 1, from, aux, to, acc);
  acc.push([from, to]);
  hanoiMoves(n - 1, aux, to, from, acc);
  return acc;
}

function Hanoi() {
  // Canvas‑style Hanoi like the Open Book Project demo; plus hints & undo
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [n, setN] = useState(3);
  const [pegs, setPegs] = useState<{ [k: string]: number[] }>({ A: [], B: [], C: [] });
  const [moves, setMoves] = useState<Array<[string, string]>>([]); // optimal moves for auto solve
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [delay, setDelay] = useState(400);
  const [selected, setSelected] = useState<string | null>(null); // manual move: selected peg key
  const [manualCount, setManualCount] = useState(0);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [showHints, setShowHints] = useState(true);
  const [history, setHistory] = useState<{ A:number[]; B:number[]; C:number[]; }[]>([]);
  const [invalidMove, setInvalidMove] = useState<string | null>(null);
  const [mode, setMode] = useState<'select' | 'manual' | 'auto'>('select');

  const init = (disks: number) => {
    const initial = { A: Array.from({ length: disks }, (_, i) => disks - i), B: [], C: [] } as { [k: string]: number[] };
    setPegs(initial as any);
    setMoves(hanoiMoves(disks, "A", "C", "B"));
    setIdx(0);
    setPlaying(false);
    setSelected(null);
    setManualCount(0);
    setHistory([]);
    setStatusMsg("");
    setInvalidMove(null);
    setMode('select');
  };

  useEffect(() => { init(n); }, [n]);

  // ---- Derived helpers ----
  const minMoves = Math.pow(2, n) - 1;
  const isSolved = pegs.C?.length === n;
  const initialA = useMemo(() => Array.from({ length: n }, (_, i) => n - i), [n]);
  const isInitial = useMemo(() => (
    JSON.stringify(pegs.A) === JSON.stringify(initialA) && (pegs.B?.length ?? 0) === 0 && (pegs.C?.length ?? 0) === 0 && idx === 0
  ), [pegs, initialA, idx]);

  // ---- Drawing ----
  const draw = () => {
    const cvs = canvasRef.current; if (!cvs) return;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;
    const W = cvs.width, H = cvs.height;
    ctx.clearRect(0, 0, W, H);

    // base
    ctx.fillStyle = '#ddd';
    ctx.fillRect(20, H - 30, W - 40, 10);

    const pegX = [W * 0.2, W * 0.5, W * 0.8];
    const pegKeys: Array<'A'|'B'|'C'> = ['A','B','C'];

    // pegs
    ctx.fillStyle = '#888';
    pegX.forEach(x => { ctx.fillRect(x - 4, H - 160, 8, 130); });

    // labels
    ctx.fillStyle = '#444';
    ctx.font = '12px system-ui, sans-serif';
    pegKeys.forEach((k, i) => { const x = pegX[i]; ctx.fillText(k, x - 3, H - 40); });

    // compute hints
    const top = (arr: number[]) => arr[arr.length - 1];
    const legalTargetsFrom = (src: 'A'|'B'|'C') => pegKeys.filter(dst => dst !== src && (pegs[dst].length === 0 || top(pegs[dst]) > top(pegs[src])));
    const legalSources = pegKeys.filter(src => pegs[src].length > 0 && legalTargetsFrom(src as 'A'|'B'|'C').length > 0);

    // highlight invalid move (red)
    if (invalidMove) {
      const i = pegKeys.indexOf(invalidMove as 'A'|'B'|'C');
      if (i >= 0) {
        ctx.strokeStyle = '#ef4444'; // red for invalid moves
        ctx.lineWidth = 3;
        ctx.strokeRect(pegX[i] - 60, H - 170, 120, 150);
      }
    }

    // highlight selected peg or hints
    if (selected) {
      const i = pegKeys.indexOf(selected as 'A'|'B'|'C');
      if (i >= 0) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.strokeRect(pegX[i] - 60, H - 170, 120, 150);
      }
      if (showHints) {
        ctx.strokeStyle = '#fbbf24'; // amber for legal targets
        ctx.lineWidth = 2;
        legalTargetsFrom(selected as 'A'|'B'|'C').forEach(dst => {
          const j = pegKeys.indexOf(dst);
          ctx.strokeRect(pegX[j] - 60, H - 170, 120, 150);
        });
      }
    } else if (showHints) {
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      legalSources.forEach(src => {
        const i = pegKeys.indexOf(src);
        ctx.strokeRect(pegX[i] - 60, H - 170, 120, 150);
      });
    }

    // draw disks
    const maxWidth = 120; // visual width span for largest disk
    const diskHeight = 14;
    const colorFor = (d: number) => `hsl(${(d*40)%360} 65% 55%)`;

    pegKeys.forEach((k, i) => {
      const stack = pegs[k];
      stack.forEach((d, idxOnPeg) => {
        const w = 30 + (d / n) * maxWidth;
        const x = pegX[i];
        const y = H - 40 - idxOnPeg * (diskHeight + 2);
        ctx.fillStyle = colorFor(d);
        const anyCtx = ctx as any;
        if (typeof anyCtx.roundRect === 'function') {
          ctx.beginPath();
          anyCtx.roundRect(x - w/2, y - diskHeight, w, diskHeight, 6);
          ctx.fill();
        } else {
          ctx.fillRect(x - w/2, y - diskHeight, w, diskHeight);
        }
        ctx.fillStyle = 'rgba(255,255,255,0.25)';
        ctx.fillRect(x - w/2, y - diskHeight, w, 2);
      });
    });
  };
  useEffect(draw, [pegs, selected, n, showHints, invalidMove]);

  // ---- Auto step ----
  const total = moves.length; // optimal = 2^n - 1
  const stepAuto = () => {
    if (idx >= total) return;
    const [from, to] = moves[idx];
    setPegs(prev => {
      const nf: { [k: string]: number[] } = { ...prev, [from]: [...prev[from]], [to]: [...prev[to]] } as any;
      const disk = nf[from].pop();
      nf[to].push(disk as number);
      return nf;
    });
    setIdx(i => i + 1);
  };

  const handleToggleAuto = () => {
    if (!playing) {
      // about to start auto; ensure a clean initial state
      if (!isInitial) {
        init(n);
        setStatusMsg("Auto-solve starts from the initial state.");
        setMode('auto');
        setPlaying(true);
        return;
      }
      setMode('auto');
    }
    setPlaying(p => !p);
  };

  useInterval(() => { if (playing) { if (idx < total) stepAuto(); else setPlaying(false); } }, playing ? delay : null);

  // ---- Manual move (click peg areas) ----
  const pegFromX = (x: number) => {
    const cvs = canvasRef.current; if (!cvs) return null;
    const W = cvs.width; const pegX = [W*0.2, W*0.5, W*0.8];
    const keys: Array<'A'|'B'|'C'> = ['A','B','C'];
    let best: 'A'|'B'|'C' | null = null; let bestDist = Infinity;
    pegX.forEach((px, i) => { const d = Math.abs(px - x); if (d < bestDist) { bestDist = d; best = keys[i]; } });
    return best;
  };

  const handleCanvasInteraction = (clientX: number, currentTarget: HTMLCanvasElement) => {
    // Only allow interaction in manual mode
    if (mode !== 'manual') return;
    
    const rect = currentTarget.getBoundingClientRect();
    const x = clientX - rect.left; // px relative to canvas
    const target = pegFromX(x);
    if (!target) return;
    
    // Clear any previous invalid move indicator
    setInvalidMove(null);
    
    if (!selected) {
      if (pegs[target].length === 0) return; // ignore empty peg as source
      setSelected(target);
      return;
    }
    if (selected === target) { setSelected(null); return; }
    // attempt move selected -> target
    const fromArr = pegs[selected];
    const toArr = pegs[target];
    if (fromArr.length === 0) { setSelected(null); return; }
    const disk = fromArr[fromArr.length - 1];
    const ok = toArr.length === 0 || toArr[toArr.length - 1] > disk;
    if (!ok) { 
      setInvalidMove(target);
      setSelected(null); 
      return; 
    }
    setPegs(prev => {
      const snapshot = JSON.parse(JSON.stringify(prev));
      const nf: { [k: string]: number[] } = { ...prev, [selected]: [...prev[selected]], [target]: [...prev[target]] } as any;
      nf[selected!].pop();
      nf[target].push(disk);
      setHistory(h => [...h, snapshot]);
      return nf;
    });
    setManualCount(c => c + 1);
    setSelected(null);
  };

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    handleCanvasInteraction(e.clientX, e.currentTarget);
  };

  const onCanvasTouchEnd = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // Prevent scrolling and other default touch behaviors
    const touch = e.changedTouches[0];
    if (touch) {
      handleCanvasInteraction(touch.clientX, e.currentTarget);
    }
  };

  const onCanvasTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // Prevent scrolling but don't handle interaction here
  };

  const undo = () => {
    setHistory(h => {
      if (h.length === 0) return h;
      const last = h[h.length - 1];
      setPegs(last);
      setManualCount(c => Math.max(0, c - 1));
      setSelected(null);
      return h.slice(0, - 1);
    });
  };

  // Status when solved
  useEffect(() => {
    if (isSolved) {
      setPlaying(false);
      if (manualCount > 0) {
        setStatusMsg(manualCount === minMoves ? "Optimal solution! 🎉" : `Solved in ${manualCount} moves (optimal is ${minMoves}).`);
        // Return to select mode after manual completion
        setTimeout(() => setMode('select'), 2000);
      }
    }
  }, [isSolved, manualCount, minMoves]);

  // Auto-clear transient status messages
  useEffect(() => {
    if (!statusMsg) return;
    const t = setTimeout(() => setStatusMsg(""), 3000);
    return () => clearTimeout(t);
  }, [statusMsg]);

  // Auto-clear invalid move indicator
  useEffect(() => {
    if (!invalidMove) return;
    const t = setTimeout(() => setInvalidMove(null), 1500);
    return () => clearTimeout(t);
  }, [invalidMove]);

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Tower of Hanoi — Canvas Demo</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-sm" htmlFor="disks">Disks:</label>
            <Input id="disks" type="number" value={n} min={1} max={8} onChange={(e)=>setN(Math.max(1, Math.min(8, Number(e.target.value)||1)))} className="w-20" />
            <Button size="sm" variant="outline" onClick={()=>init(n)} aria-label="Reset">
              <RotateCcw className="w-4 h-4 mr-1" /> Reset
            </Button>
          </div>
          
          {mode === 'select' && (
            <div className="p-4 rounded-xl border-2 border-dashed border-muted-foreground/30 text-center">
              <p className="text-sm text-muted-foreground mb-3">Choose how to solve the Tower of Hanoi:</p>
              <div className="flex flex-wrap justify-center gap-2">
                <Button onClick={() => setMode('manual')} className="flex-1 min-w-32">
                  👤 Manual Play
                </Button>
                <Button onClick={() => setMode('auto')} variant="secondary" className="flex-1 min-w-32">
                  🤖 Auto Solve
                </Button>
              </div>
            </div>
          )}
          
          {mode === 'manual' && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium">👤 Manual Mode</span>
              <Button size="sm" variant="ghost" onClick={undo} disabled={history.length===0} aria-label="Undo last move">Undo</Button>
              <Button size="sm" variant="outline" onClick={() => setMode('select')} aria-label="Back to mode selection">Back</Button>
            </div>
          )}
          
          {mode === 'auto' && (
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-sm font-medium">🤖 Auto Mode</span>
              <div className="flex items-center gap-2">
                <Button size="sm" variant="secondary" onClick={handleToggleAuto} disabled={idx >= total} aria-label="Toggle Auto Solve">
                  {playing ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />} {playing?"Pause":"Play"}
                </Button>
                <Button size="sm" variant="secondary" onClick={stepAuto} disabled={idx >= total} aria-label="Step one move">
                  <StepForward className="w-4 h-4 mr-1" /> Next Move
                </Button>
                <Button size="sm" variant="outline" onClick={() => setMode('select')} aria-label="Back to mode selection">Back</Button>
              </div>
              <div className="flex items-center gap-2 ml-auto text-sm">
                <span>Speed</span>
                <input aria-label="Playback speed" type="range" min={100} max={1200} step={50} value={delay} onChange={(e)=>setDelay(Number(e.target.value))} />
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-2xl bg-muted">
            <canvas ref={canvasRef} width={540} height={220} onClick={onCanvasClick} onTouchStart={onCanvasTouchStart} onTouchEnd={onCanvasTouchEnd} className="w-full h-auto bg-background rounded-xl shadow-inner" />
            <div className="text-xs text-muted-foreground mt-2">
              {mode === 'select' && "Choose Manual Play or Auto Solve above to begin."}
              {mode === 'manual' && "Click or tap a source peg, then a destination peg to make a legal move."}
              {mode === 'auto' && "Watch the automatic solution or step through it manually."}
            </div>
          </div>

          <div className="space-y-3">
            <Card className="shadow-none border">
              <CardHeader className="py-2"><CardTitle className="text-base">Status</CardTitle></CardHeader>
              <CardContent className="text-sm space-y-2">
                <div className="flex justify-between"><span>Auto step:</span><span>{idx} / {total} (optimal {minMoves})</span></div>
                <div className="flex justify-between"><span>Manual moves:</span><span>{manualCount}</span></div>
                {statusMsg && <div className="text-xs rounded-md bg-emerald-50 text-emerald-700 px-2 py-1">{statusMsg}</div>}
                <Separator />
                <div className="flex items-center gap-2">
                  <Switch id="hints" checked={showHints} onCheckedChange={setShowHints} />
                  <label htmlFor="hints" className="text-sm">Show hints (legal pegs highlighted)</label>
                </div>
                <p className="text-muted-foreground">Recursive plan (problem reduction):</p>
                <div className="font-mono text-xs bg-muted p-2 rounded-xl">
                  {`Move n disks A→C via B:\n1) Move n-1 A→B\n2) Move 1 A→C\n3) Move n-1 B→C`}
                </div>
              </CardContent>
            </Card>
            <Card className="shadow-none border">
              <CardHeader className="py-2"><CardTitle className="text-base">AND/OR view for n=3 (example)</CardTitle></CardHeader>
              <CardContent><AndOrSnippet /></CardContent>
            </Card>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AndOrSnippet() {
  // A tiny static depiction of 3AC = (2AB AND 1AC AND 2BC)
  return (
    <div className="text-xs">
      <div className="font-semibold mb-1">Root (goal): 3AC</div>
      <div className="ml-3">
        <div className="inline-block px-2 py-1 rounded-full bg-emerald-100 text-emerald-700 mr-2">AND</div>
        <span className="text-muted-foreground">All subgoals must be satisfied</span>
        <ul className="list-disc ml-6 mt-2">
          <li>2AB</li>
          <li>1AC</li>
          <li>2BC</li>
        </ul>
      </div>
    </div>
  );
}

// ---------- Hanoi Goal Tree (deterministic AND decomposition) ----------
function GoalTreeRender({ node, depth }: { node: any; depth: number; }) {
  const isLeaf = node.type === 'LEAF';
  return (
    <div className="ml-2" style={{ marginLeft: depth * 12 }}>
      <div className={cn("p-2 rounded-xl border bg-background inline-flex items-center gap-2", isLeaf ? "border-emerald-300" : "border-sky-300")}> 
        <span className="text-xs px-2 py-0.5 rounded-full bg-muted">{isLeaf? 'LEAF' : 'AND'}</span>
        <span className="text-sm font-mono">{node.label}</span>
      </div>
      {!isLeaf && (
        <div className="ml-4 mt-1 space-y-1">
          {node.children.map((ch: any, i: number) => (<GoalTreeRender key={i} node={ch} depth={depth+1} />))}
        </div>
      )}
    </div>
  );
}

function HanoiGoalTree() {
  const [n, setN] = useState(3);
  const make = (k: number, from: string, to: string, aux: string): any => {
    if (k === 1) return { label: `1${from}${to}`, type: 'LEAF', children: [] };
    return {
      label: `${k}${from}${to}`,
      type: 'AND',
      children: [
        make(k-1, from, aux, to),
        { label: `1${from}${to}`, type: 'LEAF', children: [] },
        make(k-1, aux, to, from),
      ],
    };
  };
  const tree = useMemo(() => make(n, 'A','C','B'), [n]);

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2"><CardTitle className="text-lg">Hanoi Goal Tree (AND concept)</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2"><span className="text-sm">n:</span><Input className="w-24" type="number" min={1} max={6} value={n} onChange={(e)=>setN(Math.max(1, Math.min(6, Number(e.target.value)||1)))} /></div>
        <div className="rounded-2xl bg-muted p-3">
          <GoalTreeRender node={tree} depth={0} />
        </div>
        <p className="text-sm text-muted-foreground">Concept: Each <strong>{`kXY`}</strong> goal decomposes into <strong>AND</strong> of three subgoals: <strong>{`(k-1)X?`}</strong>, <strong>{`1XY`}</strong>, <strong>{`(k-1)?Y`}</strong>.</p>
      </CardContent>
    </Card>
  );
}

// ---------- Recursion Sandbox (Factorial) ----------
function RecursionSandbox() {
  const [n, setN] = useState(5);
  const [trace, setTrace] = useState<{k:number; type:'call'|'return'; value?: number;}[]>([]);

  useEffect(() => {
    const steps: {k:number; type:'call'|'return'; value?: number;}[] = [];
    function fact(k: number): number {
      steps.push({ k, type: "call" });
      if (k === 0 || k === 1) {
        steps.push({ k, type: "return", value: 1 });
        return 1;
      }
      const res = k * fact(k - 1);
      steps.push({ k, type: "return", value: res });
      return res;
    }
    fact(Math.max(0, Math.min(8, n)));
    setTrace(steps);
  }, [n]);

  const result = useMemo(() => {
    function f(x:number){ return x<=1?1:x*f(x-1);} return f(Math.max(0, Math.min(8, n))); }, [n]);

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Recursion Sandbox — Factorial</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-3">
          <label className="text-sm">n:</label>
          <Input type="number" value={n} min={0} max={8} className="w-24" onChange={(e)=>setN(Number(e.target.value)||0)} />
          <div className="text-sm ml-auto">Result: <span className="font-mono">{result}</span></div>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="rounded-2xl bg-muted p-3 h-64 overflow-auto">
            <div className="font-mono text-xs">
              {trace.map((t, i) => (
                <div key={i} className={cn("py-0.5", t.type === "call" ? "" : "font-semibold")}>{t.type === "call" ? `call  f(${t.k})` : `return f(${t.k}) = ${t.value}`}</div>
              ))}
            </div>
          </div>
          <div className="rounded-2xl bg-muted p-3 h-64 overflow-auto">
            <p className="text-sm mb-2 text-muted-foreground">Pseudocode</p>
            <pre className="text-xs font-mono whitespace-pre-wrap">{`function fact(n):
  if n == 0 or n == 1:
    return 1          # base case
  else:
    return n * fact(n-1)  # recursive case`}</pre>
            <Separator className="my-2" />
            <p className="text-sm text-muted-foreground">Idea: solve a large problem by solving a slightly smaller version of the same problem, and stop at the base case.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------- Generate & Test Demo ----------
function GenerateTest() {
  const [N, setN] = useState(1000);
  const [mode, setMode] = useState<'naive'|'pruned'>("naive"); // naive or pruned
  const [running, setRunning] = useState(false);
  const [i, setI] = useState(1);
  const [found, setFound] = useState<number | null>(null);

  const isGoal = (x: number) => x % 7 === 0 && x % 9 === 0; // divisible by 63

  const step = () => {
    if (found !== null) return;
    if (i > N) return;
    let next = i;
    if (mode === "pruned") {
      // generate only multiples of 7
      if (next % 7 !== 0) next = next + (7 - (next % 7));
    }
    if (next > N) return;
    if (isGoal(next)) {
      setFound(next);
      setI(next);
      setRunning(false);
    } else {
      setI(next + (mode === "pruned" ? 7 : 1));
    }
  };

  useInterval(() => { if (running) step(); }, running ? 50 : null);

  const reset = () => { setI(1); setFound(null); setRunning(false); };

  const theoretical = 63; // first LCM

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Generate & Test — Naïve vs Pruned</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <label>Search up to:</label>
          <Input className="w-24" type="number" value={N} min={1} onChange={(e)=>setN(Math.max(1, Number(e.target.value)||1))} />
          <div className="ml-auto flex items-center gap-2">
            <Button size="sm" variant="secondary" onClick={()=>setRunning(r=>!r)}>{running?"Pause":"Run"}</Button>
            <Button size="sm" variant="secondary" onClick={step}>Step</Button>
            <Button size="sm" variant="outline" onClick={reset}>Reset</Button>
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className={cn("px-2 py-1 rounded-full cursor-pointer", mode==="naive"?"bg-primary text-primary-foreground":"bg-muted")} onClick={()=>setMode("naive")}>Naïve (1,2,3,⋯)</span>
          <span className={cn("px-2 py-1 rounded-full cursor-pointer", mode==="pruned"?"bg-primary text-primary-foreground":"bg-muted")} onClick={()=>setMode("pruned")}>Pruned (multiples of 7)</span>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="rounded-2xl bg-muted p-3 h-56 overflow-auto font-mono text-xs">
            <div>testing: {i}</div>
            <div>found: {found===null?"(not yet)":found}</div>
            <Separator className="my-2" />
            <div>
              {Array.from({length: Math.min(i, 400)}, (_,k)=>k+1).filter(x=> mode==="naive"? x<=i : x<=i && x%7===0).map(x=> (
                <span key={x} className={cn("inline-block px-2 py-1 m-0.5 rounded", isGoal(x)?"bg-emerald-200":"bg-background border")}>{x}</span>
              ))}
            </div>
          </div>
          <div className="rounded-2xl bg-muted p-3 text-sm">
            <p><strong>Goal:</strong> find the smallest number ≤ N divisible by both 7 and 9.</p>
            <p className="text-muted-foreground">Naïve generate & test checks every candidate; a simple pruning strategy generates only multiples of 7, then tests the 9-condition.</p>
            <Separator className="my-2" />
            <ul className="list-disc ml-5 text-sm">
              <li>First solution theoretically = LCM(7,9) = {theoretical}.</li>
              <li>Compare <em>number of tests</em> performed by each strategy.</li>
              <li>Think: how would an AND/OR tree represent these constraints?</li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------- Quiz ----------
const QUESTIONS = [
  {
    q: "At an AND node in a goal tree, when is the node satisfied?",
    choices: [
      "When any child is satisfied",
      "When all children are satisfied",
      "When the parent is an OR node",
    ],
    ans: 1,
    why: "AND means all subgoals must be achieved to satisfy the node.",
  },
  {
    q: "Which decomposition matches the 3-disk Hanoi goal 3AC?",
    choices: ["2AC, 1AB, 2BC", "2AB, 1AC, 2BC", "2BC, 1BA, 2AC"],
    ans: 1,
    why: "Move 2 from A→B, move 1 from A→C, move 2 from B→C.",
  },
  {
    q: "In recursion, what is the base case for factorial?",
    choices: ["n = 2", "n = 0 or n = 1", "n = -1"],
    ans: 1,
    why: "By definition, 0! = 1 and 1! = 1.",
  },
  {
    q: "Generate & Test (pruned) reduces work by…",
    choices: ["testing more candidates", "generating fewer but better candidates", "changing the goal"],
    ans: 1,
    why: "Pruning narrows the generator so fewer checks are needed.",
  },
];

function Quiz() {
  // Shuffle answers per mount; reset button can reshuffle
  const [seed, setSeed] = useState(0);

  const shuffled = useMemo(() => (
    QUESTIONS.map(q => ({
      q: q.q,
      why: q.why,
      options: shuffle(q.choices.map((text, i) => ({ text, correct: i === q.ans })))
    }))
  ), [seed]);

  const [picked, setPicked] = useState<(number | null)[]>(Array(QUESTIONS.length).fill(null));
  const [graded, setGraded] = useState(false);

  const score = useMemo(() => picked.reduce((acc, choiceIdx, i) => {
    if (choiceIdx == null) return acc;
    return acc + (shuffled[i].options[choiceIdx].correct ? 1 : 0);
  }, 0), [picked, shuffled]);

  const resetAll = () => {
    setPicked(Array(QUESTIONS.length).fill(null));
    setGraded(false);
    setSeed(s => s + 1); // reshuffle answers
  };

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg text-center">Quick Self‑Check</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 max-w-2xl mx-auto text-center">
        {shuffled.map((item, qi) => (
          <div key={qi} className="p-3 rounded-xl border">
            <div className="font-medium mb-3">{qi + 1}. {item.q}</div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 justify-items-center">
              {item.options.map((opt, ci) => {
                const active = picked[qi] === ci;
                const correct = graded && opt.correct;
                const wrong = graded && active && !opt.correct;
                return (
                  <div key={ci}
                       onClick={() => !graded && setPicked(p => { const a = [...p]; a[qi] = ci; return a; })}
                       className={cn(
                         "w-full sm:w-72 cursor-pointer p-2 rounded-lg border",
                         active && !graded && "border-primary",
                         correct && "bg-emerald-100 border-emerald-300",
                         wrong && "bg-red-100 border-red-300"
                       )}>{opt.text}</div>
                );
              })}
            </div>
            {graded && (
              <div className="text-xs mt-2 text-muted-foreground">{item.why}</div>
            )}
          </div>
        ))}
        <div className="flex items-center justify-center gap-2">
          <Button onClick={() => setGraded(true)} variant="secondary">Grade</Button>
          <Button onClick={resetAll} variant="outline">Reset & Reshuffle</Button>
        </div>
        {graded && <div className="text-sm">Score: <strong>{score}</strong> / {shuffled.length}</div>}
      </CardContent>
    </Card>
  );
}

// ---------- Symbolic Integration (Problem Reduction) ----------
function SymbolicIntegration() {
  // Integrate (x^2 + 1)/x dx. Safe transformation → split into ∫x dx AND ∫1/x dx
  const tree = {
    label: "∫ (x^2 + 1)/x dx",
    type: "OR",
    children: [
      {
        label: "Safe: split fraction",
        type: "AND",
        children: [
          { label: "∫ x dx = x^2/2", type: "LEAF", children: [], done: true },
          { label: "∫ 1/x dx = ln|x|", type: "LEAF", children: [], done: true },
        ],
        done: true,
      },
      {
        label: "Heuristic: u‑substitution (u = x^2+1)",
        type: "OR",
        children: [ { label: "du = 2x dx → needs x dx present (dead end)", type: "LEAF", children: [], done: false } ],
        done: false,
      },
    ],
    done: true,
  } as const;

  const render = (node: any, depth=0) => (
    <div className="ml-2" style={{ marginLeft: depth * 12 }}>
      <div className={cn("p-2 rounded-xl border bg-background inline-flex items-center gap-2", node.done ? "border-emerald-300" : "border-transparent")}> 
        <span className="text-xs px-2 py-0.5 rounded-full bg-muted">{node.type}</span>
        <span className="text-sm">{node.label}</span>
        {node.done && <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
      </div>
      <div className="ml-4 mt-1 space-y-1">
        {node.children?.map((ch: any, i: number) => <div key={i}>{render(ch, depth+1)}</div>)}
      </div>
    </div>
  );

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Symbolic Integration — Problem Reduction & Goal Trees</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p>
          <em>Safe</em> vs <em>heuristic</em> transformations: safe transforms always help and often create <strong>AND</strong> nodes (multiple terms to integrate). Heuristics are optional <strong>OR</strong> choices that may or may not work.
        </p>
        <div className="rounded-2xl bg-muted p-3">{render(tree)}</div>
        <ul className="list-disc ml-5">
          <li>Safe branch splits into two integrals that <em>both</em> must be solved (AND).</li>
          <li>Heuristic branch is displayed but unused; illustrates that not all leaves are part of the final solution.</li>
          <li>Depth here is 2 (root → AND → leaves). Try drawing your own example with degree division.</li>
        </ul>
      </CardContent>
    </Card>
  );
}

// ---------- Developer Tests ----------
function DevTests() {
  const results = useMemo(() => {
    const out: {name:string; pass:boolean;}[] = [];
    const assert = (name: string, cond: any) => out.push({ name, pass: !!cond });

    // Test 1: hanoiMoves for n=3
    const m3 = hanoiMoves(3, "A", "C", "B", []);
    assert("hanoiMoves n=3 length = 7", m3.length === 7);
    assert("hanoiMoves n=3 first move A→C", m3[0][0] === "A" && m3[0][1] === "C");
    assert("hanoiMoves n=3 last move A→C", m3[m3.length - 1][0] === "A" && m3[m3.length - 1][1] === "C");

    // Additional tests for robustness
    const m1 = hanoiMoves(1, "A", "C", "B", []);
    assert("hanoiMoves n=1 length = 1", m1.length === 1);
    assert("hanoiMoves n=1 move is A→C", m1[0][0] === "A" && m1[0][1] === "C");
    const m0 = hanoiMoves(0, "A", "C", "B", []);
    assert("hanoiMoves n=0 length = 0", m0.length === 0);

    // NEW: min moves formula vs generator for n=4
    const m4 = hanoiMoves(4, "A", "C", "B", []);
    assert("hanoiMoves n=4 length = 15", m4.length === 15);

    // Test 2: evaluate AND/OR logic
    const tAND = { type: "AND", children: [{ done: true, children: [] }, { done: false, children: [] }] };
    const tOR = { type: "OR", children: [{ done: false, children: [] }, { done: true, children: [] }] };
    assert("evaluate AND(true,false) = false", evaluate(tAND) === false);
    assert("evaluate OR(false,true) = true", evaluate(tOR) === true);

    // NEW: nested structure truth
    const nested = { type: "AND", children: [ { type: "OR", children: [ { done: false, children: [] }, { done: true, children: [] } ] }, { done: true, children: [] } ] } as any;
    assert("evaluate AND( OR(false,true), true ) = true", evaluate(nested) === true);

    // Test 3: factorial
    function fact(n:number){ return n <= 1 ? 1 : n * fact(n-1); }
    assert("factorial(5) = 120", fact(5) === 120);

    // Test 4: isGoal check (63 is LCM of 7 and 9)
    const isGoal = (x:number) => x % 7 === 0 && x % 9 === 0;
    assert("isGoal(63) === true", isGoal(63) === true);
    assert("isGoal(64) === false", isGoal(64) === false);
    assert("isGoal(126) === true", isGoal(126) === true);

    // Test 5: HanoiGoalTree shape for n=2
    const make = (k:number, from:string, to:string, aux:string): any => k===1 ? { label:`1${from}${to}`, type:'LEAF', children:[] } : ({ label:`${k}${from}${to}`, type:'AND', children:[ { label:`1${from}${aux}`, type:'LEAF', children:[] }, { label:`1${from}${to}`, type:'LEAF', children:[] }, { label:`1${aux}${to}`, type:'LEAF', children:[] } ] });
    const t2 = make(2,'A','C','B');
    assert("HanoiGoalTree n=2 label is 2AC", t2.label === '2AC');
    assert("HanoiGoalTree n=2 middle child is 1AC", t2.children[1].label === '1AC');

    // Test 6: SymbolicIntegration tree leaves are typed as LEAF and marked done appropriately
    const intTree = {
      label: "∫ (x^2 + 1)/x dx",
      type: "OR",
      children: [
        { label: "Safe: split fraction", type: "AND", done: true, children: [
          { label: "∫ x dx = x^2/2", type: "LEAF", done: true, children: [] },
          { label: "∫ 1/x dx = ln|x|", type: "LEAF", done: true, children: [] },
        ]},
        { label: "Heuristic: u‑substitution (u = x^2+1)", type: "OR", done: false, children: [
          { label: "du = 2x dx → needs x dx present (dead end)", type: "LEAF", done: false, children: [] },
        ]},
      ], done: true } as any;
    const leaves = [intTree.children[0].children[0], intTree.children[0].children[1], intTree.children[1].children[0]] as any[];
    assert("Integration tree leaves use type=LEAF", leaves.every(l => l.type === 'LEAF'));
    assert("Integration safe leaves marked done", leaves[0].done === true && leaves[1].done === true);
    assert("Integration heuristic leaf not done", leaves[2].done === false);

        // NEW: shuffle utility preserves exactly one correct option
    const opts = [
      { text: 'A', correct: false },
      { text: 'B', correct: true },
      { text: 'C', correct: false },
      { text: 'D', correct: false },
    ];
    const rng = () => 0.42; // deterministic
    const sh = shuffle(opts, rng);
    assert("shuffle keeps same length", sh.length === opts.length);
    assert("shuffle keeps one correct option", sh.filter(o => o.correct).length === 1);
    assert("shuffle keeps set of texts", sh.map(o=>o.text).sort().join('') === opts.map(o=>o.text).sort().join(''));

    return out;
  }, []);

  const passed = results.filter(r => r.pass).length;

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Developer Tests</CardTitle>
      </CardHeader>
      <CardContent className="text-sm space-y-2">
        <div>Passed {passed} / {results.length} checks</div>
        <ul className="list-disc ml-5">
          {results.map((r, i) => (
            <li key={i} className={r.pass ? "text-emerald-700" : "text-red-700"}>
              {r.pass ? "✅" : "❌"} {r.name}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

// ---------- Main App ----------
export default function Topic2Playground() {
  return (
    <div className="p-4 md:p-8 max-w-6xl mx-auto">
      <Header />
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid grid-cols-2 md:grid-cols-7 gap-2 bg-muted p-1 rounded-2xl">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="hanoi">Hanoi</TabsTrigger>
          <TabsTrigger value="andor">Goal Tree</TabsTrigger>
          <TabsTrigger value="rec">Recursion</TabsTrigger>
          <TabsTrigger value="gt">Generate & Test</TabsTrigger>
          <TabsTrigger value="quiz">Quiz</TabsTrigger>
          <TabsTrigger value="integration">Integration</TabsTrigger>
        </TabsList>
        <div className="mt-4 grid gap-4">
          <TabsContent value="overview"><div className="grid lg:grid-cols-2 gap-4"><LearningGoals /><Overview /></div></TabsContent>
          <TabsContent value="hanoi"><Hanoi /></TabsContent>
          <TabsContent value="andor"><HanoiGoalTree /></TabsContent>
          <TabsContent value="rec"><RecursionSandbox /></TabsContent>
          <TabsContent value="gt"><GenerateTest /></TabsContent>
          <TabsContent value="quiz"><Quiz /></TabsContent>
          <TabsContent value="integration"><SymbolicIntegration /></TabsContent>
        </div>
      </Tabs>

      <div className="mt-6 grid gap-4">
        <DevTests />
      </div>

      <footer className="mt-6 text-xs text-muted-foreground">
        Built for COE 292 — Topic 2 (Goal Trees & Problem Solving). Use it as a study companion alongside the course slides.
      </footer>
    </div>
  );
}
