import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * COE 292 — SVM Interactive (SVG)
 * -------------------------------------------------
 * Fixes applied:
 *  • Removed stray "refine/return" block that caused: SyntaxError 'return outside of function'.
 *  • Kept a clean, closed‑form max‑margin fitter that scans θ and solves b analytically.
 *  • Auxiliary margins (black long-dashed) are perpendicular to the selected (+1, −1) pair
 *    and PASS THROUGH those two points (matches lecture construction).
 *  • Added a tiny "Sanity Tests" section to validate math at runtime.
 *
 * Interactions
 *  • Click a point → toggle active/inactive
 *  • Shift+Click a point → permanently delete
 *  • Drag a point → move it
 *  • Add Mode → click empty canvas to place a new +1 or −1 point
 *  • Aux Pair → click "Select pair", then click one +1 and one −1 → black long-dashed helpers
 */

// ---------- World & helpers ----------
const WORLD = { xmin: 0, xmax: 8, ymin: 0, ymax: 8 };
// Responsive sizing - will be overridden by state
let WIDTH = 820, HEIGHT = 600;
const PAD = 50;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

function toScreen(x: number, y: number) {
  const u = (x - WORLD.xmin) / (WORLD.xmax - WORLD.xmin);
  const v = 1 - (y - WORLD.ymin) / (WORLD.ymax - WORLD.ymin);
  return { sx: PAD + u * (WIDTH - 2 * PAD), sy: PAD + v * (HEIGHT - 2 * PAD) };
}
function toWorld(sx: number, sy: number) {
  const u = (sx - PAD) / (WIDTH - 2 * PAD);
  const v = (sy - PAD) / (HEIGHT - 2 * PAD);
  return { x: WORLD.xmin + u * (WORLD.xmax - WORLD.xmin), y: WORLD.ymax - v * (WORLD.ymax - WORLD.ymin) };
}

// ---------- Types ----------
type Pt = { id: string; x: number; y: number; label: 1 | -1; active: boolean };

// ---------- Init data (≈ p1,p2,p3,n1,n2,n3) ----------
const INIT: Pt[] = [
  { id: "p1", x: 1,   y: 4,   label: +1, active: true },
  { id: "p2", x: 0,   y: 2,   label: +1, active: true },
  { id: "p3", x: 2.5, y: 7,   label: +1, active: true },
  { id: "n1", x: 3,   y: 2,   label: -1, active: true },
  { id: "n2", x: 4,   y: 4,   label: -1, active: true },
  { id: "n3", x: 5,   y: 6,   label: -1, active: true },
];

// ---------- SVM math (hard‑margin, linear) ----------
// Unit normal n = (cos θ, sin θ); line is n·x + b = 0. Signed distance of point (x,y) is n·x + b.
const signedDist = (p: Pt, n: {x:number;y:number}, b: number) => n.x * p.x + n.y * p.y + b;

function computeMarginAndSVs(points: Pt[], n: {x:number;y:number}, b: number) {
  const pos = points.filter(p => p.active && p.label === +1);
  const neg = points.filter(p => p.active && p.label === -1);
  if (pos.length === 0 || neg.length === 0) return { half: 0, svIds: [] as string[], violations: Infinity };

  let minPos = Infinity, minNeg = Infinity; const eps = 1e-6; const svIds = new Set<string>();
  let violations = 0;
  for (const p of points) {
    if (!p.active) continue;
    const d = signedDist(p, n, b);
    const correct = (d >= 0 && p.label === +1) || (d <= 0 && p.label === -1);
    if (!correct) { violations++; continue; }
    const mabs = Math.abs(d);
    if (p.label === +1) minPos = Math.min(minPos, mabs); else minNeg = Math.min(minNeg, mabs);
  }
  const half = Math.min(minPos, minNeg);
  if (!isFinite(half) || half <= eps) return { half: 0, svIds: [], violations };

  for (const p of points) {
    if (!p.active) continue; const d = Math.abs(signedDist(p, n, b));
    if (Math.abs(d - half) <= 1e-3) svIds.add(p.id);
  }
  return { half, svIds: [...svIds], violations };
}

// Analytic max‑margin for a given orientation (hard‑margin). We scan θ and compute
// the best feasible margin and corresponding b in closed form.
function fitMaxMargin(points: Pt[]): { valid: false } | { valid: true; theta: number; b: number; half: number; svIds: string[] } {
  const act = points.filter(p => p.active);
  const pos = act.filter(p => p.label === +1);
  const neg = act.filter(p => p.label === -1);
  if (pos.length === 0 || neg.length === 0) return { valid: false };

  let best: { valid: false } | { valid: true; theta: number; b: number; half: number; svIds: string[] } = { valid: false };
  const STEPS = 720; // 0.25° resolution
  for (let i = 0; i < STEPS; i++) {
    const theta = (i / STEPS) * Math.PI; // [0, π)
    const n = { x: Math.cos(theta), y: Math.sin(theta) }; // unit normal

    // Projections along n
    const posProj = pos.map(p => n.x * p.x + n.y * p.y);
    const negProj = neg.map(p => n.x * p.x + n.y * p.y);
    const minPos = Math.min(...posProj);
    const maxNeg = Math.max(...negProj);

    // Half‑margin for this θ (feasible iff > 0)
    const half = (minPos - maxNeg) / 2;
    if (half <= 1e-9) continue; // not separable for this θ

    // Optimal bias that centers H0 between the two extreme supports
    const b = - (minPos + maxNeg) / 2;

    // Support vectors = points that achieve the extremes
    const eps = 1e-4;
    const svIds = act.filter(p => {
      const s = n.x * p.x + n.y * p.y;
      return (p.label === 1 ? Math.abs(s - minPos) <= eps : Math.abs(s - maxNeg) <= eps);
    }).map(p => p.id);

    if (!best.valid || (best.valid && half > best.half)) best = { valid: true, theta, b, half, svIds };
  }
  return best;
}

// ---------- Component ----------
export default function SVMInteractiveAux() {
  const [pts, setPts] = useState<Pt[]>(() => INIT.map(p=>({...p})));
  const [dragId, setDragId] = useState<string|null>(null);
  const [auxPair, setAuxPair] = useState<{a?: string; b?: string} | null>(null);
  const [pairMode, setPairMode] = useState(false);

  // Add mode state
  const [addMode, setAddMode] = useState<{enabled:boolean; label: 1|-1}>({ enabled:false, label: 1 });

  // Select-then-click interaction mode
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);

  // Responsive canvas size
  const [canvasSize, setCanvasSize] = useState({ width: 820, height: 600 });

  useEffect(() => {
    const updateCanvasSize = () => {
      const isMobile = window.innerWidth < 768;
      const isTablet = window.innerWidth < 1024;

      if (isMobile) {
        const size = { width: Math.min(380, window.innerWidth - 32), height: 300 };
        setCanvasSize(size);
        WIDTH = size.width;
        HEIGHT = size.height;
      } else if (isTablet) {
        const size = { width: 600, height: 450 };
        setCanvasSize(size);
        WIDTH = size.width;
        HEIGHT = size.height;
      } else {
        const size = { width: 820, height: 600 };
        setCanvasSize(size);
        WIDTH = size.width;
        HEIGHT = size.height;
      }
    };

    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, []);

  // Current optimum (animate to updates)
  const [theta, setTheta] = useState(Math.PI/4);
  const [b, setB] = useState(0);
  const [half, setHalf] = useState(0);
  const [svIds, setSvIds] = useState<string[]>([]);

  // Refit on any data change
  useEffect(()=>{
    const fit = fitMaxMargin(pts);
    if (!fit.valid){ setSvIds([]); setHalf(0); return; }
    const startTheta = theta, startB = b, startHalf = half;
    const targetTheta = fit.theta, targetB = fit.b, targetHalf = fit.half;
    const T = 550; let raf = 0; let t0: number | null = null;
    const step = (now:number)=>{
      if (t0==null) t0 = now; const u = Math.min(1,(now-t0)/T); const w = 0.5-0.5*Math.cos(Math.PI*u);
      setTheta(startTheta + (targetTheta-startTheta)*w);
      setB(startB + (targetB-startB)*w);
      setHalf(startHalf + (targetHalf-startHalf)*w);
      if (u<1) raf=requestAnimationFrame(step); else setSvIds(fit.svIds);
    };
    raf=requestAnimationFrame(step); return ()=> cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(pts.map(p=>({id:p.id,x:p.x,y:p.y,label:p.label,active:p.active})))]);

  // Drag handling with touch support
  const svgRef = useRef<SVGSVGElement|null>(null);

  useEffect(()=>{
    if (!dragId || !svgRef.current) return;

    const updatePosition = (clientX: number, clientY: number) => {
      if (!svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const w = toWorld(clientX - rect.left, clientY - rect.top);
      setPts(ps=>ps.map(p=>p.id===dragId?{...p, x: clamp(w.x, WORLD.xmin, WORLD.xmax), y: clamp(w.y, WORLD.ymin, WORLD.ymax)}:p));
    };

    const onMouseMove = (e: MouseEvent) => updatePosition(e.clientX, e.clientY);
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      if (e.touches.length > 0) updatePosition(e.touches[0].clientX, e.touches[0].clientY);
    };
    const onEnd = () => setDragId(null);

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onEnd);
    window.addEventListener('touchmove', onTouchMove, { passive: false });
    window.addEventListener('touchend', onEnd);

    return ()=>{
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onEnd);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onEnd);
    };
  }, [dragId]);

  // Escape cancels add mode
  useEffect(()=>{
    const onKey=(e:KeyboardEvent)=>{ if (e.key==='Escape') setAddMode(m=>({...m, enabled:false})); };
    window.addEventListener('keydown', onKey); return ()=> window.removeEventListener('keydown', onKey);
  },[]);

  // Derived geometry
  const n = useMemo(()=>({ x: Math.cos(theta), y: Math.sin(theta) }),[theta]);
  const linePoint = useMemo(()=>({ x: -b*n.x, y: -b*n.y }),[b,n]); // any point on H0
  const tdir = useMemo(()=>({ x: -n.y, y: n.x }),[n]);
  const L = 100;
  const H0 = useMemo(()=>({ A:{x:linePoint.x - L*tdir.x, y: linePoint.y - L*tdir.y}, B:{x:linePoint.x + L*tdir.x, y: linePoint.y + L*tdir.y} }),[linePoint,tdir]);
  const H1 = useMemo(()=>({ A:{x:H0.A.x + n.x*half, y:H0.A.y + n.y*half}, B:{x:H0.B.x + n.x*half, y:H0.B.y + n.y*half} }),[H0,n,half]);
  const H2 = useMemo(()=>({ A:{x:H0.A.x - n.x*half, y:H0.A.y - n.y*half}, B:{x:H0.B.x - n.x*half, y:H0.B.y - n.y*half} }),[H0,n,half]);

  // Auxiliary margins from user pair (perpendicular to the segment connecting them)
  const auxLines = useMemo(() => {
    if (!auxPair?.a || !auxPair?.b) return null;
    const A = pts.find(p => p.id === auxPair.a && p.active);
    const B = pts.find(p => p.id === auxPair.b && p.active);
    if (!A || !B || A.label === B.label) return null;

    // Normal parallel to AB; dashed lines pass THROUGH A and B respectively
    const dx = B.x - A.x, dy = B.y - A.y;
    const len = Math.hypot(dx, dy);
    if (len < 1e-9) return null;
    const nAux = { x: dx / len, y: dy / len };
    const tAux = { x: -nAux.y, y: nAux.x };

    const lineThrough = (P: { x: number; y: number }) => {
      const b = - (nAux.x * P.x + nAux.y * P.y); // n·x + b = 0 through P
      const x0 = { x: -b * nAux.x, y: -b * nAux.y };
      return {
        A: { x: x0.x - L * tAux.x, y: x0.y - L * tAux.y },
        B: { x: x0.x + L * tAux.x, y: x0.y + L * tAux.y },
      };
    };

    return { La: lineThrough(A), Lb: lineThrough(B) };
  }, [auxPair, pts]);

  // Utilities
  const deletePoint = (id:string)=> setPts(ps=> ps.filter(p=> p.id !== id));
  const resetAll = ()=> setPts(INIT.map(p=>({...p})));
  const restoreAll = ()=> setPts(ps=>ps.map(p=>({...p, active:true})));
  const autoPairClosest = ()=>{
    const act = pts.filter(p=>p.active);
    let best:{a:string;b:string;d:number}|null=null;
    for (const a of act) for (const b of act){ if (a.label===b.label) continue; const d=(a.x-b.x)**2+(a.y-b.y)**2; if(!best||d<best.d) best={a:a.id,b:b.id,d}; }
    if (best) setAuxPair({ a: best.a, b: best.b });
  };

  const nextId = (label:1|-1)=>{
    const prefix = label===1? 'p' : 'n';
    const used = new Set(pts.map(p=>p.id));
    let k = 1; while (used.has(`${prefix}${k}`)) k++; return `${prefix}${k}`;
  };

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.target !== e.currentTarget) return; // only empty canvas
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const w = toWorld(e.clientX-rect.left, e.clientY-rect.top);

    // If a point is selected, move it to the clicked location
    if (selectedPointId) {
      setPts(ps => ps.map(p =>
        p.id === selectedPointId
          ? { ...p, x: clamp(w.x, WORLD.xmin, WORLD.xmax), y: clamp(w.y, WORLD.ymin, WORLD.ymax) }
          : p
      ));
      setSelectedPointId(null); // Deselect after moving
      return;
    }

    // Add mode functionality
    if (addMode.enabled) {
      const id = nextId(addMode.label);
      setPts(ps => ps.concat({ id, x: clamp(w.x, WORLD.xmin, WORLD.xmax), y: clamp(w.y, WORLD.ymin, WORLD.ymax), label: addMode.label, active: true }));
    }
  };

  // ---------- Render ----------
  return (
    <div className="w-full h-full p-4 flex flex-col gap-4 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Interactive SVM — Maximum Margin Classifier</h1>
        <div className="flex items-center gap-2">
          <button onClick={restoreAll} className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md hover:shadow-lg transition-all">Restore all</button>
          <button onClick={resetAll} className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-gray-600 to-gray-700 text-white shadow-md hover:shadow-lg transition-all">Reset</button>
        </div>
      </div>

      <div className="flex flex-col xl:grid xl:grid-cols-[1fr_320px] gap-4">
        <div className="rounded-2xl shadow p-3 bg-white">
          <svg
            ref={svgRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className="rounded-xl border border-gray-200 bg-white select-none"
            onClick={handleSvgClick}
          >
            {/* Grid & axes */}
            {Array.from({length: WORLD.xmax - WORLD.xmin + 1}).map((_,i)=>{
              const x = WORLD.xmin + i; const a = toScreen(x, WORLD.ymin), bpt = toScreen(x, WORLD.ymax);
              return <line key={'vx'+x} x1={a.sx} y1={a.sy} x2={bpt.sx} y2={bpt.sy} stroke="#000000" strokeWidth="0.5" strokeDasharray="1 3" />;
            })}
            {Array.from({length: WORLD.ymax - WORLD.ymin + 1}).map((_,i)=>{
              const y = WORLD.ymin + i; const a = toScreen(WORLD.xmin, y), bpt = toScreen(WORLD.xmax, y);
              return <line key={'hz'+y} x1={a.sx} y1={a.sy} x2={bpt.sx} y2={bpt.sy} stroke="#000000" strokeWidth="0.5" strokeDasharray="1 3" />;
            })}
            <line x1={toScreen(WORLD.xmin,0).sx} y1={toScreen(WORLD.xmin,0).sy} x2={toScreen(WORLD.xmax,0).sx} y2={toScreen(WORLD.xmax,0).sy} stroke="#000000" strokeWidth="2" strokeDasharray="2 2" />
            <line x1={toScreen(0,WORLD.ymin).sx} y1={toScreen(0,WORLD.ymin).sy} x2={toScreen(0,WORLD.ymax).sx} y2={toScreen(0,WORLD.ymax).sy} stroke="#000000" strokeWidth="2" strokeDasharray="2 2" />

            {/* Auxiliary margins (black long-dash pattern) */}
            {auxLines && (
              <>
                <line x1={toScreen(auxLines.La.A.x, auxLines.La.A.y).sx} y1={toScreen(auxLines.La.A.x, auxLines.La.A.y).sy} x2={toScreen(auxLines.La.B.x, auxLines.La.B.y).sx} y2={toScreen(auxLines.La.B.x, auxLines.La.B.y).sy} stroke="#000000" strokeDasharray="12 6" strokeWidth={3} />
                <line x1={toScreen(auxLines.Lb.A.x, auxLines.Lb.A.y).sx} y1={toScreen(auxLines.Lb.A.x, auxLines.Lb.A.y).sy} x2={toScreen(auxLines.Lb.B.x, auxLines.Lb.B.y).sx} y2={toScreen(auxLines.Lb.B.x, auxLines.Lb.B.y).sy} stroke="#000000" strokeDasharray="12 6" strokeWidth={3} />
              </>
            )}

            {/* SVM margins & boundary */}
            <line x1={toScreen(H1.A.x,H1.A.y).sx} y1={toScreen(H1.A.x,H1.A.y).sy} x2={toScreen(H1.B.x,H1.B.y).sx} y2={toScreen(H1.B.x,H1.B.y).sy} stroke="#3B82F6" strokeDasharray="4 8" strokeWidth={3} />
            <line x1={toScreen(H0.A.x,H0.A.y).sx} y1={toScreen(H0.A.x,H0.A.y).sy} x2={toScreen(H0.B.x,H0.B.y).sx} y2={toScreen(H0.B.x,H0.B.y).sy} stroke="#8B5CF6" strokeWidth={4} />
            <line x1={toScreen(H2.A.x,H2.A.y).sx} y1={toScreen(H2.A.x,H2.A.y).sy} x2={toScreen(H2.B.x,H2.B.y).sx} y2={toScreen(H2.B.x,H2.B.y).sy} stroke="#EF4444" strokeDasharray="4 8" strokeWidth={3} />

            {/* Points */}
            {pts.map(p=>{
              const s = toScreen(p.x,p.y); const active = p.active;
              const isSV = svIds.includes(p.id);
              const fill = p.label===1?"#3B82F6":"#EF4444"; // blue for +1, red for -1
              let stroke = "#000000";
              let strokeDasharray = "";
              if (selectedPointId === p.id) {
                stroke = "#000000";
                strokeDasharray = "2 2";
              } else if (!active) {
                stroke = "#808080";
                strokeDasharray = "3 3";
              } else if (isSV) {
                stroke = "#000000";
                strokeDasharray = "6 2";
              } else {
                stroke = "#000000";
                strokeDasharray = "";
              }
              const r = isSV?8:6;
              return (
                <g key={p.id}
                   onMouseDown={()=>{ setDragId(p.id); }}
                   onTouchStart={(e)=>{ e.preventDefault(); e.stopPropagation(); setDragId(p.id); }}
                   onClick={(e)=>{
                     if (pairMode){
                       setAuxPair(prev=>{
                         if (!prev || (!prev.a && !prev.b)) return { a: p.label===1? p.id: undefined, b: p.label===-1? p.id: undefined };
                         if (prev.a && !prev.b && p.label===-1) return { a: prev.a, b: p.id };
                         if (prev.b && !prev.a && p.label===+1) return { a: p.id, b: prev.b };
                         if (prev.a && prev.b) return { a: p.label===1? p.id: prev.a, b: p.label===-1? p.id: prev.b };
                         return { a: p.label===1? p.id: undefined, b: p.label===-1? p.id: undefined };
                       });
                     } else if ((e as React.MouseEvent).shiftKey) {
                       deletePoint(p.id);
                     } else if (selectedPointId === p.id) {
                       // Second click on selected point - toggle active state
                       setPts(ps=>ps.map(q=> q.id===p.id?{...q, active:!q.active}:q));
                       setSelectedPointId(null);
                     } else {
                       // First click - select point for movement
                       setSelectedPointId(p.id);
                     }
                   }}
                   style={{ cursor: "pointer", touchAction: "none" }}>
                  <circle cx={s.sx} cy={s.sy} r={r} fill={active?fill:"#f3f4f6"} stroke={stroke} strokeWidth={selectedPointId === p.id ? 5 : isSV?4:3} strokeDasharray={strokeDasharray} />
                  {selectedPointId === p.id && (
                    <circle
                      cx={s.sx} cy={s.sy} r={r + 8}
                      fill="none"
                      stroke="#000000"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      opacity={0.8}
                    >
                      <animate attributeName="r" values={`${r + 8};${r + 12};${r + 8}`} dur="1.5s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values="0.8;0.4;0.8" dur="1.5s" repeatCount="indefinite" />
                    </circle>
                  )}
                  <text x={s.sx+14} y={s.sy-14} className="text-xl font-bold fill-black select-none" style={{textShadow: '0 0 5px white, 0 0 5px white, 0 0 5px white, 0 0 5px white'}}>{p.id}</text>
                </g>
              );
            })}

            {/* خى  */}
            <text x={WIDTH-PAD-10} y={PAD-18} className="text-lg font-bold fill-black" textAnchor="end">Margin = {(2*half).toFixed(3)}</text>
          </svg>

          {/* Legend below canvas */}
          <div className="mt-3 p-3 bg-gray-50 rounded-xl">
            <div className="flex flex-wrap gap-x-6 gap-y-2 items-center text-sm font-semibold">
              <div className="flex items-center gap-2">
                <svg width="20" height="20">
                  <circle cx="10" cy="10" r="7" fill="#3B82F6" stroke="#1E40AF" strokeWidth="3" />
                </svg>
                <span className="font-bold text-blue-600">Class +1</span>
              </div>
              <div className="flex items-center gap-2">
                <svg width="20" height="20">
                  <circle cx="10" cy="10" r="7" fill="#EF4444" stroke="#B91C1C" strokeWidth="3" />
                </svg>
                <span className="font-bold text-red-600">Class −1</span>
              </div>
              <div className="flex items-center gap-2">
                <svg width="20" height="20">
                  <circle cx="10" cy="10" r="7" fill="#3B82F6" stroke="#1E40AF" strokeWidth="3" strokeDasharray="6 2" />
                </svg>
                <span className="font-semibold">Support Vector</span>
              </div>
              <div className="flex items-center gap-2">
                <svg width="35" height="20">
                  <line x1="0" y1="10" x2="35" y2="10" stroke="#8B5CF6" strokeWidth="4"/>
                </svg>
                <span className="font-semibold text-purple-600">H₀ Decision Boundary</span>
              </div>
              <div className="flex items-center gap-2">
                <svg width="35" height="20">
                  <line x1="0" y1="10" x2="35" y2="10" stroke="#3B82F6" strokeDasharray="4 8" strokeWidth="3"/>
                  <line x1="0" y1="10" x2="35" y2="10" stroke="#EF4444" strokeDasharray="4 8" strokeWidth="3" transform="translate(0, 5)"/>
                </svg>
                <span className="font-semibold text-gray-700">H₁ & H₂ Margins</span>
              </div>
              <div className="flex items-center gap-2">
                <svg width="35" height="20">
                  <line x1="0" y1="10" x2="35" y2="10" stroke="#000000" strokeDasharray="12 6" strokeWidth="3"/>
                </svg>
                <span>Auxiliary Margins</span>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl shadow bg-white p-4 space-y-4 text-sm text-gray-700 leading-6">
          <h2 className="text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Controls</h2>
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              <button className={`px-3 py-1.5 rounded-xl shadow-md hover:shadow-lg transition-all ${pairMode? 'bg-gradient-to-r from-rose-600 to-pink-600 text-white':'bg-white text-gray-700 border-2 border-rose-300 hover:border-rose-500'}`} onClick={()=> setPairMode(m=>!m)}>
                {pairMode? 'Selecting… click +1 then −1':'Auxiliary: Select pair'}
              </button>
              <button className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-rose-500 to-pink-500 text-white shadow-md hover:shadow-lg transition-all" onClick={autoPairClosest}>Auxiliary: Auto closest pair</button>
              <button className="px-3 py-1.5 rounded-xl bg-white text-gray-700 border-2 border-gray-300 hover:border-gray-500 shadow hover:shadow-md transition-all" onClick={()=> setAuxPair(null)}>Clear aux</button>
            </div>
            <p className="text-xs text-gray-500">Auxiliary margins are perpendicular to the line joining the selected opposite‑class pair and pass through those two points. They appear as black long-dashed lines.</p>
          </div>

          {/* Add / Remove */}
          <div className="space-y-3">
            <div className="font-medium">Add / Remove</div>
            <div className="flex flex-wrap gap-2">
              <button className={`px-3 py-1.5 rounded-xl shadow-md hover:shadow-lg transition-all ${addMode.enabled && addMode.label===1? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold':'bg-white text-gray-700 border-2 border-blue-300 hover:border-blue-500'}`} onClick={()=> setAddMode({enabled:true,label:1})}>➕ Add +1 (click canvas)</button>
              <button className={`px-3 py-1.5 rounded-xl shadow-md hover:shadow-lg transition-all ${addMode.enabled && addMode.label===-1? 'bg-gradient-to-r from-orange-500 to-red-500 text-white font-bold':'bg-white text-gray-700 border-2 border-orange-300 hover:border-orange-500'}`} onClick={()=> setAddMode({enabled:true,label:-1})}>➕ Add −1 (click canvas)</button>
              <button className="px-3 py-1.5 rounded-xl bg-white text-gray-700 border-2 border-gray-300 hover:border-gray-500 shadow hover:shadow-md transition-all" onClick={()=> setAddMode(m=>({...m,enabled:false}))}>Cancel Add</button>
              <button className="px-3 py-1.5 rounded-xl bg-white text-gray-700 border-2 border-gray-300 hover:border-gray-500 shadow hover:shadow-md transition-all" onClick={()=> setPts(ps=> ps.length>0? ps.slice(0,-1): ps)}>Remove last</button>
            </div>
            <div className="text-xs text-gray-500">In add mode, click an empty spot on the canvas to place a point. Shift+Click any point to delete it permanently. Regular click toggles active status.</div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-50 rounded-xl p-3 border">
              <div className="text-xs text-gray-700">θ (deg)</div>
              <div className="text-lg font-mono">{(theta*180/Math.PI).toFixed(2)}</div>
            </div>
            <div className="bg-gray-50 rounded-xl p-3 border">
              <div className="text-xs text-gray-700">Half‑margin ρ</div>
              <div className="text-lg font-mono">{half.toFixed(4)}</div>
            </div>
            <div className="bg-slate-50 rounded-xl p-3 border col-span-2">
              <div className="text-xs text-slate-500">Support vectors</div>
              <div className="text-sm font-mono break-words">{svIds.join(', ') || '—'}</div>
            </div>
          </div>

          <div className="text-xs text-gray-500">
            Tip: removing a non‑support point usually does NOT change H0. Removing a support often does, because it changes which constraints are active at optimum.
          </div>

          {/* --- Sanity Tests (runtime) --- */}
          <SanityTests />
        </div>
      </div>
    </div>
  );
}

// ---------- Tiny runtime tests to validate math ----------
function SanityTests() {
  type T = { name: string; pass: boolean; details: string };
  const tests: T[] = [];

  // Test 1: two points only (+1 and −1) → margin = distance/2, both are SVs
  const A: Pt = { id: 'A', x: 1, y: 1, label: 1, active: true };
  const B: Pt = { id: 'B', x: 4, y: 5, label: -1, active: true };
  const fitAB = fitMaxMargin([A, B]);
  const distAB = Math.hypot(B.x - A.x, B.y - A.y);
  tests.push({
    name: 'Two points (opposite classes) separable',
    pass: !!fitAB.valid && Math.abs(2*fitAB.half - distAB) < 1e-2 && (fitAB.svIds.includes('A') && fitAB.svIds.includes('B')),
    details: `half=${fitAB.half?.toFixed(3)}, dist/2=${(distAB/2).toFixed(3)}, SVs=${fitAB.svIds?.join(',')}`,
  });

  // Test 2: clearly separable clusters (horizontal split)
  const C1: Pt[] = [
    { id: 'c1', x: 1, y: 1, label: -1, active: true },
    { id: 'c2', x: 2, y: 1.2, label: -1, active: true },
    { id: 'c3', x: 3, y: 1.1, label: -1, active: true },
  ];
  const C2: Pt[] = [
    { id: 'd1', x: 1, y: 6, label: 1, active: true },
    { id: 'd2', x: 2, y: 6.2, label: 1, active: true },
    { id: 'd3', x: 3, y: 5.7, label: 1, active: true },
  ];
  const fitSep = fitMaxMargin([...C1, ...C2]);
  tests.push({
    name: 'Separable clusters → valid fit with positive margin',
    pass: !!fitSep.valid && fitSep.half > 0.3,
    details: `valid=${fitSep.valid}, half=${fitSep.half?.toFixed(3)}`,
  });

  // Test 3: non‑separable (labels conflict)
  const bad: Pt[] = [
    { id: 'k1', x: 0, y: 0, label: 1, active: true },
    { id: 'k2', x: 0.2, y: 0.2, label: -1, active: true },
    { id: 'k3', x: 0.1, y: 0.1, label: 1, active: true },
  ];
  const fitBad = fitMaxMargin(bad);
  tests.push({
    name: 'Non‑separable tiny set',
    pass: !fitBad.valid, details: `valid=${fitBad.valid}`
  });

  return (
    <div className="mt-3 border-t pt-3">
      <div className="text-xs font-semibold mb-2 text-black">Sanity Tests</div>
      <div className="grid grid-cols-1 gap-2">
        {tests.map((t, i) => (
          <div key={i} className={`rounded-md p-2 text-xs border ${t.pass ? 'bg-white border-black text-black' : 'bg-gray-100 border-gray-400 text-black'}`}>
            <div className="font-medium">{t.pass ? '✓ PASS' : '✗ FAIL'} — {t.name}</div>
            <div className="opacity-80 font-mono">{t.details}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
