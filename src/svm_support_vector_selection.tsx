import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * COE 292 — SVM Interactive (SVG)
 * Goal: Add / remove points; recompute and PLOT the NEW maximum margin. Also show
 * an "auxiliary margin" pair (red dashed) based on a user‑chosen cross‑class pair,
 * as in the lecture figures.
 *
 * Key interactions
 *  • Click a point → toggle active/inactive
 *  • Shift+Click a point → permanently delete
 *  • Drag a point → move it
 *  • Add Mode → click empty canvas to place a new +1 or −1 point
 *  • Aux Pair: click "Select pair" then click one +1 and one −1 → shows red dashed
 *    auxiliary margins perpendicular to the segment joining them. Click "Clear" to remove.
 *  • Auto‑Fit always recomputes after any change and animates H0/H1/H2 to the new optimum.
 */

// ---------- World & helpers ----------
const WORLD = { xmin: 0, xmax: 8, ymin: 0, ymax: 8 };
const WIDTH = 820, HEIGHT = 600, PAD = 50;
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

// ---------- Init data (≈ your p1,p2,p3,n1,n2,n3) ----------
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

  // functional margins on each side among correctly classified points
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

  // SVs = points lying on the margins (within small tolerance)
  for (const p of points) {
    if (!p.active) continue; const d = Math.abs(signedDist(p, n, b));
    if (Math.abs(d - half) <= 1e-3) svIds.add(p.id);
  }
  return { half, svIds: [...svIds], violations };
}

// Coarse + refined search over θ∈[0,π) and b∈[bmin,bmax] to maximize hard‑margin (penalize violations strongly)
function fitMaxMargin(points: Pt[], currentTheta?: number) {
  const act = points.filter(p => p.active);
  const pos = act.filter(p => p.label === +1); const neg = act.filter(p => p.label === -1);
  if (pos.length === 0 || neg.length === 0) return { valid:false } as const;

  // Bias range heuristic from extreme projections
  const xs = act.map(p => p.x), ys = act.map(p => p.y);
  const xrange = Math.max(...xs) - Math.min(...xs); const yrange = Math.max(...ys) - Math.min(...ys);

  let best = { valid:false, theta: 0, b: 0, half: 0, svIds: [] as string[] };
  const tryThetaB = (theta:number, b:number, stabilityBonus: number = 0) => {
    const n = { x: Math.cos(theta), y: Math.sin(theta) };
    const { half, svIds, violations } = computeMarginAndSVs(points, n, b);
    // Hard‑margin: forbid violations, add stability bonus for orientations close to current
    const score = (violations === 0 ? half + stabilityBonus : -1e6 * violations);
    if (!best.valid || score > (best.half + (best.valid ? 0 : 0))) {
      best = { valid:true, theta, b, half, svIds };
    }
  };

  // Coarse θ/b search
  for (let i=0;i<181;i++){
    const theta = (i/180)*Math.PI;
    const n = { x: Math.cos(theta), y: Math.sin(theta) };

    // Add stability bonus for angles close to current orientation
    let stabilityBonus = 0;
    if (currentTheta !== undefined) {
      const angleDiff = Math.min(
        Math.abs(theta - currentTheta),
        Math.abs(theta - currentTheta + Math.PI),
        Math.abs(theta - currentTheta - Math.PI)
      );
      // Small bonus (up to 5% of typical margins) for staying close to current orientation
      stabilityBonus = Math.max(0, 0.1 - angleDiff) * 0.01;
    }

    // Project all points to get a safe bias span
    const projs = act.map(p => -(n.x*p.x + n.y*p.y));
    const pmin = Math.min(...projs), pmax = Math.max(...projs);
    const pad = 1.2 * Math.hypot(xrange, yrange) / 8; // small slack
    const bmin = pmin - pad, bmax = pmax + pad;
    for (let j=0;j<=160;j++){
      const t = j/160; const b = bmin + t*(bmax-bmin);
      tryThetaB(theta, b, stabilityBonus);
    }
  }
  // Local refine around best
  if (!best.valid) return best;
  const refine = (center:number, rad:number, steps:number) => {
    for (let i=0;i<=steps;i++){
      const theta = center - rad + (2*rad*i)/steps;
      const n = { x: Math.cos(theta), y: Math.sin(theta) };
      const projs = act.map(p => -(n.x*p.x + n.y*p.y));
      const bmin = Math.min(...projs) - 0.5, bmax = Math.max(...projs) + 0.5;
      for (let j=0;j<=steps;j++){
        const b = bmin + (bmax-bmin)*j/steps; tryThetaB(theta,b);
      }
    }
  };
  refine(best.theta, 0.08, 80);
  refine(best.theta, 0.02, 120);
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

  // Current optimum (animate to updates)
  const [theta, setTheta] = useState(Math.PI/4);
  const [b, setB] = useState(0);
  const [half, setHalf] = useState(0);
  const [svIds, setSvIds] = useState<string[]>([]);

  // Refit on any data change
  useEffect(()=>{
    const fit = fitMaxMargin(pts, theta);
    if (!fit.valid){ setSvIds([]); setHalf(0); return; }
    // soft animation by interpolation
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

  // Drag handling
  const svgRef = useRef<SVGSVGElement|null>(null);
  useEffect(()=>{
    const onMove=(e:MouseEvent)=>{
      if (!dragId || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const w = toWorld(e.clientX-rect.left, e.clientY-rect.top);
      setPts(ps=>ps.map(p=>p.id===dragId?{...p, x: clamp(w.x, WORLD.xmin, WORLD.xmax), y: clamp(w.y, WORLD.ymin, WORLD.ymax)}:p));
    };
    const onUp=()=> setDragId(null);
    window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
    return ()=>{ window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
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
  const auxLines = useMemo(()=>{
    if (!auxPair?.a || !auxPair?.b) return null;
    const A = pts.find(p=>p.id===auxPair.a && p.active); const Bp = pts.find(p=>p.id===auxPair.b && p.active);
    if (!A || !Bp || A.label===Bp.label) return null;
    const dx = Bp.x - A.x, dy = Bp.y - A.y; // pair vector
    // ✅ Correct construction: auxiliary margins must be PERPENDICULAR to the pair AB.
    // That means their NORMAL is PARALLEL to AB (n_aux ∥ AB), and the line DIRECTION is t_aux ⟂ AB.
    const thetaPair = Math.atan2(dy, dx);      // along the pair
    const nAux = { x: Math.cos(thetaPair), y: Math.sin(thetaPair) }; // normal parallel to AB
    const tAux = { x: -nAux.y, y: nAux.x };    // line direction (perpendicular to AB)

    // Place two parallel lines so that A and B lie on them, symmetrically around the midline.
    // Lines: nAux·x + c = ±k, where c = -(sA + sB)/2, k = (sA - sB)/2, sP = nAux·P
    const sA = nAux.x*A.x + nAux.y*A.y;
    const sB = nAux.x*Bp.x + nAux.y*Bp.y;
    const c  = -(sA + sB)/2;      // midline shift
    const k  =  (sA - sB)/2;      // half distance between the two margins in signed projection

    const makeSeg = (sign:number)=>{
      const b = c + sign*k;                 // nAux·x + b = 0
      const x0 = { x: -b*nAux.x, y: -b*nAux.y }; // any point on the line
      return { A:{ x: x0.x - L*tAux.x, y: x0.y - L*tAux.y }, B:{ x: x0.x + L*tAux.x, y: x0.y + L*tAux.y } };
    };

    // Calculate auxiliary margin value (distance between the two lines)
    const auxMargin = 2 * Math.abs(k);

    return {
      La: makeSeg(+1),
      Lb: makeSeg(-1),
      margin: auxMargin,
      theta: thetaPair * 180 / Math.PI
    };
  },[auxPair, pts]);

  // Utilities
  const deletePoint = (id:string)=> setPts(ps=> ps.filter(p=> p.id !== id));
  const resetAll = ()=> setPts(INIT.map(p=>({...p})));
  const restoreAll = ()=> setPts(ps=>ps.map(p=>({...p, active:true})));
  const autoPairClosest = ()=>{
    // pick closest opposite‑class active pair (Euclidean)
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
    if (!addMode.enabled) return;
    if (e.target !== e.currentTarget) return; // only empty canvas
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const w = toWorld(e.clientX-rect.left, e.clientY-rect.top);
    const id = nextId(addMode.label);
    setPts(ps => ps.concat({ id, x: clamp(w.x, WORLD.xmin, WORLD.xmax), y: clamp(w.y, WORLD.ymin, WORLD.ymax), label: addMode.label, active: true }));
  };

  // ---------- Render ----------
  return (
    <div className="w-full h-full p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold">Interactive SVM — Add/Remove points, new maximum margin + auxiliary margins</h1>
        <div className="flex items-center gap-2">
          <button onClick={restoreAll} className="px-3 py-1.5 rounded-xl bg-blue-100 text-blue-900 hover:bg-blue-200">Restore all</button>
          <button onClick={resetAll} className="px-3 py-1.5 rounded-xl bg-gray-100 hover:bg-gray-200">Reset</button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
        <div className="rounded-2xl shadow p-3 bg-white">
          <svg ref={svgRef} width={WIDTH} height={HEIGHT} className="rounded-xl border border-gray-200 bg-white select-none" onClick={handleSvgClick}>
            {/* Grid & axes */}
            {Array.from({length: WORLD.xmax - WORLD.xmin + 1}).map((_,i)=>{
              const x = WORLD.xmin + i; const a = toScreen(x, WORLD.ymin), bpt = toScreen(x, WORLD.ymax);
              return <line key={'vx'+x} x1={a.sx} y1={a.sy} x2={bpt.sx} y2={bpt.sy} stroke="#eef2f7" />;
            })}
            {Array.from({length: WORLD.ymax - WORLD.ymin + 1}).map((_,i)=>{
              const y = WORLD.ymin + i; const a = toScreen(WORLD.xmin, y), bpt = toScreen(WORLD.xmax, y);
              return <line key={'hz'+y} x1={a.sx} y1={a.sy} x2={bpt.sx} y2={bpt.sy} stroke="#eef2f7" />;
            })}
            <line x1={toScreen(WORLD.xmin,0).sx} y1={toScreen(WORLD.xmin,0).sy} x2={toScreen(WORLD.xmax,0).sx} y2={toScreen(WORLD.xmax,0).sy} stroke="#cbd5e1" />
            <line x1={toScreen(0,WORLD.ymin).sx} y1={toScreen(0,WORLD.ymin).sy} x2={toScreen(0,WORLD.ymax).sx} y2={toScreen(0,WORLD.ymax).sy} stroke="#cbd5e1" />

            {/* Auxiliary margins (red dashed) */}
            {auxLines && (
              <>
                <line x1={toScreen(auxLines.La.A.x, auxLines.La.A.y).sx} y1={toScreen(auxLines.La.A.x, auxLines.La.A.y).sy} x2={toScreen(auxLines.La.B.x, auxLines.La.B.y).sx} y2={toScreen(auxLines.La.B.x, auxLines.La.B.y).sy} stroke="#ef4444" strokeDasharray="6 6" strokeWidth={2} />
                <line x1={toScreen(auxLines.Lb.A.x, auxLines.Lb.A.y).sx} y1={toScreen(auxLines.Lb.A.x, auxLines.Lb.A.y).sy} x2={toScreen(auxLines.Lb.B.x, auxLines.Lb.B.y).sx} y2={toScreen(auxLines.Lb.B.x, auxLines.Lb.B.y).sy} stroke="#ef4444" strokeDasharray="6 6" strokeWidth={2} />
              </>
            )}

            {/* SVM margins & boundary */}
            <line x1={toScreen(H1.A.x,H1.A.y).sx} y1={toScreen(H1.A.x,H1.A.y).sy} x2={toScreen(H1.B.x,H1.B.y).sx} y2={toScreen(H1.B.x,H1.B.y).sy} stroke="#111827" strokeDasharray="6 6" strokeWidth={2} />
            <line x1={toScreen(H0.A.x,H0.A.y).sx} y1={toScreen(H0.A.x,H0.A.y).sy} x2={toScreen(H0.B.x,H0.B.y).sx} y2={toScreen(H0.B.x,H0.B.y).sy} stroke="#22c55e" strokeWidth={3} />
            <line x1={toScreen(H2.A.x,H2.A.y).sx} y1={toScreen(H2.A.x,H2.A.y).sy} x2={toScreen(H2.B.x,H2.B.y).sx} y2={toScreen(H2.B.x,H2.B.y).sy} stroke="#111827" strokeDasharray="6 6" strokeWidth={2} />

            {/* Points */}
            {pts.map(p=>{
              const s = toScreen(p.x,p.y); const active = p.active;
              const isSV = svIds.includes(p.id);
              const fill = p.label===1?"#2563eb":"#f97316";
              const stroke = !active?"#d1d5db": isSV?"#f59e0b":"#111827";
              const r = isSV?8:6;
              return (
                <g key={p.id}
                   onMouseDown={(e)=>{ setDragId(p.id); }}
                   onClick={(e)=>{
                     // If pair selection mode → record the pair; else handle add/delete/toggle
                     if (pairMode){
                       setAuxPair(prev=>{
                         if (!prev || (!prev.a && !prev.b)) return { a: p.label===1? p.id: undefined, b: p.label===-1? p.id: undefined };
                         if (prev.a && !prev.b && p.label===-1) return { a: prev.a, b: p.id };
                         if (prev.b && !prev.a && p.label===+1) return { a: p.id, b: prev.b };
                         if (prev.a && prev.b) return { a: p.label===1? p.id: prev.a, b: p.label===-1? p.id: prev.b };
                         return { a: p.label===1? p.id: undefined, b: p.label===-1? p.id: undefined };
                       });
                     } else if (e.shiftKey) {
                       deletePoint(p.id); // permanent delete
                     } else {
                       setPts(ps=>ps.map(q=> q.id===p.id?{...q, active:!q.active}:q)); // toggle
                     }
                   }}
                   style={{ cursor: "pointer" }}>
                  <circle cx={s.sx} cy={s.sy} r={r} fill={active?fill:"#f3f4f6"} stroke={stroke} strokeWidth={isSV?3:2} />
                  <text x={s.sx+10} y={s.sy-10} className="text-xs fill-gray-700 select-none">{p.id} ({p.x.toFixed(1)},{p.y.toFixed(1)})</text>
                </g>
              );
            })}

            {/* Labels */}
            <text x={WIDTH-PAD-260} y={PAD-18} className="text-sm fill-gray-700">marg. d = {(2*half).toFixed(3)}  |  θ = {(theta*180/Math.PI).toFixed(1)}°</text>
            <text x={PAD} y={PAD-18} className="text-sm fill-gray-700">
              {addMode.enabled ? `Add mode: click empty canvas to place a ${addMode.label===1?'+1':'−1'} point (Esc to cancel)` :
               auxPair?.a && auxPair?.b && auxLines ? `Aux pair: margin = ${auxLines.margin.toFixed(3)}, θ = ${auxLines.theta.toFixed(1)}°` :
               pairMode? "Click one +1 and one −1 to set aux pair" : "Click a point to toggle; Shift+Click to delete; drag to move"}
            </text>

            {/* Legend */}
            <g>
              <circle cx={PAD+10} cy={HEIGHT-PAD+6} r={6} fill="#2563eb" stroke="#111827" strokeWidth={2} />
              <text x={PAD+26} y={HEIGHT-PAD+10} className="text-xs fill-gray-700">+1</text>
              <circle cx={PAD+62} cy={HEIGHT-PAD+6} r={6} fill="#f97316" stroke="#111827" strokeWidth={2} />
              <text x={PAD+78} y={HEIGHT-PAD+10} className="text-xs fill-gray-700">−1</text>
              <circle cx={PAD+116} cy={HEIGHT-PAD+6} r={6} fill="#fff" stroke="#f59e0b" strokeWidth={3} />
              <text x={PAD+132} y={HEIGHT-PAD+10} className="text-xs fill-gray-700">support</text>
              <line x1={PAD+200} y1={HEIGHT-PAD+6} x2={PAD+240} y2={HEIGHT-PAD+6} stroke="#22c55e" strokeWidth={3}/>
              <text x={PAD+248} y={HEIGHT-PAD+10} className="text-xs fill-gray-700">H0</text>
              <line x1={PAD+280} y1={HEIGHT-PAD+6} x2={PAD+320} y2={HEIGHT-PAD+6} stroke="#111827" strokeDasharray="6 6" strokeWidth={2}/>
              <text x={PAD+328} y={HEIGHT-PAD+10} className="text-xs fill-gray-700">H1/H2</text>
              <line x1={PAD+372} y1={HEIGHT-PAD+6} x2={PAD+412} y2={HEIGHT-PAD+6} stroke="#ef4444" strokeDasharray="6 6" strokeWidth={2}/>
              <text x={PAD+420} y={HEIGHT-PAD+10} className="text-xs fill-gray-700">auxiliary</text>
            </g>
          </svg>
        </div>

        <div className="rounded-2xl shadow bg-white p-4 space-y-4 text-sm text-gray-700 leading-6">
          <h2 className="text-lg font-semibold">Controls</h2>
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              <button className={`px-3 py-1.5 rounded-xl ${pairMode? 'bg-rose-600 text-white':'bg-rose-100 text-rose-900 hover:bg-rose-200'}`} onClick={()=> setPairMode(m=>!m)}>
                {pairMode? 'Selecting… click +1 then −1':'Auxiliary: Select pair'}
              </button>
              <button className="px-3 py-1.5 rounded-xl bg-rose-100 text-rose-900 hover:bg-rose-200" onClick={autoPairClosest}>Auxiliary: Auto closest pair</button>
              <button className="px-3 py-1.5 rounded-xl bg-gray-100 hover:bg-gray-200" onClick={()=> setAuxPair(null)}>Clear aux</button>
            </div>
            <p className="text-xs text-gray-500">Auxiliary margins are perpendicular to the line joining the selected opposite‑class pair (matching the red dashed lines in the slides). They help visualize which third point must become a support.</p>
          </div>

          {/* Add / Remove */}
          <div className="space-y-3">
            <div className="font-medium">Add / Remove</div>
            <div className="flex flex-wrap gap-2">
              <button className={`px-3 py-1.5 rounded-xl ${addMode.enabled && addMode.label===1? 'bg-blue-600 text-white':'bg-blue-100 text-blue-900 hover:bg-blue-200'}`} onClick={()=> setAddMode({enabled:true,label:1})}>➕ Add +1 (click canvas)</button>
              <button className={`px-3 py-1.5 rounded-xl ${addMode.enabled && addMode.label===-1? 'bg-orange-600 text-white':'bg-orange-100 text-orange-900 hover:bg-orange-200'}`} onClick={()=> setAddMode({enabled:true,label:-1})}>➕ Add −1 (click canvas)</button>
              <button className="px-3 py-1.5 rounded-xl bg-gray-100 hover:bg-gray-200" onClick={()=> setAddMode(m=>({...m,enabled:false}))}>Cancel Add</button>
              <button className="px-3 py-1.5 rounded-xl bg-slate-200" onClick={()=> setPts(ps=> ps.length>0? ps.slice(0,-1): ps)}>Remove last</button>
            </div>
            <div className="text-xs text-gray-500">In add mode, click an empty spot on the canvas to place a point. Shift+Click any point to delete it permanently. Regular click toggles active status.</div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-50 rounded-xl p-3 border">
              <div className="text-xs text-slate-500">θ (deg)</div>
              <div className="text-lg font-mono">{(theta*180/Math.PI).toFixed(2)}</div>
            </div>
            <div className="bg-slate-50 rounded-xl p-3 border">
              <div className="text-xs text-slate-500">Half‑margin ρ</div>
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
        </div>
      </div>
    </div>
  );
}
