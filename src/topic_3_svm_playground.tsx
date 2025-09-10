import React, { useMemo, useState, useRef, useEffect, useCallback } from "react";

/**
 * Human SVM & KNN Playground — Visual & Interactive Demo (Topic 3)
 * ----------------------------------------------------------
 * Interactive SVM and KNN visualization with:
* 1) Linear SVM intuition (drag points, rotate/shift line, auto margins, SV highlighting)
 * 2) Kernel Hopscotch (ring → z = r^2 lift with threshold)
 * 3) k-NN Classification with multiple distance metrics
 * 4) Train/Test split and k-Fold CV
 * 5) Real-time metrics and effectiveness tracking
 */

// ---------- Utility math ----------
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

const world = {
  xMin: -10,
  xMax: 10,
  yMin: -7,
  yMax: 7,
};

function useSvgCoords(width: number, height: number) {
  const sx = width / (world.xMax - world.xMin);
  const sy = -height / (world.yMax - world.yMin); // invert y for SVG
  const tx = -world.xMin * sx;
  const ty = -world.yMax * sy;
  const toScreen = (x: number, y: number) => ({ x: x * sx + tx, y: y * sy + ty });
  const toWorld = (sxv: number, syv: number) => ({
    x: (sxv - tx) / sx,
    y: (syv - ty) / sy,
  });
  return { toScreen, toWorld };
}

// ---------- Types ----------
interface Pt { id: string; x: number; y: number; label: 1 | -1; r2?: number }
interface N2 { x: number; y: number }
interface TestPoint { x: number; y: number }

// k-NN Distance functions
function euclideanDistance(p1: TestPoint | Pt, p2: TestPoint | Pt): number {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2))
}

function manhattanDistance(p1: TestPoint | Pt, p2: TestPoint | Pt): number {
  return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y)
}

function cosineSimilarity(p1: TestPoint | Pt, p2: TestPoint | Pt): number {
  const dotProduct = p1.x * p2.x + p1.y * p2.y
  const magnitude1 = Math.sqrt(p1.x * p1.x + p1.y * p1.y)
  const magnitude2 = Math.sqrt(p2.x * p2.x + p2.y * p2.y)
  
  if (magnitude1 === 0 || magnitude2 === 0) return 0
  return dotProduct / (magnitude1 * magnitude2)
}

// ---------- Point + dataset helpers ----------
function randomPoints(n = 18): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 18; // ~[-9, 9]
    const y = (Math.random() - 0.5) * 12; // ~[-6, 6]
    const label = Math.random() < 0.5 ? -1 : 1;
    pts.push({ id: `p${i}`, x, y, label: label as 1 | -1 });
  }
  return pts;
}

function presetLayout(name: "clusters" | "parallel" | "overlap"): Pt[] {
  if (name === "parallel") {
    const pts: Pt[] = [];
    for (let i = 0; i < 9; i++) pts.push({ id: `a${i}`, x: -4 + Math.random() * 2, y: -5 + i, label: -1 });
    for (let i = 0; i < 9; i++) pts.push({ id: `b${i}`, x: 3 + Math.random() * 2, y: -5 + i, label: 1 });
    return pts;
  }
  if (name === "overlap") {
    const pts: Pt[] = [];
    for (let i = 0; i < 20; i++) {
      const x = (Math.random() - 0.5) * 10;
      const y = (Math.random() - 0.5) * 8;
      const lbl = x + y + (Math.random() - 0.5) * 2 > 0 ? 1 : -1;
      if (Math.random() < 0.25) pts.push({ id: `o${i}`, x, y, label: -lbl as 1 | -1 });
      else pts.push({ id: `o${i}`, x, y, label: lbl as 1 | -1 });
    }
    return pts;
  }
  // default: two clusters
  const pts: Pt[] = [];
  for (let i = 0; i < 10; i++) pts.push({ id: `c${i}`, x: -4 + Math.random() * 2, y: -2 + Math.random() * 3, label: -1 });
  for (let i = 0; i < 10; i++) pts.push({ id: `d${i}`, x: 4 + Math.random() * 2, y: -1 + Math.random() * 3, label: 1 });
  return pts;
}

// Signed distance to line n·x + b = 0, with ||n|| = 1
function signedDistance(point: Pt, n: N2, b: number) {
  const { x, y } = point;
  return n.x * x + n.y * y + b; // since n is unit, this is signed distance
}

// ---------- Metrics ----------
function computeMetrics(points: Pt[], n: N2, b: number) {
  let TP = 0, TN = 0, FP = 0, FN = 0, hingeSum = 0;
  points.forEach(p => {
    const d = signedDistance(p, n, b);
    const pred = d >= 0 ? 1 : -1;
    if (pred === 1 && p.label === 1) TP++;
    else if (pred === -1 && p.label === -1) TN++;
    else if (pred === 1 && p.label === -1) FP++;
    else if (pred === -1 && p.label === 1) FN++;
    const marginFunc = (p.label as number) * d; // functional margin
    hingeSum += Math.max(0, 1 - marginFunc);
  });
  const total = TP + TN + FP + FN;
  const acc = total ? (TP + TN) / total : 0;
  const prec = (TP + FP) ? TP / (TP + FP) : 0;
  const rec = (TP + FN) ? TP / (TP + FN) : 0;
  const f1 = (prec + rec) ? (2 * prec * rec) / (prec + rec) : 0;
  const avgHinge = total ? hingeSum / total : 0;
  return { TP, TN, FP, FN, acc, prec, rec, f1, avgHinge };
}

// Compute margin as the smallest absolute distance among correctly classified points on each side
function computeMarginAndSVs(points: Pt[], n: N2, b: number) {
  const bySide: { pos: Array<Pt & { d: number; correct: boolean }>; neg: Array<Pt & { d: number; correct: boolean }>; } = { pos: [], neg: [] };
  points.forEach((p) => {
    const d = signedDistance(p, n, b);
    const side = p.label === 1 ? "pos" : "neg";
    const correct = (d >= 0 && p.label === 1) || (d <= 0 && p.label === -1);
    (bySide as any)[side].push({ ...(p as any), d, correct });
  });

  const minPos = bySide.pos.filter(p => p.correct).reduce((m, p) => Math.min(m, Math.abs(p.d)), Infinity);
  const minNeg = bySide.neg.filter(p => p.correct).reduce((m, p) => Math.min(m, Math.abs(p.d)), Infinity);
  const marginHalf = Math.min(minPos, minNeg);

  // Identify SVs: those on or nearest to the margins on each side (within epsilon)
  const eps = 0.05;
  const SVs = points.filter(p => {
    const d = Math.abs(signedDistance(p, n, b));
    return Math.abs(d - marginHalf) <= eps;
  }).map(p => p.id);

  // Violations = inside margin or misclassified
  const violations = points.filter(p => {
    const d = signedDistance(p, n, b);
    const correct = (d >= 0 && p.label === 1) || (d <= 0 && p.label === -1);
    if (!correct) return true; // misclass
    return Math.abs(d) < marginHalf - 1e-6; // inside margin
  }).length;

  return { marginHalf: isFinite(marginHalf) ? marginHalf : 0, SVs, violations };
}

// ---------- Simple training (coarse grid-search auto-fit) ----------
function fitLinearSVM(data: Pt[], C = 1) {
  let best = { score: -Infinity, theta: 0, bias: 0 };
  const thetas = 64, biases = 49;
  for (let i = 0; i < thetas; i++) {
    const theta = (i / (thetas - 1)) * Math.PI; // [0, pi]
    const n = { x: Math.cos(theta), y: Math.sin(theta) };
    for (let j = 0; j < biases; j++) {
      const b = -6 + (j / (biases - 1)) * 12; // [-6, 6]
      const { marginHalf, violations } = computeMarginAndSVs(data, n, b);
      const { avgHinge } = computeMetrics(data, n, b);
      const obj = 2 * marginHalf - C * violations - 0.1 * avgHinge; // blended objective
      if (obj > best.score) best = { score: obj, theta, bias: b };
    }
  }
  return best;
}

// ---------- Perceptron Learning Algorithm ----------
function perceptronClassify(x1: number, x2: number, w0: number, w1: number, w2: number): 1 | -1 {
  const activation = w0 + w1 * x1 + w2 * x2;
  return activation >= 0 ? 1 : -1;
}

function trainPerceptron(
  data: Pt[], 
  alpha: number = 0.1, 
  maxIter: number = 100,
  onUpdate?: (iteration: number, weights: {w0: number; w1: number; w2: number}, errors: number) => void
): { weights: {w0: number; w1: number; w2: number}; history: Array<{iteration: number; weights: {w0: number; w1: number; w2: number}; errors: number}> } {
  // Initialize weights
  let w0 = Math.random() * 0.2 - 0.1; // bias term
  let w1 = Math.random() * 0.2 - 0.1; // weight for x1
  let w2 = Math.random() * 0.2 - 0.1; // weight for x2
  
  const history: Array<{iteration: number; weights: {w0: number; w1: number; w2: number}; errors: number}> = [];
  
  for (let iter = 0; iter < maxIter; iter++) {
    let errorCount = 0;
    
    // For each training instance
    for (let i = 0; i < data.length; i++) {
      const Xi = data[i];
      const Yi = Xi.label; // actual class
      const predicted = perceptronClassify(Xi.x, Xi.y, w0, w1, w2);
      
      // If misclassified, update weights
      if (predicted !== Yi) {
        errorCount++;
        // Update rule: wj = wj + α(Yi - h(Xi)) × xi,j
        const error = Yi - predicted; // Yi - h(Xi)
        w0 = w0 + alpha * error * 1; // bias term (x0 = 1)
        w1 = w1 + alpha * error * Xi.x; // weight for x1
        w2 = w2 + alpha * error * Xi.y; // weight for x2
      }
    }
    
    const weights = { w0, w1, w2 };
    history.push({ iteration: iter, weights: { ...weights }, errors: errorCount });
    
    if (onUpdate) {
      onUpdate(iter, weights, errorCount);
    }
    
    // Stop if no errors (converged)
    if (errorCount === 0) {
      break;
    }
  }
  
  return { weights: { w0, w1, w2 }, history };
}

function generatePerceptronSteps(
  data: Pt[], 
  alpha: number = 0.1, 
  maxIter: number = 100
): Array<{
  iteration: number;
  pointIndex: number;
  point: Pt;
  oldWeights: {w0: number; w1: number; w2: number};
  prediction: 1 | -1;
  actual: 1 | -1;
  isCorrect: boolean;
  newWeights: {w0: number; w1: number; w2: number};
  weightChange: {w0: number; w1: number; w2: number};
}> {
  const steps = [];
  
  // Initialize weights
  let w0 = Math.random() * 0.2 - 0.1;
  let w1 = Math.random() * 0.2 - 0.1;
  let w2 = Math.random() * 0.2 - 0.1;
  
  for (let iter = 0; iter < maxIter; iter++) {
    let errorCount = 0;
    
    for (let i = 0; i < data.length; i++) {
      const point = data[i];
      const oldWeights = { w0, w1, w2 };
      const prediction = perceptronClassify(point.x, point.y, w0, w1, w2);
      const actual = point.label;
      const isCorrect = prediction === actual;
      
      let newWeights = { ...oldWeights };
      let weightChange = { w0: 0, w1: 0, w2: 0 };
      
      if (!isCorrect) {
        errorCount++;
        const error = actual - prediction;
        const dw0 = alpha * error * 1;
        const dw1 = alpha * error * point.x;
        const dw2 = alpha * error * point.y;
        
        newWeights = {
          w0: w0 + dw0,
          w1: w1 + dw1,
          w2: w2 + dw2
        };
        
        weightChange = { w0: dw0, w1: dw1, w2: dw2 };
        
        w0 = newWeights.w0;
        w1 = newWeights.w1;
        w2 = newWeights.w2;
      }
      
      steps.push({
        iteration: iter,
        pointIndex: i,
        point,
        oldWeights,
        prediction,
        actual,
        isCorrect,
        newWeights,
        weightChange
      });
    }
    
    // Stop if no errors (converged)
    if (errorCount === 0) {
      break;
    }
  }
  
  return steps;
}

function splitTrainTest(points: Pt[], trainRatio = 0.7) {
  const shuffled = points.slice().sort(() => Math.random() - 0.5);
  const cut = Math.floor(shuffled.length * trainRatio);
  return { train: shuffled.slice(0, cut), test: shuffled.slice(cut) };
}

function kFold(points: Pt[], k = 5, C = 1) {
  const shuffled = points.slice().sort(() => Math.random() - 0.5);
  const foldSize = Math.floor(points.length / k) || 1;
  const results: Array<any> = [];
  for (let i = 0; i < k; i++) {
    const start = i * foldSize;
    const end = i === k - 1 ? shuffled.length : start + foldSize;
    const val = shuffled.slice(start, end);
    const train = shuffled.slice(0, start).concat(shuffled.slice(end));
    const fit = fitLinearSVM(train, C);
    const n = { x: Math.cos(fit.theta), y: Math.sin(fit.theta) };
    const m = computeMetrics(val, n, fit.bias);
    results.push({ ...m, theta: fit.theta, bias: fit.bias });
  }
  const avg = results.reduce((acc, r) => {
    acc.acc += r.acc; acc.prec += r.prec; acc.rec += r.rec; acc.f1 += r.f1; acc.avgHinge += r.avgHinge;
    return acc;
  }, { acc: 0, prec: 0, rec: 0, f1: 0, avgHinge: 0 });
  const nRes = results.length || 1;
  return {
    mean: { acc: avg.acc / nRes, prec: avg.prec / nRes, rec: avg.rec / nRes, f1: avg.f1 / nRes, avgHinge: avg.avgHinge / nRes },
    perFold: results,
  };
}

// ---------- Kernel Hopscotch dataset ----------
function makeRingData(n = 160): Pt[] {
  const innerR = 2.0;
  const outerR = 4.0;
  const pts: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const radius = innerR + (outerR - innerR) * Math.sqrt(Math.random());
    const x = radius * Math.cos(a);
    const y = radius * Math.sin(a);
    pts.push({ id: `r${i}`, x, y, r2: x * x + y * y, label: -1 }); // class A: ring
  }
  // outside class B
  for (let i = 0; i < n / 2; i++) {
    const a = Math.random() * Math.PI * 2;
    const radius = 5 + Math.random() * 3.5; // outside ring
    const x = radius * Math.cos(a);
    const y = radius * Math.sin(a);
    pts.push({ id: `o${i}`, x, y, r2: x * x + y * y, label: 1 });
  }
  // center class B
  for (let i = 0; i < n / 3; i++) {
    const x = (Math.random() - 0.5) * 2.0;
    const y = (Math.random() - 0.5) * 2.0;
    const r2 = x * x + y * y;
    pts.push({ id: `c${i}`, x, y, r2, label: 1 });
  }
  return pts;
}

// ---------- Main Component ----------
export default function SVMPlayground() {
  // LEFT panel state
  const [points, setPoints] = useState<Pt[]>(() => presetLayout("clusters"));
  const [theta, setTheta] = useState(0.4 * Math.PI); // angle for normal vector
  const [bias, setBias] = useState(0); // line shift along normal
  const [C, setC] = useState(1.0); // penalty weight (soft margin feel)
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [hoverId, setHoverId] = useState<string | null>(null);

  // RIGHT panel state (kernel view)
  const [ringData, setRingData] = useState<Pt[]>(() => makeRingData());
  const [zThresh, setZThresh] = useState(10.5); // threshold on r^2

  // Topic 3 Playground state
  const [trainRatio, setTrainRatio] = useState(0.7);
  const [k, setK] = useState(5);
  const [cv, setCv] = useState<any>(null);
  
  // k-NN state
  const [testPoint, setTestPoint] = useState<TestPoint>({ x: 0, y: 0 });
  const [kNN, setKNN] = useState(3);
  const [distanceMethod, setDistanceMethod] = useState<'euclidean' | 'manhattan' | 'cosine'>('euclidean');
  const [selectedClass, setSelectedClass] = useState<1 | -1>(1);
  const [showKNN, setShowKNN] = useState(false);

  // Perceptron state
  const [showPerceptron, setShowPerceptron] = useState(false);
  const [perceptronWeights, setPerceptronWeights] = useState<{ w0: number; w1: number; w2: number }>({ w0: 0, w1: 0, w2: 0 });
  const [learningRate, setLearningRate] = useState(0.1);
  const [maxIterations, setMaxIterations] = useState(100);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState<Array<{iteration: number; weights: {w0: number; w1: number; w2: number}; errors: number}>>([]);
  
  // Step-by-step training state
  const [stepByStepMode, setStepByStepMode] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [trainingSteps, setTrainingSteps] = useState<Array<{
    iteration: number;
    pointIndex: number;
    point: Pt;
    oldWeights: {w0: number; w1: number; w2: number};
    prediction: 1 | -1;
    actual: 1 | -1;
    isCorrect: boolean;
    newWeights: {w0: number; w1: number; w2: number};
    weightChange: {w0: number; w1: number; w2: number};
  }>>([]);
  const [autoPlaySpeed, setAutoPlaySpeed] = useState(1000); // ms between steps

  const svgRef = useRef<SVGSVGElement | null>(null);
  const width = 620;
  const height = 420;
  const { toScreen, toWorld } = useSvgCoords(width, height);
  
  // k-NN distance function
  const getDistance = useCallback((p1: TestPoint | Pt, p2: TestPoint | Pt) => {
    switch (distanceMethod) {
      case 'euclidean':
        return euclideanDistance(p1, p2)
      case 'manhattan':
        return manhattanDistance(p1, p2)
      case 'cosine':
        return 1 - cosineSimilarity(p1, p2) // Convert similarity to distance
      default:
        return euclideanDistance(p1, p2)
    }
  }, [distanceMethod])

  // k-NN classification
  const { nearestNeighbors, knnPrediction } = useMemo(() => {
    if (!showKNN) return { nearestNeighbors: [], knnPrediction: null }
    
    const distances = points.map(point => ({
      point,
      distance: getDistance(testPoint, point)
    }))
    
    distances.sort((a, b) => a.distance - b.distance)
    
    const kNearest = distances.slice(0, kNN)
    
    // Count votes for each class
    const votes: Record<number, number> = {}
    kNearest.forEach(({ point }) => {
      votes[point.label] = (votes[point.label] || 0) + 1
    })
    
    // Find the class with most votes
    const predictedClass = Object.entries(votes).reduce((a, b) => 
      votes[parseInt(a[0])] > votes[parseInt(b[0])] ? a : b
    )[0]
    
    return {
      nearestNeighbors: kNearest,
      knnPrediction: parseInt(predictedClass)
    }
  }, [points, testPoint, kNN, getDistance, showKNN])

  const n = useMemo<N2>(() => ({ x: Math.cos(theta), y: Math.sin(theta) }), [theta]);
  const { marginHalf, SVs, violations } = useMemo(() => computeMarginAndSVs(points, n, bias), [points, n, bias]);
  const metrics = useMemo(() => computeMetrics(points, n, bias), [points, n, bias]);

  // A simple score that rewards margin and penalizes violations with C
  const score = useMemo(() => 2 * marginHalf - C * violations, [marginHalf, C, violations]);

  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!draggingId) return;
      const svg = svgRef.current;
      if (!svg) return;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const m = svg.getScreenCTM();
      if (!m) return;
      const { x, y } = pt.matrixTransform(m.inverse());
      const w = toWorld(x, y);
      setPoints((ps) => ps.map(p => (p.id === draggingId ? { ...p, x: clamp(w.x, world.xMin, world.xMax), y: clamp(w.y, world.yMin, world.yMax) } : p)));
    }
    function onUp() { setDraggingId(null); }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [draggingId, toWorld]);

  // Auto-play effect for step-by-step training
  useEffect(() => {
    if (!stepByStepMode || !isTraining || currentStep >= trainingSteps.length) return;
    
    const timer = setTimeout(() => {
      const step = trainingSteps[currentStep];
      setPerceptronWeights(step.newWeights);
      setCurrentStep(prev => prev + 1);
      
      if (currentStep >= trainingSteps.length - 1) {
        setIsTraining(false);
      }
    }, autoPlaySpeed);
    
    return () => clearTimeout(timer);
  }, [stepByStepMode, isTraining, currentStep, trainingSteps, autoPlaySpeed]);

  const linePoints = useMemo(() => {
    // Compute two endpoints of the line across the viewport: n·x + b = 0
    // Direction vector t is perpendicular to n
    const t = { x: -n.y, y: n.x };
    // Find a point on the line: pick origin projected onto the line: x0 = -b * n
    const x0 = { x: -bias * n.x, y: -bias * n.y };
    // Extend far along t within world bounds
    const L = 100;
    const A = { x: x0.x - L * t.x, y: x0.y - L * t.y };
    const B = { x: x0.x + L * t.x, y: x0.y + L * t.y };
    return { A, B, t, x0 };
  }, [n, bias]);

  const marginLines = useMemo(() => {
    const { t, x0 } = linePoints;
    const L = 100;
    const up = { x: x0.x + (marginHalf) * n.x, y: x0.y + (marginHalf) * n.y };
    const down = { x: x0.x - (marginHalf) * n.x, y: x0.y - (marginHalf) * n.y };
    return {
      upA: { x: up.x - L * t.x, y: up.y - L * t.y },
      upB: { x: up.x + L * t.x, y: up.y + L * t.y },
      dnA: { x: down.x - L * t.x, y: down.y - L * t.y },
      dnB: { x: down.x + L * t.x, y: down.y + L * t.y },
    };
  }, [linePoints, marginHalf, n]);
  
  // Handle canvas click for both SVM and k-NN
  const handleCanvasClick = (e: React.MouseEvent<SVGElement>) => {
    const svg = e.currentTarget
    const rect = svg.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const worldCoords = toWorld(x, y)
    
    if (showKNN) {
      if (e.ctrlKey || e.metaKey) {
        // Add new training point in k-NN mode
        const newPoint: Pt = {
          id: Math.random().toString(36).slice(2),
          x: clamp(worldCoords.x, world.xMin, world.xMax),
          y: clamp(worldCoords.y, world.yMin, world.yMax),
          label: selectedClass
        }
        setPoints([...points, newPoint])
      } else {
        // Move test point in k-NN mode
        setTestPoint({ 
          x: clamp(worldCoords.x, world.xMin, world.xMax), 
          y: clamp(worldCoords.y, world.yMin, world.yMax) 
        })
      }
    }
  }
  

  // ---------- JSX ----------
  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 text-slate-900">
      <div className="max-w-7xl mx-auto">
        <header className="flex items-center justify-between gap-4 mb-6">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">ML Classification Playground</h1>
          <div className="flex items-center gap-4">
            <div className="text-sm md:text-base text-slate-600">Interactive ML algorithms • Drag points • Train models • Compare approaches</div>
            <div className="flex items-center gap-2">
              <button 
                onClick={() => {
                  setShowKNN(false);
                  setShowPerceptron(false);
                }} 
                className={`px-3 py-1.5 rounded-xl text-sm ${!showKNN && !showPerceptron ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-900'}`}
              >
                SVM Mode
              </button>
              <button 
                onClick={() => {
                  setShowKNN(true);
                  setShowPerceptron(false);
                  // Ensure we have data when switching to k-NN mode
                  if (points.length === 0) {
                    setPoints(presetLayout("clusters"))
                  }
                  // Ensure test point is in visible area
                  setTestPoint({ x: 0, y: 0 })
                }} 
                className={`px-3 py-1.5 rounded-xl text-sm ${showKNN && !showPerceptron ? 'bg-green-600 text-white' : 'bg-slate-200 text-slate-900'}`}
              >
                k-NN Mode
              </button>
              <button 
                onClick={() => {
                  setShowKNN(false);
                  setShowPerceptron(true);
                  // Ensure we have data when switching to Perceptron mode
                  if (points.length === 0) {
                    setPoints(presetLayout("clusters"))
                  }
                  // Reset perceptron weights
                  setPerceptronWeights({ w0: 0, w1: 0, w2: 0 });
                  setTrainingHistory([]);
                }} 
                className={`px-3 py-1.5 rounded-xl text-sm ${showPerceptron ? 'bg-purple-600 text-white' : 'bg-slate-200 text-slate-900'}`}
              >
                Perceptron Mode
              </button>
            </div>
          </div>
        </header>

        {/* Linear SVM intuition - Full width */}
        <div className="bg-white rounded-2xl shadow p-4 md:p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">
              {showPerceptron ? '1) Perceptron Learning Algorithm' : 
               showKNN ? '1) k-Nearest Neighbors Classification' : 
               '1) Max‑Margin & Soft‑Margin Intuition'}
            </h2>
            <div className="flex gap-2 flex-wrap">
              <button onClick={() => setPoints(randomPoints())} className="px-3 py-1.5 rounded-xl bg-slate-900 text-white text-sm shadow">Randomize</button>
              <button onClick={() => setPoints(presetLayout("clusters"))} className="px-3 py-1.5 rounded-xl bg-slate-200 text-slate-900 text-sm">Clusters</button>
              <button onClick={() => setPoints(presetLayout("parallel"))} className="px-3 py-1.5 rounded-xl bg-slate-200 text-slate-900 text-sm">Parallel</button>
              <button onClick={() => setPoints(presetLayout("overlap"))} className="px-3 py-1.5 rounded-xl bg-slate-200 text-slate-900 text-sm">Overlap</button>
              <button onClick={() => {
                // Reset to default cluster data and default parameters
                setPoints(presetLayout("clusters"))
                setTestPoint({ x: 0, y: 0 })
                setTheta(0.4 * Math.PI)
                setBias(0)
                setC(1.0)
                setKNN(3)
                setDistanceMethod('euclidean')
                setSelectedClass(1)
              }} className="px-3 py-1.5 rounded-xl bg-blue-600 text-white text-sm shadow">Reset</button>
              <button onClick={() => {
                // Specific example: points (1,2) class +1 and (3,4) class -1
                setPoints([
                  { id: "example1", x: 1, y: 2, label: 1 },
                  { id: "example2", x: 3, y: 4, label: -1 }
                ]);
                // Set expected values: w1 = w2 = -0.5, b = 2.5
                // θ = arctan(-0.5/-0.5) = arctan(1) = π/4 + π = 5π/4 (but normalized to [0,π])
                setTheta(Math.PI * 3/4); // 135 degrees
                setBias(2.5);
                console.log("📚 Set example: (1,2) class +1, (3,4) class -1");
                console.log("Expected: W=(-0.5,-0.5), b=2.5, margin=2√2≈2.828");
              }} className="px-3 py-1.5 rounded-xl bg-blue-600 text-white text-sm shadow"> Example</button>
              <button onClick={() => { const fit = fitLinearSVM(points, C); setTheta(fit.theta); setBias(fit.bias); }} className="px-3 py-1.5 rounded-xl bg-emerald-600 text-white text-sm shadow">Auto‑Fit</button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[1fr_350px] gap-6">
            {/* Canvas */}
            <div className="relative">
              <svg ref={svgRef} width={width} height={height} className="rounded-2xl border border-slate-200 bg-white" onClick={handleCanvasClick}>
                {/* Axes */}
                <line x1={toScreen(world.xMin,0).x} y1={toScreen(world.xMin,0).y} x2={toScreen(world.xMax,0).x} y2={toScreen(world.xMax,0).y} stroke="#e5e7eb" />
                <line x1={toScreen(0,world.yMin).x} y1={toScreen(0,world.yMin).y} x2={toScreen(0,world.yMax).x} y2={toScreen(0,world.yMax).y} stroke="#e5e7eb" />

                {/* Perceptron-specific elements */}
                {showPerceptron && (() => {
                  // Compute perceptron decision boundary: w0 + w1*x + w2*y = 0 => y = -(w0 + w1*x)/w2
                  const { w0, w1, w2 } = perceptronWeights;
                  if (Math.abs(w2) > 1e-9) {
                    const y1 = -(w0 + w1 * world.xMin) / w2;
                    const y2 = -(w0 + w1 * world.xMax) / w2;
                    const A = toScreen(world.xMin, y1);
                    const B = toScreen(world.xMax, y2);
                    return (
                      <>
                        <line x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke="#8b5cf6" strokeWidth={3} />
                        <text x={(A.x + B.x) / 2} y={(A.y + B.y) / 2 - 10} className="text-sm font-semibold fill-purple-600" textAnchor="middle">
                          Perceptron Boundary
                        </text>
                      </>
                    );
                  } else if (Math.abs(w1) > 1e-9) {
                    const x = -w0 / w1;
                    const A = toScreen(x, world.yMin);
                    const B = toScreen(x, world.yMax);
                    return (
                      <>
                        <line x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke="#8b5cf6" strokeWidth={3} />
                        <text x={A.x + 10} y={(A.y + B.y) / 2} className="text-sm font-semibold fill-purple-600">
                          Perceptron Boundary
                        </text>
                      </>
                    );
                  }
                  return null;
                })()}

                {/* SVM-specific elements - only show when not in k-NN or Perceptron mode */}
                {!showKNN && !showPerceptron && (
                  <>
                    {/* Margin lines */}
                    <line x1={toScreen(marginLines.upA.x, marginLines.upA.y).x} y1={toScreen(marginLines.upA.x, marginLines.upA.y).y} x2={toScreen(marginLines.upB.x, marginLines.upB.y).x} y2={toScreen(marginLines.upB.x, marginLines.upB.y).y} stroke="#34d399" strokeDasharray="6 6" strokeWidth={2} />
                    <line x1={toScreen(marginLines.dnA.x, marginLines.dnA.y).x} y1={toScreen(marginLines.dnA.x, marginLines.dnA.y).y} x2={toScreen(marginLines.dnB.x, marginLines.dnB.y).x} y2={toScreen(marginLines.dnB.x, marginLines.dnB.y).y} stroke="#34d399" strokeDasharray="6 6" strokeWidth={2} />

                    {/* H1 and H2 Labels */}
                    <text x={toScreen((linePoints as any).x0.x + (marginHalf) * n.x - 1, (linePoints as any).x0.y + (marginHalf) * n.y).x} y={toScreen((linePoints as any).x0.x + (marginHalf) * n.x - 1, (linePoints as any).x0.y + (marginHalf) * n.y).y - 8} className="text-sm font-semibold fill-emerald-600" textAnchor="middle">H₁</text>
                    <text x={toScreen((linePoints as any).x0.x - (marginHalf) * n.x - 1, (linePoints as any).x0.y - (marginHalf) * n.y).x} y={toScreen((linePoints as any).x0.x - (marginHalf) * n.x - 1, (linePoints as any).x0.y - (marginHalf) * n.y).y - 8} className="text-sm font-semibold fill-emerald-600" textAnchor="middle">H₂</text>

                    {/* Decision boundary */}
                    <line x1={toScreen((linePoints as any).A.x, (linePoints as any).A.y).x} y1={toScreen((linePoints as any).A.x, (linePoints as any).A.y).y} x2={toScreen((linePoints as any).B.x, (linePoints as any).B.y).x} y2={toScreen((linePoints as any).B.x, (linePoints as any).B.y).y} stroke="#111827" strokeWidth={3} />

                    {/* H0 Label */}
                    <text x={toScreen((linePoints as any).x0.x - 1, (linePoints as any).x0.y).x} y={toScreen((linePoints as any).x0.x - 1, (linePoints as any).x0.y).y - 8} className="text-sm font-semibold fill-gray-800" textAnchor="middle">H₀</text>

                    {/* Normal vector arrow */}
                    {(() => {
                      const start = toScreen((linePoints as any).x0.x, (linePoints as any).x0.y);
                      const end = toScreen((linePoints as any).x0.x + n.x * 2, (linePoints as any).x0.y + n.y * 2);
                      const angDeg = (Math.atan2(n.y, n.x) * 180 / Math.PI) + 90;
                      const pointsAttr = `${end.x},${end.y} ${end.x - 6},${end.y - 6} ${end.x + 6},${end.y - 6}`;
                      const transformAttr = `rotate(${angDeg}, ${end.x}, ${end.y})`;
                      return (
                        <g>
                          <line x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke="#111827" strokeWidth={2} />
                          <polygon points={pointsAttr} fill="#111827" transform={transformAttr} />
                        </g>
                      );
                    })()}
                  </>
                )}

                {/* k-NN visualization elements */}
                {showKNN && nearestNeighbors.map(({ point, distance }, i) => (
                  <circle
                    key={`circle-${i}`}
                    cx={toScreen(testPoint.x, testPoint.y).x}
                    cy={toScreen(testPoint.x, testPoint.y).y}
                    r={distanceMethod === 'cosine' ? distance * 50 : Math.min(distance * 30, 200)}
                    fill="none"
                    stroke={point.label === 1 ? "#2563eb" : "#f97316"}
                    strokeWidth="1"
                    strokeOpacity="0.3"
                    strokeDasharray="5,5"
                  />
                ))}
                
                {/* k-NN test point */}
                {showKNN && (
                  <>
                    <circle
                      cx={toScreen(testPoint.x, testPoint.y).x}
                      cy={toScreen(testPoint.x, testPoint.y).y}
                      r="10"
                      fill={knnPrediction === 1 ? "#2563eb" : knnPrediction === -1 ? "#f97316" : "#6b7280"}
                      stroke="#000"
                      strokeWidth="3"
                    />
                    <text
                      x={toScreen(testPoint.x, testPoint.y).x}
                      y={toScreen(testPoint.x, testPoint.y).y - 15}
                      textAnchor="middle"
                      className="text-sm font-bold fill-gray-800"
                    >
                      Test Point
                    </text>
                  </>
                )}

                {/* Data points */}
                {points.map((p) => {
                  // Check if this is the current point being processed in step-by-step mode
                  const isCurrentStep = showPerceptron && stepByStepMode && trainingSteps.length > 0 && currentStep < trainingSteps.length && 
                                       trainingSteps[currentStep]?.point.id === p.id;
                  
                  return renderPoint(p, { 
                    toScreen, 
                    n, 
                    bias, 
                    marginHalf, 
                    SVs, 
                    setHoverId, 
                    setDraggingId, 
                    hoverId, 
                    showKNN, 
                    nearestNeighbors,
                    isCurrentPerceptronStep: isCurrentStep
                  });
                })}
              </svg>

              {/* Legend */}
              <div className="absolute top-2 left-2 bg-white/80 backdrop-blur rounded-xl px-3 py-2 text-xs shadow flex items-center gap-3 flex-wrap">
                <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{ background: "#2563eb" }}></span> Class +1</div>
                <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{ background: "#f97316" }}></span> Class −1</div>
                {!showKNN && !showPerceptron && (
                  <>
                    <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full border-2" style={{ borderColor: "#10b981" }}></span> Support Vectors</div>
                    <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full border-2" style={{ borderColor: "#f59e0b" }}></span> Inside Margin</div>
                    <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full border-2" style={{ borderColor: "#ef4444" }}></span> Misclassified</div>
                  </>
                )}
                {showKNN && (
                  <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full border-2" style={{ borderColor: "#10b981" }}></span> k-Nearest</div>
                )}
                {showPerceptron && (
                  <div className="flex items-center gap-1"><span className="inline-block w-4 h-1" style={{ background: "#8b5cf6" }}></span> Decision Boundary</div>
                )}
              </div>
            {/* MetricsCard positioned directly under plot */}
            <div className="mt-4">
              <MetricsCard metrics={metrics} />
            </div>
            </div>

            {/* Controls + Readouts */}
            <div className="space-y-4">
              {/* Perceptron Controls */}
              {showPerceptron && (
                <>
                  <div className="bg-slate-50 rounded-2xl p-4 border">
                    <h3 className="font-semibold mb-2">Perceptron Controls</h3>
                    
                    <div className="space-y-3">
                      <div className="grid grid-cols-3 gap-3">
                        <div className="text-center">
                          <label className="text-xs font-medium">w₀ (bias)</label>
                          <div className="text-lg font-mono">{perceptronWeights.w0.toFixed(3)}</div>
                        </div>
                        <div className="text-center">
                          <label className="text-xs font-medium">w₁</label>
                          <div className="text-lg font-mono">{perceptronWeights.w1.toFixed(3)}</div>
                        </div>
                        <div className="text-center">
                          <label className="text-xs font-medium">w₂</label>
                          <div className="text-lg font-mono">{perceptronWeights.w2.toFixed(3)}</div>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Learning Rate (α):</label>
                        <input
                          type="number"
                          min="0.01"
                          max="1"
                          step="0.01"
                          value={learningRate}
                          onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.1)}
                          className="w-full p-2 border rounded"
                        />
                      </div>

                      <div className="space-y-2">
                        <label className="text-sm font-medium">Max Iterations:</label>
                        <input
                          type="number"
                          min="10"
                          max="1000"
                          step="10"
                          value={maxIterations}
                          onChange={(e) => setMaxIterations(parseInt(e.target.value) || 100)}
                          className="w-full p-2 border rounded"
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center gap-2 mb-2">
                          <label className="text-sm font-medium">Training Mode:</label>
                          <select
                            value={stepByStepMode ? 'step' : 'auto'}
                            onChange={(e) => setStepByStepMode(e.target.value === 'step')}
                            className="flex-1 p-1 border rounded text-sm"
                          >
                            <option value="auto">Auto (Batch)</option>
                            <option value="step">Step-by-Step</option>
                          </select>
                        </div>

                        {stepByStepMode && (
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Auto-play Speed:</label>
                            <input
                              type="range"
                              min="200"
                              max="2000"
                              step="100"
                              value={autoPlaySpeed}
                              onChange={(e) => setAutoPlaySpeed(parseInt(e.target.value))}
                              className="w-full"
                            />
                            <div className="text-xs text-center">{autoPlaySpeed}ms per step</div>
                          </div>
                        )}

                        <button
                          onClick={() => {
                            if (points.length === 0) return;
                            setIsTraining(true);
                            setTrainingHistory([]);
                            setCurrentStep(0);
                            
                            if (stepByStepMode) {
                              // Generate all steps for step-by-step mode
                              const steps = generatePerceptronSteps(points, learningRate, maxIterations);
                              setTrainingSteps(steps);
                              if (steps.length > 0) {
                                setPerceptronWeights(steps[0].oldWeights);
                              }
                            } else {
                              // Regular batch training
                              const result = trainPerceptron(points, learningRate, maxIterations, (iter, weights, errors) => {
                                setTimeout(() => {
                                  setPerceptronWeights(weights);
                                  setTrainingHistory(prev => [...prev, {iteration: iter, weights: {...weights}, errors}]);
                                }, iter * 50);
                              });
                              
                              setTimeout(() => {
                                setPerceptronWeights(result.weights);
                                setIsTraining(false);
                              }, result.history.length * 50);
                            }
                          }}
                          disabled={isTraining || points.length === 0}
                          className="w-full px-3 py-2 rounded-xl bg-purple-600 text-white text-sm shadow hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                        >
                          {isTraining ? 'Training...' : stepByStepMode ? 'Start Step-by-Step' : 'Train Perceptron'}
                        </button>

                        {stepByStepMode && trainingSteps.length > 0 && (
                          <div className="flex gap-2">
                            <button
                              onClick={() => {
                                if (currentStep > 0) {
                                  const newStep = currentStep - 1;
                                  setCurrentStep(newStep);
                                  setPerceptronWeights(trainingSteps[newStep].newWeights);
                                }
                              }}
                              disabled={currentStep === 0}
                              className="flex-1 px-2 py-1 rounded text-sm bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed"
                            >
                              ← Prev
                            </button>
                            
                            <button
                              onClick={() => {
                                setIsTraining(!isTraining);
                              }}
                              disabled={currentStep >= trainingSteps.length}
                              className="flex-1 px-2 py-1 rounded text-sm bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-400"
                            >
                              {isTraining ? 'Pause' : 'Play'}
                            </button>
                            
                            <button
                              onClick={() => {
                                if (currentStep < trainingSteps.length - 1) {
                                  const newStep = currentStep + 1;
                                  setCurrentStep(newStep);
                                  setPerceptronWeights(trainingSteps[newStep].newWeights);
                                }
                              }}
                              disabled={currentStep >= trainingSteps.length - 1}
                              className="flex-1 px-2 py-1 rounded text-sm bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed"
                            >
                              Next →
                            </button>
                          </div>
                        )}
                      </div>
                      
                      <button
                        onClick={() => {
                          setPerceptronWeights({ w0: Math.random() * 0.2 - 0.1, w1: Math.random() * 0.2 - 0.1, w2: Math.random() * 0.2 - 0.1 });
                          setTrainingHistory([]);
                        }}
                        className="w-full px-3 py-2 rounded-xl bg-slate-500 text-white text-sm shadow hover:bg-slate-600"
                      >
                        Reset Weights
                      </button>
                    </div>

                    <div className="mt-4 p-3 bg-purple-50 rounded-xl border border-purple-200">
                      <h4 className="text-sm font-medium text-purple-800 mb-2">Perceptron Algorithm</h4>
                      <div className="text-xs text-purple-700 space-y-1 font-mono">
                        <div><strong>Input:</strong> T training instances (X₀, Y₀), (X₁, Y₁), ..., (Xₜ₋₁, Yₜ₋₁)</div>
                        <div><strong>Where:</strong> Xᵢ = ⟨xᵢ,₀, xᵢ,₁, ..., xᵢ,ₙ₋₁⟩ (N input features)</div>
                        <div><strong>Output:</strong> Decision boundary hyperplane W = ⟨w₀, w₁, ..., wₙ₋₁⟩</div>
                        <div className="mt-2"><strong>Algorithm:</strong></div>
                        <div>1. Initialize W</div>
                        <div>2. Do:</div>
                        <div className="ml-4">For i = 0 to T-1:</div>
                        <div className="ml-8">For j = 0 to N-1:</div>
                        <div className="ml-12">wⱼ = wⱼ + α(Yᵢ - h(Xᵢ)) × xᵢ,ⱼ</div>
                        <div>3. Until min classification error</div>
                      </div>
                    </div>
                    
                    {/* Step-by-Step Details */}
                    {stepByStepMode && trainingSteps.length > 0 && currentStep < trainingSteps.length && (
                      <div className="mt-4 p-3 bg-purple-50 rounded-xl border border-purple-200">
                        <div className="text-sm font-medium text-purple-800 mb-2">
                          Step {currentStep + 1} of {trainingSteps.length}
                        </div>
                        <div className="text-xs text-purple-700 space-y-1">
                          {(() => {
                            const step = trainingSteps[currentStep];
                            return (
                              <>
                                <div><strong>Iteration:</strong> {step.iteration + 1}</div>
                                <div><strong>Point:</strong> ({step.point.x.toFixed(2)}, {step.point.y.toFixed(2)})</div>
                                <div><strong>Actual y:</strong> {step.actual}</div>
                                <div><strong>Predicted ŷ:</strong> {step.prediction}</div>
                                <div className={`${step.isCorrect ? 'text-green-700' : 'text-red-700'} font-medium`}>
                                  {step.isCorrect ? '✓ Correct' : '✗ Misclassified'}
                                </div>
                                {!step.isCorrect && (
                                  <>
                                    <div className="mt-1 text-xs">
                                      <strong>Weight Updates:</strong>
                                    </div>
                                    <div className="ml-2 font-mono text-xs">
                                      <div>Δw₀ = {step.weightChange.w0.toFixed(3)}</div>
                                      <div>Δw₁ = {step.weightChange.w1.toFixed(3)}</div>
                                      <div>Δw₂ = {step.weightChange.w2.toFixed(3)}</div>
                                    </div>
                                    <div className="mt-1 text-xs">
                                      <strong>New Weights:</strong>
                                    </div>
                                    <div className="ml-2 font-mono text-xs">
                                      <div>w₀ = {step.newWeights.w0.toFixed(3)}</div>
                                      <div>w₁ = {step.newWeights.w1.toFixed(3)}</div>
                                      <div>w₂ = {step.newWeights.w2.toFixed(3)}</div>
                                    </div>
                                  </>
                                )}
                              </>
                            );
                          })()}
                        </div>
                      </div>
                    )}

                    {/* Regular Training Progress */}
                    {!stepByStepMode && trainingHistory.length > 0 && (
                      <div className="mt-4 p-3 bg-blue-50 rounded-xl border border-blue-200">
                        <div className="text-sm font-medium text-blue-800 mb-2">Training Progress</div>
                        <div className="text-xs text-blue-700">
                          <div>Iteration: {trainingHistory[trainingHistory.length - 1]?.iteration + 1}</div>
                          <div>Errors: {trainingHistory[trainingHistory.length - 1]?.errors}</div>
                          {trainingHistory[trainingHistory.length - 1]?.errors === 0 && (
                            <div className="text-green-700 font-medium">✓ Converged!</div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Perceptron Algorithm Explanation - Added to fill space */}
                  <div className="bg-purple-50 rounded-2xl p-4 border border-purple-200">
                    <h4 className="text-sm font-medium text-purple-800 mb-2">How Perceptron Works</h4>
                    <div className="text-xs text-purple-700 space-y-2">
                      <div>
                        <strong>1. Initialization:</strong> Start with random weights w₀, w₁, w₂
                      </div>
                      <div>
                        <strong>2. For each point:</strong> 
                        <div className="ml-2 mt-1">
                          • Compute activation: a = w₀ + w₁x + w₂y<br/>
                          • Predict: ŷ = 1 if a ≥ 0, else ŷ = -1<br/>
                          • If wrong: update weights by α(y-ŷ)x
                        </div>
                      </div>
                      <div>
                        <strong>3. Convergence:</strong> Repeat until no misclassifications
                      </div>
                      <div className="mt-2 p-2 bg-white rounded text-xs">
                        <strong>Key Insight:</strong> The perceptron finds <em>any</em> separating line if one exists, unlike SVM which finds the <em>optimal</em> one with maximum margin.
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* k-NN Controls */}
              {showKNN && (
                <>
                  <div className="bg-slate-50 rounded-2xl p-4 border">
                    <h3 className="font-semibold mb-2">k-NN Controls</h3>
                    
                    <div className="space-y-2 mb-3">
                      <label className="text-sm font-medium">k value:</label>
                      <input
                        type="number"
                        min="1"
                        max={points.length || 1}
                        value={kNN}
                        onChange={(e) => setKNN(Math.max(1, parseInt(e.target.value) || 1))}
                        className="w-full p-2 border rounded"
                      />
                    </div>

                    <div className="space-y-2 mb-3">
                      <label className="text-sm font-medium">Distance Method:</label>
                      <select
                        value={distanceMethod}
                        onChange={(e) => setDistanceMethod(e.target.value as any)}
                        className="w-full p-2 border rounded"
                      >
                        <option value="euclidean">Euclidean Distance</option>
                        <option value="manhattan">Manhattan Distance</option>
                        <option value="cosine">Cosine Similarity</option>
                      </select>
                    </div>

                    <div className="space-y-2 mb-3">
                      <label className="text-sm font-medium">New Point Class:</label>
                      <select
                        value={selectedClass}
                        onChange={(e) => setSelectedClass(parseInt(e.target.value) as 1 | -1)}
                        className="w-full p-2 border rounded"
                      >
                        <option value={1}>Class +1</option>
                        <option value={-1}>Class -1</option>
                      </select>
                    </div>

                    <div className="text-sm text-gray-600 space-y-1">
                      <p><strong>Instructions:</strong></p>
                      <p>• Click to move test point (black)</p>
                      <p>• Ctrl/Cmd + Click to add training points</p>
                      <p>• Change k and distance method to see different results</p>
                    </div>
                    
                    {knnPrediction !== null && (
                      <div className="mt-3 p-3 bg-blue-50 rounded-xl border border-blue-200">
                        <div className="text-sm font-medium text-blue-800 mb-1">
                          k-NN Prediction: <span className={`font-bold ${knnPrediction === 1 ? 'text-blue-600' : 'text-orange-600'}`}>
                            Class {knnPrediction === 1 ? '+1' : '-1'}
                          </span>
                        </div>
                        <div className="text-xs text-blue-700">
                          Using {distanceMethod} distance with k={kNN}
                        </div>
                        {nearestNeighbors.length > 0 && (
                          <div className="mt-2">
                            <div className="text-xs font-medium text-blue-800 mb-1">Nearest {kNN} Neighbors:</div>
                            <div className="space-y-1 max-h-24 overflow-y-auto">
                              {nearestNeighbors.map(({ point, distance }, i) => (
                                <div key={i} className="flex justify-between items-center text-xs">
                                  <span className={point.label === 1 ? 'text-blue-600' : 'text-orange-600'}>
                                    Class {point.label === 1 ? '+1' : '-1'}
                                  </span>
                                  <span className="text-gray-600">
                                    d = {distance.toFixed(2)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}

              {/* SVM-specific controls - only show in SVM mode */}
              {!showKNN && !showPerceptron && (
                <>
                  <div className="bg-slate-50 rounded-2xl p-4 border">
                    <h3 className="font-semibold mb-2">Decision Boundary</h3>
                    <label className="block text-sm mb-1">Rotate (θ)</label>
                    <input type="range" min={0} max={Math.PI} step={0.01} value={theta} onChange={e => setTheta(parseFloat(e.target.value))} className="w-full" />
                    <div className="text-xs text-slate-600 mb-2">θ = {theta.toFixed(2)} rad</div>

                    <label className="block text-sm mb-1">Shift (bias b)</label>
                    <input type="range" min={-6} max={6} step={0.02} value={bias} onChange={e => setBias(parseFloat(e.target.value))} className="w-full" />
                    <div className="text-xs text-slate-600">b = {bias.toFixed(2)} (moves line along its normal)</div>
                  </div>

                  <div className="bg-slate-50 rounded-2xl p-4 border">
                    <h3 className="font-semibold mb-2">Soft‑Margin Feel</h3>
                    <label className="block text-sm mb-1">Penalty (C)</label>
                    <input type="range" min={0} max={4} step={0.1} value={C} onChange={e => setC(parseFloat(e.target.value))} className="w-full" />
                    <div className="text-xs text-slate-600 mb-3">Small C → tolerate violations (wider margin). Large C → punish violations.</div>

                    <div className="grid grid-cols-3 gap-2 text-center">
                      <SmallStat label="Half‑Margin" value={marginHalf.toFixed(2)} />
                      <SmallStat label="Violations" value={String(violations)} />
                      <SmallStat label="Score" value={score.toFixed(2)} tone={score >= 0 ? "pos" : "neg"} />
                    </div>
                  </div>
                </>
              )}

              {/* Data Editing - available in all modes */}
              <div className="bg-slate-50 rounded-2xl p-4 border">
                <h3 className="font-semibold mb-2">Data Editing</h3>
                <div className="flex gap-2 flex-wrap">
                  <button onClick={() => setPoints(ps => ps.concat({ id: Math.random().toString(36).slice(2), x: 0, y: 0, label: 1 }))} className="px-3 py-1.5 rounded-xl bg-blue-600 text-white text-sm shadow">Add +1 at (0,0)</button>
                  <button onClick={() => setPoints(ps => ps.concat({ id: Math.random().toString(36).slice(2), x: 0, y: 0, label: -1 }))} className="px-3 py-1.5 rounded-xl bg-orange-500 text-white text-sm shadow">Add −1 at (0,0)</button>
                </div>
                <p className="text-xs text-slate-600 mt-2">Tip: drag a point; hover to see coordinates{!showKNN && !showPerceptron ? '; widen the green dashed margins while reducing violations' : ''}.</p>
              </div>

            </div>
          </div>

          {/* Train/Test & Cross-Validation - Moved here to reduce white space */}
          <div className="bg-slate-50 rounded-2xl p-4 border mt-6">
            <h3 className="text-xl font-semibold mb-3">Train/Test Split & Cross-Validation</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-white rounded-xl p-4 border">
                  <h4 className="font-semibold mb-2">Train/Test Split</h4>
                  <div className="flex items-center gap-4">
                    <label className="text-sm">Train %</label>
                    <input type="range" min={0.5} max={0.9} step={0.05} value={trainRatio} onChange={e => setTrainRatio(parseFloat(e.target.value))} className="flex-1" />
                    <div className="text-sm w-16 text-right">{Math.round(trainRatio * 100)}%</div>
                  </div>
                  <div className="mt-3 flex gap-2 flex-wrap">
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        console.log('🎯 Auto-Fit button clicked, points:', points.length, 'trainRatio:', trainRatio);
                        try {
                          if (points.length === 0) {
                            console.warn('No points to train on');
                            return;
                          }
                          const { train, test } = splitTrainTest(points, trainRatio);
                          console.log(`📊 Split: ${train.length} train, ${test.length} test`);

                          if (train.length === 0) {
                            console.warn('Train set is empty');
                            return;
                          }

                          const fit = fitLinearSVM(train, C);
                          console.log('🔧 Fit result:', { theta: fit.theta, bias: fit.bias, score: fit.score });

                          setTheta(fit.theta);
                          setBias(fit.bias);

                          console.log(`✅ Trained on ${train.length} points: θ=${fit.theta.toFixed(2)}, b=${fit.bias.toFixed(2)}`);

                          // Compute metrics on test set for feedback
                          if (test.length > 0) {
                            const n = { x: Math.cos(fit.theta), y: Math.sin(fit.theta) };
                            const testMetrics = computeMetrics(test, n, fit.bias);
                            console.log(`📈 Test accuracy: ${(testMetrics.acc * 100).toFixed(1)}%`);
                          }
                        } catch (error) {
                          console.error('❌ Error in Auto-Fit:', error);
                        }
                      }}
                      className="px-3 py-1.5 rounded-xl bg-emerald-600 text-white text-sm shadow hover:bg-emerald-700 active:bg-emerald-800 transition-colors duration-150 cursor-pointer select-none"
                      title="Train SVM on training subset and update the decision boundary"
                    >
                      🎯 Auto‑Fit on Train → Set Line
                    </button>

                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        console.log('🔀 Reshuffle button clicked, current points:', points.length);
                        try {
                          if (points.length === 0) {
                            console.warn('No points to shuffle');
                            return;
                          }

                          // Fisher-Yates shuffle for proper randomization
                          const shuffled = [...points];
                          for (let i = shuffled.length - 1; i > 0; i--) {
                            const j = Math.floor(Math.random() * (i + 1));
                            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
                          }

                          setPoints(shuffled);
                          console.log('✅ Points reshuffled successfully');
                        } catch (error) {
                          console.error('❌ Error in Reshuffle:', error);
                        }
                      }}
                      className="px-3 py-1.5 rounded-xl bg-slate-200 text-slate-900 text-sm hover:bg-slate-300 active:bg-slate-400 transition-colors duration-150 cursor-pointer select-none"
                      title="Randomly reorder the same set of points for different train/test splits"
                    >
                      🔀 Reshuffle (same set)
                    </button>
                  </div>
                  <p className="text-xs text-slate-600 mt-2">We fit θ and b using only the training subset (grid search), then evaluate with the live metrics.</p>
                </div>

                <div className="bg-white rounded-xl p-4 border">
                  <h4 className="font-semibold mb-2">k‑Fold Cross‑Validation</h4>
                  <div className="flex items-center gap-4">
                    <label className="text-sm">k</label>
                    <input type="range" min={3} max={8} step={1} value={k} onChange={e => setK(parseInt(e.target.value))} className="flex-1" />
                    <div className="text-sm w-10 text-right">{k}</div>
                    <button onClick={() => setCv(kFold(points, k, C))} className="px-3 py-1.5 rounded-xl bg-indigo-600 text-white text-sm shadow">Run k‑Fold</button>
                  </div>
                  {cv && (
                    <div className="grid grid-cols-5 gap-2 text-center mt-3">
                      <SmallStat label="Mean Acc" value={`${(cv.mean.acc * 100).toFixed(1)}%`} />
                      <SmallStat label="Mean Prec" value={`${(cv.mean.prec * 100).toFixed(1)}%`} />
                      <SmallStat label="Mean Recall" value={`${(cv.mean.rec * 100).toFixed(1)}%`} />
                      <SmallStat label="Mean F1" value={`${(cv.mean.f1 * 100).toFixed(1)}%`} />
                      <SmallStat label="Mean Hinge" value={cv.mean.avgHinge.toFixed(2)} />
                    </div>
                  )}
                  <p className="text-xs text-slate-600 mt-2">Rotate folds, train on k−1, validate on the held‑out fold, then average.</p>
                </div>
              </div>

              <div className="bg-white rounded-xl p-4 border h-full">
                <h4 className="font-semibold mb-2">Why these metrics?</h4>
                <ul className="list-disc pl-5 space-y-1 text-sm text-slate-700">
                  <li><span className="font-medium">Accuracy</span>: overall correctness (can mislead on class imbalance).</li>
                  <li><span className="font-medium">Precision</span>: of predicted +1, how many were truly +1?</li>
                  <li><span className="font-medium">Recall</span>: of actual +1, how many did we catch?</li>
                  <li><span className="font-medium">F1</span>: harmonic mean of precision & recall.</li>
                  <li><span className="font-medium">Hinge loss</span>: penalizes errors and low‑margin correct points; lower is better.</li>
                </ul>
                <p className="text-xs text-slate-600 mt-2">Tune C and refit to see generalization change across folds.</p>
              </div>
            </div>
          </div>

          {/* Additional Analysis Sections - Moved below canvas to reduce white space */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4 mt-6">
            {/* H0, H1, H2 Planes Analysis - Only show for SVM */}
            {!showKNN && !showPerceptron && (
              <div className="bg-slate-50 rounded-2xl p-4 border lg:col-span-2">
                <h3 className="font-semibold mb-2">H0, H1, H2 Plane Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="font-medium text-slate-700 mb-1">SVM Parameters:</div>
                    <div className="bg-white rounded-xl p-3 border font-mono text-xs space-y-1">
                      <div><strong>Weight Vector:</strong> 𝐖 = ({n.x.toFixed(3)}, {n.y.toFixed(3)})</div>
                      <div><strong>Norm:</strong> ‖𝐖‖ = {Math.sqrt(n.x * n.x + n.y * n.y).toFixed(3)}</div>
                      <div><strong>Bias:</strong> b = {bias.toFixed(3)}</div>
                      <div><strong>Penalty:</strong> C = {C.toFixed(2)}</div>
                      <div><strong>Full Margin:</strong> d = {(2 * marginHalf).toFixed(3)}</div>
                      <div><strong>Half Margin:</strong> ρ = {marginHalf.toFixed(3)}</div>
                      <div><strong>Angle θ:</strong> {theta.toFixed(3)} rad ({(theta * 180 / Math.PI).toFixed(1)}°)</div>
                    </div>
                  </div>

                  <div>
                    <div className="font-medium text-slate-700 mb-1">Plane Equations:</div>
                    <div className="bg-white rounded-xl p-3 border font-mono text-xs space-y-1">
                      <div><span className="text-slate-600">H0:</span> {n.x.toFixed(3)}x₁ + {n.y.toFixed(3)}x₂ + {bias.toFixed(3)} = 0</div>
                      <div><span className="text-emerald-600">H1:</span> {n.x.toFixed(3)}x₁ + {n.y.toFixed(3)}x₂ + {bias.toFixed(3)} = +1</div>
                      <div><span className="text-emerald-600">H2:</span> {n.x.toFixed(3)}x₁ + {n.y.toFixed(3)}x₂ + {bias.toFixed(3)} = -1</div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center mt-3">
                  <SmallStat label="SVs Found" value={String(SVs.length)} />
                  <SmallStat label="Violations" value={String(violations)} />
                  <SmallStat label="Half Margin" value={marginHalf.toFixed(3)} />
                </div>

                <div className="mt-3 p-3 bg-blue-50 rounded-xl border border-blue-200">
                  <div className="text-xs font-medium text-blue-800 mb-1">📚  Example Analysis:</div>
                  <div className="text-xs text-blue-700 space-y-1">
                    <div>For points (1,2) and (3,4) with constraint w₁ = w₂:</div>
                    <div>Expected: 𝐖 = (-0.5, -0.5), b = 2.5, margin = 2√2 ≈ 2.828</div>
                    <div>👆 Try manually adjusting θ and bias to match these values!</div>
                  </div>
                </div>
              </div>
            )}

            {/* Algorithm Comparison Card */}
            <div className="bg-slate-50 rounded-2xl p-4 border">
              <h3 className="font-semibold mb-2">Algorithm Summary</h3>
              <div className="space-y-3 text-sm">
                {showPerceptron && (
                  <div className="p-3 bg-purple-50 rounded-xl border border-purple-200">
                    <h4 className="font-medium text-purple-800 mb-1">Perceptron</h4>
                    <div className="text-xs text-purple-700 space-y-1">
                      <div>• Error-driven learning</div>
                      <div>• Guaranteed convergence if linearly separable</div>
                      <div>• Simple update rule: w ← w + α(y-ŷ)x</div>
                      <div>• Current weights: ({perceptronWeights.w0.toFixed(2)}, {perceptronWeights.w1.toFixed(2)}, {perceptronWeights.w2.toFixed(2)})</div>
                    </div>
                  </div>
                )}
                
                {showKNN && (
                  <div className="p-3 bg-green-50 rounded-xl border border-green-200">
                    <h4 className="font-medium text-green-800 mb-1">k-NN</h4>
                    <div className="text-xs text-green-700 space-y-1">
                      <div>• Lazy learning (no training phase)</div>
                      <div>• Local decision boundaries</div>
                      <div>• k = {kNN}, distance: {distanceMethod}</div>
                      {knnPrediction && <div>• Current prediction: Class {knnPrediction === 1 ? '+1' : '-1'}</div>}
                    </div>
                  </div>
                )}
                
                {!showKNN && !showPerceptron && (
                  <div className="p-3 bg-blue-50 rounded-xl border border-blue-200">
                    <h4 className="font-medium text-blue-800 mb-1">SVM</h4>
                    <div className="text-xs text-blue-700 space-y-1">
                      <div>• Margin maximization</div>
                      <div>• Global optimal solution</div>
                      <div>• Support vectors: {SVs.length}</div>
                      <div>• Margin: {(2 * marginHalf).toFixed(3)}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Kernel Hopscotch - Only show in SVM mode */}
        {!showKNN && !showPerceptron && (
          <div className="bg-white rounded-2xl shadow p-4 md:p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold">2) Kernel Hopscotch — Feature lift z = r^2</h2>
            <div className="flex gap-2">
              <button onClick={() => setRingData(makeRingData())} className="px-3 py-1.5 rounded-xl bg-slate-200 text-slate-900 text-sm">Remix Data</button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
            {/* Original space */}
            <div className="flex flex-col">
              <h4 className="text-sm font-semibold mb-3 text-slate-700">Original plane (x, y)</h4>
              <div className="flex justify-center">
                <RingPlot data={ringData} width={480} height={350} />
              </div>
              <p className="text-xs text-slate-600 mt-2 text-center">
                Purple points form a ring (class -1), cyan points are center + outside (class +1)
              </p>
            </div>

            {/* Lifted space */}
            <div className="flex flex-col">
              <h4 className="text-sm font-semibold mb-3 text-slate-700">Lifted feature space (z = r^2)</h4>
              <div className="flex justify-center">
                <LiftedPlot data={ringData} zThresh={zThresh} onChange={setZThresh} width={480} height={350} />
              </div>
              <p className="text-xs text-slate-600 mt-2 text-center">
                Green border = correct classification, red border = incorrect
              </p>
            </div>
          </div>

          <div className="bg-slate-50 rounded-2xl p-4 border">
            <h3 className="font-semibold mb-2">What's happening?</h3>
            <p className="text-sm text-slate-700 mb-2">
              In the original (x, y) plane, a ring vs. non‑ring is not linearly separable. Define z = r^2 = x^2 + y^2. The ring points cluster by z,
              so a straight threshold in z‑space corresponds to a curved boundary back in (x, y). That's the kernel idea: linear in feature space.
            </p>
            <p className="text-xs text-slate-600">
              Try adjusting the threshold slider to see how the linear decision boundary in z-space translates to a circular boundary in the original space.
            </p>
          </div>
        </div>
        )}

        {/* Equations — SVM at a glance - Only show in SVM mode */}
        {!showKNN && !showPerceptron && (
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl p-5 shadow border">
            <h2 className="text-xl font-semibold mb-3">3) Equations — SVM at a glance</h2>
            <div className="space-y-4 text-sm">
              <div>
                <div className="font-medium mb-2">Hard‑margin (primal):</div>
                <div className="bg-slate-50 p-4 rounded-xl border">
                  <div className="text-center font-mono text-base">
                    min<sub>w,b</sub> ½‖w‖²
                  </div>
                  <div className="text-center text-xs mt-1">
                    subject to: y<sub>i</sub>(w<sup>T</sup>x<sub>i</sub> + b) ≥ 1
                  </div>
                </div>
              </div>

              <div>
                <div className="font-medium mb-2">Soft‑margin (primal):</div>
                <div className="bg-slate-50 p-4 rounded-xl border">
                  <div className="text-center font-mono text-base">
                    min<sub>w,b,ξ</sub> ½‖w‖² + C∑<sub>i</sub>ξ<sub>i</sub>
                  </div>
                  <div className="text-center text-xs mt-1">
                    subject to: y<sub>i</sub>(w<sup>T</sup>x<sub>i</sub> + b) ≥ 1 − ξ<sub>i</sub>, ξ<sub>i</sub> ≥ 0
                  </div>
                </div>
              </div>

              <div>
                <div className="font-medium mb-2">Dual (with kernel K):</div>
                <div className="bg-slate-50 p-4 rounded-xl border">
                  <div className="text-center font-mono text-base">
                    max<sub>α</sub> ∑<sub>i</sub>α<sub>i</sub> − ½∑<sub>i,j</sub>α<sub>i</sub>α<sub>j</sub>y<sub>i</sub>y<sub>j</sub>K(x<sub>i</sub>, x<sub>j</sub>)
                  </div>
                  <div className="text-center text-xs mt-1">
                    subject to: ∑<sub>i</sub>α<sub>i</sub>y<sub>i</sub> = 0, 0 ≤ α<sub>i</sub> ≤ C
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="font-medium mb-2">Decision rule:</div>
                  <div className="bg-slate-50 p-3 rounded-xl border text-center font-mono">
                    f(x) = sign(∑<sub>i</sub>α<sub>i</sub>y<sub>i</sub>K(x<sub>i</sub>, x) + b)
                  </div>
                </div>
                <div>
                  <div className="font-medium mb-2">Margin width:</div>
                  <div className="bg-slate-50 p-3 rounded-xl border text-center font-mono">
                    d = 2/‖w‖
                  </div>
                </div>
              </div>

              <div>
                <div className="font-medium mb-2">Hinge loss (training surrogate):</div>
                <div className="bg-slate-50 p-4 rounded-xl border text-center font-mono">
                  L = ∑<sub>i</sub>max(0, 1 − y<sub>i</sub>(w<sup>T</sup>x<sub>i</sub> + b))
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-5 shadow border">
            <h2 className="text-xl font-semibold mb-3">Lecture examples — What to notice</h2>
            <ul className="list-disc pl-5 space-y-2 text-sm text-slate-700">
              <li><span className="font-medium">H0 / H1 / H2 construction:</span> pick nearest opposite‑class points, margins are parallel lines, midline is H0. Adjust until no violations.</li>
              <li><span className="font-medium">Different SV sets, same w:</span> multiple SV subsets can pin the same max‑margin separator.</li>
              <li><span className="font-medium">Two‑band layout:</span> any pair that fixes equal margins works; remove others to test necessity.</li>
              <li><span className="font-medium">1‑D threshold:</span> max‑margin sits midway between edge points; outliers motivate soft‑margin (tune C).</li>
            </ul>
            <p className="text-xs text-slate-500 mt-2">Use Auto‑Fit above to mimic the slides' construction with a coarse grid search.</p>
          </div>
        </div>
        )}

        {/* Tests & Sanity Checks - Only show in SVM mode */}
        {!showKNN && !showPerceptron && (
          <>
            <TestsPanel />

            {/* Footer: quick prompts - Only show in SVM mode */}
            <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
              <PromptCard title="Quick Check #1" text="If you remove a non‑support vector inside the class cluster, does the line move? Why or why not?" />
              <PromptCard title="Quick Check #2" text="Increase C. What happens to violations and to the margin width?" />
              <PromptCard title="Quick Check #3" text="In the ring example, where do most ring points land in z = r^2 space? How does the threshold separate classes?" />
            </div>
          </>
        )}

        {/* Perceptron Algorithm Explanations */}
        {showPerceptron && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl p-5 shadow border">
              <h2 className="text-xl font-semibold mb-3">Perceptron Learning Algorithm</h2>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Algorithm Overview</h4>
                  <p className="text-sm text-gray-600">
                    A linear binary classifier that learns a decision boundary by iteratively updating weights based on misclassified examples. 
                    The algorithm is guaranteed to converge if the data is linearly separable.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Mathematical Foundation</h4>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p><strong>Decision function:</strong> h(x) = sign(w₀ + w₁x₁ + w₂x₂)</p>
                    <p><strong>Update rule:</strong> wⱼ ← wⱼ + α(y - ŷ)xⱼ</p>
                    <p><strong>Learning rate α:</strong> Controls the step size of updates</p>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Convergence Properties</h4>
                  <p className="text-sm text-gray-600">
                    If the data is linearly separable, the perceptron is guaranteed to find a separating hyperplane in finite steps. 
                    For non-separable data, it may oscillate indefinitely.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-5 shadow border">
              <h2 className="text-xl font-semibold mb-3">Perceptron vs SVM vs k-NN</h2>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Learning Approach</h4>
                  <p className="text-sm text-gray-600">
                    <strong>Perceptron:</strong> Error-driven updates, finds any separating line<br />
                    <strong>SVM:</strong> Margin maximization, finds optimal separating line<br />
                    <strong>k-NN:</strong> Instance-based, no explicit boundary
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Decision Boundary</h4>
                  <p className="text-sm text-gray-600">
                    <strong>Perceptron:</strong> Linear, may not be unique<br />
                    <strong>SVM:</strong> Linear with maximum margin<br />
                    <strong>k-NN:</strong> Non-linear, locally adaptive
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Guarantees</h4>
                  <p className="text-sm text-gray-600">
                    <strong>Perceptron:</strong> Convergence if linearly separable<br />
                    <strong>SVM:</strong> Global optimum, handles non-separable data<br />
                    <strong>k-NN:</strong> No convergence issues, always produces result
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Computational Complexity</h4>
                  <p className="text-sm text-gray-600">
                    <strong>Perceptron:</strong> O(n) per iteration, fast training<br />
                    <strong>SVM:</strong> O(n²) training, but better generalization<br />
                    <strong>k-NN:</strong> O(1) training, O(n) prediction
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* k-NN Algorithm Explanations */}
        {showKNN && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl p-5 shadow border">
              <h2 className="text-xl font-semibold mb-3">k-Nearest Neighbors (k-NN)</h2>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Algorithm Overview</h4>
                  <p className="text-sm text-gray-600">
                    A lazy learning algorithm that classifies data points based on the class of their k nearest neighbors. 
                    It makes no assumptions about the data distribution and can capture complex decision boundaries.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Distance Metrics</h4>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p><strong>Euclidean:</strong> Standard straight-line distance. Good for continuous features.</p>
                    <p><strong>Manhattan:</strong> City-block distance. Less sensitive to outliers.</p>
                    <p><strong>Cosine:</strong> Measures angle between vectors. Good for high-dimensional sparse data.</p>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Choosing k</h4>
                  <p className="text-sm text-gray-600">
                    Small k: More sensitive to noise, complex boundaries. 
                    Large k: Smoother boundaries, may miss local patterns. 
                    Odd k helps avoid ties in binary classification.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-5 shadow border">
              <h2 className="text-xl font-semibold mb-3">k-NN vs SVM Comparison</h2>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Learning Type</h4>
                  <p className="text-sm text-gray-600">
                    <strong>k-NN:</strong> Lazy (instance-based) - No training phase<br />
                    <strong>SVM:</strong> Eager (model-based) - Learns decision boundary
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Decision Boundary</h4>
                  <p className="text-sm text-gray-600">
                    <strong>k-NN:</strong> Complex, local regions (Voronoi-like)<br />
                    <strong>SVM:</strong> Global optimum with maximum margin
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Computational Complexity</h4>
                  <p className="text-sm text-gray-600">
                    <strong>k-NN:</strong> O(1) training, O(n) prediction<br />
                    <strong>SVM:</strong> O(n²) training, O(k) prediction (k = support vectors)
                  </p>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Noise Sensitivity</h4>
                  <p className="text-sm text-gray-600">
                    <strong>k-NN:</strong> High (especially k=1)<br />
                    <strong>SVM:</strong> Low (margin maximization provides robustness)
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SmallStat({ label, value, tone }: { label: string; value: string; tone?: "pos" | "neg" }) {
  const bg = tone === "pos" ? "bg-emerald-50" : tone === "neg" ? "bg-rose-50" : "bg-white";
  return (
    <div className={`${bg} rounded-xl p-2 shadow-sm`}>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}

function MetricsCard({ metrics }: { metrics: ReturnType<typeof computeMetrics> }) {
  const { TP, TN, FP, FN, acc, prec, rec, f1, avgHinge } = metrics;
  return (
    <div className="bg-white rounded-2xl p-4 border shadow-sm">
      <h3 className="font-semibold mb-3">Effectiveness — Confusion Matrix & Scores</h3>
      <div className="grid grid-cols-2 gap-3 text-center mb-3">
        <div className="bg-slate-50 rounded-xl p-3 border">
          <div className="text-xs text-slate-500 mb-2">Confusion Matrix (positive = +1)</div>
          <div className="grid grid-cols-2 gap-1 text-sm">
            <div className="bg-white rounded p-2">TP<br /><span className="text-lg font-semibold">{TP}</span></div>
            <div className="bg-white rounded p-2">FP<br /><span className="text-lg font-semibold">{FP}</span></div>
            <div className="bg-white rounded p-2">FN<br /><span className="text-lg font-semibold">{FN}</span></div>
            <div className="bg-white rounded p-2">TN<br /><span className="text-lg font-semibold">{TN}</span></div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <SmallStat label="Accuracy" value={`${(acc * 100).toFixed(1)}%`} />
          <SmallStat label="Precision" value={`${(prec * 100).toFixed(1)}%`} />
          <SmallStat label="Recall" value={`${(rec * 100).toFixed(1)}%`} />
          <SmallStat label="F1" value={`${(f1 * 100).toFixed(1)}%`} />
          <SmallStat label="Avg Hinge" value={avgHinge.toFixed(2)} />
        </div>
      </div>
      <div className="text-xs text-slate-600">Classification by sign of f(x) = n·x + b. Hinge = max(0, 1 − y·f(x)).</div>
    </div>
  );
}

function PromptCard({ title, text }: { title: string; text: string }) {
  return (
    <div className="bg-white rounded-2xl p-4 shadow border">
      <div className="text-sm font-semibold mb-1">{title}</div>
      <div className="text-sm text-slate-700">{text}</div>
    </div>
  );
}

function RingPlot({ data, width = 480, height = 320 }: { data: Pt[]; width?: number; height?: number }) {
  const pad = 10;
  const rMax = 9;
  const toScreen = (x: number, y: number) => ({
    x: pad + ((x + rMax) / (2 * rMax)) * (width - 2 * pad),
    y: pad + ((rMax - y) / (2 * rMax)) * (height - 2 * pad),
  });
  return (
    <svg width={width} height={height} className="rounded-2xl border border-slate-200 bg-white">
      {/* axes */}
      <line x1={toScreen(-rMax, 0).x} y1={toScreen(-rMax, 0).y} x2={toScreen(rMax, 0).x} y2={toScreen(rMax, 0).y} stroke="#e5e7eb" />
      <line x1={toScreen(0, -rMax).x} y1={toScreen(0, -rMax).y} x2={toScreen(0, rMax).x} y2={toScreen(0, rMax).y} stroke="#e5e7eb" />

      {data.map(p => {
        const s = toScreen(p.x, p.y);
        const fill = p.label === -1 ? "#8b5cf6" : "#0ea5e9"; // purple ring vs cyan outside
        return <circle key={p.id} cx={s.x} cy={s.y} r={3.2} fill={fill} opacity={0.9} />;
      })}
    </svg>
  );
}

function LiftedPlot({ data, zThresh, onChange, width = 480, height = 320 }: { data: Pt[]; zThresh: number; onChange: (z: number) => void; width?: number; height?: number }) {
  // In lifted view, x‑axis is an index just for spreading; y‑axis is z = r^2
  const pad = 20;
  const zMin = 0;
  const zMax = Math.max(...data.map(d => d.r2 || 0)) * 1.05;
  const toScreenY = (z: number) => pad + (1 - (z - zMin) / (zMax - zMin)) * (height - 2 * pad);

  return (
    <div>
      <svg width={width} height={height} className="rounded-2xl border border-slate-200 bg-white">
        {/* y‑axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map(t => {
          const z = zMin + t * (zMax - zMin);
          const y = toScreenY(z);
          return (
            <g key={String(t)}>
              <line x1={40} x2={width - 10} y1={y} y2={y} stroke="#f1f5f9" />
              <text x={6} y={y + 4} className="text-xs fill-slate-500">{z.toFixed(1)}</text>
            </g>
          );
        })}

        {/* Threshold line */}
        <line x1={40} x2={width - 10} y1={toScreenY(zThresh)} y2={toScreenY(zThresh)} stroke="#111827" strokeWidth={2} />
        <text x={50} y={toScreenY(zThresh) - 8} className="text-xs fill-slate-700 font-medium">threshold: z = r² = {zThresh.toFixed(1)}</text>

        {/* Points (sorted by z to reduce overdraw) */}
        {data.slice().sort((a, b) => (a.r2 || 0) - (b.r2 || 0)).map((p, i) => {
          const x = 50 + (i / data.length) * (width - 80);
          const y = toScreenY(p.r2 || 0);
          const predicted = (p.r2 || 0) >= zThresh ? 1 : -1;
          const correct = predicted === p.label;
          const fill = p.label === -1 ? "#8b5cf6" : "#0ea5e9";
          const stroke = correct ? "#22c55e" : "#ef4444";
          return <circle key={p.id} cx={x} cy={y} r={3.2} fill={fill} stroke={stroke} strokeWidth={1.5} />;
        })}
      </svg>

      <div className="mt-2">
        <input type="range" min={zMin} max={zMax || 1} step={0.1} value={zThresh} onChange={e => onChange(parseFloat(e.target.value))} className="w-full" />
        <div className="text-xs text-slate-600">Slide to move the straight decision boundary in z‑space (equivalent to a curved boundary in x–y).</div>
      </div>
    </div>
  );
}

// ---------- Point renderer (extracted to avoid inline closures) ----------
function renderPoint(p: Pt, ctx: { 
  toScreen: (x: number, y: number) => { x: number; y: number }; 
  n: N2; 
  bias: number; 
  marginHalf: number; 
  SVs: string[]; 
  setHoverId: (s: string | null) => void; 
  setDraggingId: (s: string) => void; 
  hoverId: string | null;
  showKNN?: boolean;
  nearestNeighbors?: Array<{point: Pt, distance: number}>;
  isCurrentPerceptronStep?: boolean;
}) {
  const { toScreen, n, bias, marginHalf, SVs, setHoverId, setDraggingId, hoverId, showKNN, nearestNeighbors, isCurrentPerceptronStep } = ctx;
  const pos = toScreen(p.x, p.y);
  const isSV = SVs.includes(p.id);
  const d = signedDistance(p, n, bias);
  const correct = (d >= 0 && p.label === 1) || (d <= 0 && p.label === -1);
  const insideMargin = Math.abs(d) < marginHalf - 1e-6;
  const mis = !correct;

  const fill = p.label === 1 ? "#2563eb" : "#f97316"; // blue / orange
  let ring = "#111827"; // default gray
  
  if (isCurrentPerceptronStep) {
    // Special highlighting for current step in Perceptron mode
    ring = "#8b5cf6"; // purple for current step
  } else if (showKNN && nearestNeighbors) {
    const isNearest = nearestNeighbors.some(({ point: np }) => np.id === p.id)
    ring = isNearest ? "#10b981" : "#111827" // green for nearest neighbors, gray otherwise
  } else {
    ring = mis ? "#ef4444" : insideMargin ? "#f59e0b" : isSV ? "#10b981" : "#111827"; // red, amber, green, gray
  }

  return (
    <g key={p.id} onMouseEnter={() => setHoverId(p.id)} onMouseLeave={() => setHoverId(null)}>
      <circle
        cx={pos.x} cy={pos.y} r={isCurrentPerceptronStep ? 10 : isSV ? 8 : 6}
        fill={fill}
        stroke={ring} strokeWidth={isCurrentPerceptronStep ? 4 : isSV ? 3 : 2}
        onMouseDown={() => setDraggingId(p.id)}
        style={{ cursor: "grab" }}
      />
      {/* Pulsing animation for current step */}
      {isCurrentPerceptronStep && (
        <circle
          cx={pos.x} cy={pos.y} r={15}
          fill="none" 
          stroke="#8b5cf6" 
          strokeWidth={2}
          strokeDasharray="3 3"
          opacity={0.6}
        >
          <animate attributeName="r" values="15;20;15" dur="2s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.6;0.3;0.6" dur="2s" repeatCount="indefinite" />
        </circle>
      )}
      {hoverId === p.id && (
        <text x={pos.x + 10} y={pos.y - 10} className="text-xs fill-gray-800 select-none">
          ({p.x.toFixed(1)}, {p.y.toFixed(1)})
        </text>
      )}
    </g>
  );
}

// ---------- Tests Panel ----------
function TestsPanel() {
  // Test 1: Simple separable layout — horizontal split
  const pts1: Pt[] = [
    { id: "t1a", x: -2, y: -1, label: -1 },
    { id: "t1b", x: 0, y: -2, label: -1 },
    { id: "t1c", x: 2, y: -1.5, label: -1 },
    { id: "t1d", x: -2, y: 2, label: 1 },
    { id: "t1e", x: 1, y: 2.5, label: 1 },
  ];
  const n1 = { x: 0, y: 1 };
  const b1 = 0; // line y + b = 0 → y = 0
  const m1 = computeMetrics(pts1, n1, b1);

  // Test 2: Margin and SVs should be positive and SVs not empty for clusters
  const pts2 = presetLayout("clusters");
  const n2 = { x: 1 / Math.sqrt(2), y: 1 / Math.sqrt(2) };
  const b2 = 0;
  const mm2 = computeMarginAndSVs(pts2, n2, b2);

  // Test 3: Ring thresholding should classify center vs ring with a middle z
  const ring = makeRingData(60);
  const zVals = ring.map(r => r.r2 || 0);
  const zMin = Math.min(...zVals);
  const zMax = Math.max(...zVals);
  const zMid = (zMin + zMax) / 2;
  const correctRate = ring.filter(p => ((p.r2 || 0) >= zMid ? 1 : -1) === p.label).length / ring.length;

  // Test 4: Hinge loss is never negative
  const hingeNonNeg = m1.avgHinge >= 0 && mm2.marginHalf >= 0;

  // Test 5: k-fold mean metrics are within [0,1]
  const cvProbe = kFold(presetLayout("clusters"), 3, 1);
  const metricsBounded = (
    cvProbe.mean.acc >= 0 && cvProbe.mean.acc <= 1 &&
    cvProbe.mean.prec >= 0 && cvProbe.mean.prec <= 1 &&
    cvProbe.mean.rec >= 0 && cvProbe.mean.rec <= 1 &&
    cvProbe.mean.f1 >= 0 && cvProbe.mean.f1 <= 1
  );

  const tests: Array<{ name: string; pass: boolean; details: string }> = [
    { name: "Separable horizontal split accuracy", pass: m1.acc === 1, details: `acc=${m1.acc.toFixed(2)}` },
    { name: "Clusters have finite margin", pass: mm2.marginHalf > 0, details: `halfMargin=${mm2.marginHalf.toFixed(2)}` },
    { name: "Clusters SVs non-empty", pass: mm2.SVs.length > 0, details: `SVs=${mm2.SVs.length}` },
    { name: "Ring mid-z gives reasonable accuracy", pass: correctRate > 0.5, details: `rate=${(correctRate*100).toFixed(1)}%` },
    { name: "Hinge loss non-negative", pass: hingeNonNeg, details: `avgHinge(t1)=${m1.avgHinge.toFixed(2)}` },
    { name: "k-fold means in [0,1]", pass: metricsBounded, details: `acc=${cvProbe.mean.acc.toFixed(2)}, f1=${cvProbe.mean.f1.toFixed(2)}` },
  ];

  return (
    <div className="mt-6 bg-white rounded-2xl p-4 border shadow">
      <h2 className="text-xl font-semibold mb-3">Tests — Sanity checks</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {tests.map(({ name, pass, details }) => (
          <div key={name} className={`rounded-xl p-3 ${pass ? 'bg-emerald-50 border-emerald-200' : 'bg-rose-50 border-rose-200'} border`}>
            <div className={`text-sm font-semibold mb-1 ${pass ? 'text-emerald-800' : 'text-rose-800'}`}>
              {pass ? '✓ PASS' : '✗ FAIL'}
            </div>
            <div className="text-xs text-slate-700 mb-1">{name}</div>
            <div className="text-xs text-slate-500">{details}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
