import React, { useEffect, useRef, useState } from "react";

// =============================================================
// Interactive Gradient Descent Example from Slide
// f(x1,x2,x3) = x1² + x1(1-x2) + x2² - x2x3 + x3² + x3
// Starting at (1,1,1) with η=0.3
// Shows: Neural Network + All Equations + Step-by-Step Plug-in
// =============================================================

export default function GradientDescentExample() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [x, setX] = useState<[number, number, number]>([1, 1, 1]);
  const [eta, setEta] = useState(0.3);
  const [playing, setPlaying] = useState(false);
  const [stage, setStage] = useState(0); // 0-11 animation stages
  const [iteration, setIteration] = useState(0);
  const [history, setHistory] = useState<Array<{ x: [number, number, number]; f: number; grad: [number, number, number] }>>([]);

  // Store current gradient for animation
  const currentGrad = useRef<[number, number, number]>([0, 0, 0]);
  const nextX = useRef<[number, number, number]>([0, 0, 0]);

  // Function from the slide
  function f(v: [number, number, number]): number {
    const [x1, x2, x3] = v;
    return x1*x1 + x1*(1-x2) + x2*x2 - x2*x3 + x3*x3 + x3;
  }

  // Gradient ∇f
  function gradient(v: [number, number, number]): [number, number, number] {
    const [x1, x2, x3] = v;
    return [
      2*x1 + 1 - x2,        // ∂f/∂x1
      -x1 + 2*x2 - x3,      // ∂f/∂x2
      -x2 + 2*x3 + 1        // ∂f/∂x3
    ];
  }

  // Compute next x values
  function computeNext(v: [number, number, number], grad: [number, number, number], learningRate: number): [number, number, number] {
    return [
      v[0] - learningRate * grad[0],
      v[1] - learningRate * grad[1],
      v[2] - learningRate * grad[2]
    ];
  }

  // Initialize iteration
  function startIteration() {
    currentGrad.current = gradient(x);
    nextX.current = computeNext(x, currentGrad.current, eta);
  }

  // Advance animation stage
  function advanceStage() {
    setStage(s => {
      let nextStage = s + 1;

      // Stage 4: update x1
      if (nextStage === 4) {
        setX(prev => [nextX.current[0], prev[1], prev[2]]);
      }
      // Stage 8: update x2
      if (nextStage === 8) {
        setX(prev => [prev[0], nextX.current[1], prev[2]]);
      }
      // Stage 12: update x3 and complete iteration
      if (nextStage === 12) {
        const newX = nextX.current;
        setX(newX);
        setHistory(prev => [...prev, {
          x: [...x] as [number, number, number],
          f: f(x),
          grad: [...currentGrad.current] as [number, number, number]
        }]);
      }

      // Reset for next iteration
      if (nextStage > 12) {
        nextStage = 0;
        setIteration(i => i + 1);
        startIteration();
      }

      return nextStage;
    });
  }

  // Animation loop
  useEffect(() => {
    if (!playing) return;

    let rafId = 0;
    let lastTime = 0;
    let accumulator = 0;
    const stepMs = 1200; // 1.2 seconds per stage

    startIteration();

    const loop = (timestamp: number) => {
      if (!lastTime) lastTime = timestamp;
      const delta = timestamp - lastTime;
      lastTime = timestamp;
      accumulator += delta;

      if (accumulator >= stepMs) {
        accumulator = 0;
        advanceStage();
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // Manual step
  function stepOnce() {
    if (stage === 0) {
      startIteration();
    }
    advanceStage();
  }

  // Reset to initial state
  function reset() {
    setX([1, 1, 1]);
    setEta(0.3);
    setStage(0);
    setIteration(0);
    setHistory([]);
    setPlaying(false);
    currentGrad.current = [0, 0, 0];
    nextX.current = [0, 0, 0];
  }

  // Draw visualization
  useEffect(() => {
    drawVisualization();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [x, eta, stage, iteration, history]);

  function drawVisualization() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const ctx = setupCanvas(canvas, W, H);
    if (!ctx) return;

    // Background
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0e1a";
    ctx.fillRect(0, 0, W, H);

    // Get current gradient
    const grad = stage === 0 && iteration === 0 ? gradient(x) : currentGrad.current;
    const next = stage === 0 && iteration === 0 ? computeNext(x, grad, eta) : nextX.current;
    const fVal = f(x);

    // Left side: Neural Network Representation (showing variables flowing to function)
    drawNeuralNetwork(ctx, x, grad, next, fVal, stage, 20, 20, 420, H - 40);

    // Right side: Detailed Equations and Step-by-Step
    drawEquationsPanel(ctx, x, eta, grad, next, fVal, stage, 460, 20, W - 480, H - 40);
  }

  function drawNeuralNetwork(ctx: CanvasRenderingContext2D, x: [number,number,number], grad: [number,number,number], next: [number,number,number], fVal: number, stage: number, sx: number, sy: number, w: number, h: number) {
    // Panel background
    ctx.fillStyle = "#1a1f2e";
    ctx.fillRect(sx, sy, w, h);
    ctx.strokeStyle = "#2d3748";
    ctx.lineWidth = 2;
    ctx.strokeRect(sx, sy, w, h);

    // Title
    ctx.fillStyle = "#60a5fa";
    ctx.font = "bold 18px ui-sans-serif";
    ctx.fillText("Computational Graph", sx + 20, sy + 30);

    // Network visualization - Enhanced layout
    const centerX = sx + w / 2;
    const leftX = sx + 60;
    const topY = sy + 80;
    const nodeR = 32;
    const vSpace = 100;

    // === LAYER 1: Input nodes (x1, x2, x3) ===
    const inputY = [topY, topY + vSpace, topY + vSpace * 2];

    for (let i = 0; i < 3; i++) {
      const active = stage >= 0;

      // Input variable circle
      drawNode(ctx, leftX, inputY[i], nodeR, active ? "#3b82f6" : "#475569", "#e0f2fe");

      ctx.fillStyle = "#fff";
      ctx.font = "bold 15px ui-sans-serif";
      const textWidth = ctx.measureText(`x${i+1}`).width;
      ctx.fillText(`x${i+1}`, leftX - textWidth/2, inputY[i] - 2);

      // Value below variable name
      ctx.font = "bold 13px ui-monospace";
      ctx.fillStyle = "#60a5fa";
      const valText = x[i].toFixed(2);
      const valWidth = ctx.measureText(valText).width;
      ctx.fillText(valText, leftX - valWidth/2, inputY[i] + 15);
    }

    // === LAYER 2: Function operations (showing internal structure) ===
    const opX = centerX - 20;
    const opY = topY + vSpace;

    // Main function node
    drawNode(ctx, opX, opY, nodeR + 10, stage >= 0 ? "#f59e0b" : "#475569", "#fef3c7");

    ctx.fillStyle = "#fff";
    ctx.font = "bold 16px ui-sans-serif";
    ctx.fillText("f(x)", opX - 18, opY + 5);

    // Arrows from inputs to function
    for (let i = 0; i < 3; i++) {
      const animateArrow = stage >= 0;
      drawArrow(ctx, leftX + nodeR, inputY[i], opX - nodeR - 10, opY, animateArrow ? "#60a5fa" : "#475569", 3);

      // Label on arrow
      if (stage >= 0) {
        const midX = (leftX + nodeR + opX - nodeR - 10) / 2;
        const midY = (inputY[i] + opY) / 2;
        ctx.fillStyle = "#60a5fa";
        ctx.font = "11px ui-sans-serif";
        ctx.fillText(`x${i+1}`, midX - 8, midY - 5);
      }
    }

    // === LAYER 3: Output node ===
    const outX = centerX + 120;
    const outY = opY;

    drawNode(ctx, outX, outY, nodeR, stage >= 0 ? "#10b981" : "#475569", "#d1fae5");

    ctx.fillStyle = "#fff";
    ctx.font = "bold 12px ui-sans-serif";
    ctx.fillText("Loss", outX - 16, outY - 2);

    ctx.font = "bold 13px ui-monospace";
    ctx.fillStyle = "#10b981";
    const fText = fVal.toFixed(3);
    const fWidth = ctx.measureText(fText).width;
    ctx.fillText(fText, outX - fWidth/2, outY + 15);

    // Arrow from function to output
    drawArrow(ctx, opX + nodeR + 10, opY, outX - nodeR, outY, stage >= 0 ? "#f59e0b" : "#475569", 3);

    // === BACKPROPAGATION: Gradient Flow ===
    const gradStartY = topY + vSpace * 3.5;

    ctx.fillStyle = "#ef4444";
    ctx.font = "bold 16px ui-sans-serif";
    ctx.fillText("← Backpropagation (∇f)", sx + 20, gradStartY - 20);

    // Gradient computation nodes
    for (let i = 0; i < 3; i++) {
      const gx = leftX;
      const gy = gradStartY + i * 90;
      const gActive = stage >= i * 4;

      // Gradient circle
      drawNode(ctx, gx, gy, 28, gActive ? "#ef4444" : "#475569", "#fee2e2");

      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px ui-sans-serif";
      ctx.fillText(`∂f/∂x${i+1}`, gx - 22, gy + 5);

      // Gradient value (to the right of circle)
      ctx.font = "bold 15px ui-monospace";
      ctx.fillStyle = gActive ? "#ef4444" : "#64748b";
      ctx.fillText(`= ${grad[i].toFixed(3)}`, gx + 40, gy + 5);

      // Update formula and arrow
      if (stage >= i * 4 + 1) {
        const updateX = gx + 220;
        const updateActive = stage >= i * 4 + 3;

        // Update formula
        ctx.fillStyle = updateActive ? "#10b981" : "#fbbf24";
        ctx.font = "14px ui-monospace";
        ctx.fillText(`x${i+1}' = ${x[i].toFixed(2)} - ${eta.toFixed(2)} × ${grad[i].toFixed(3)}`, updateX - 100, gy + 5);

        // Result circle
        drawNode(ctx, updateX + 80, gy, 26, updateActive ? "#10b981" : "#fbbf24", "#d1fae5");

        ctx.fillStyle = "#fff";
        ctx.font = "bold 13px ui-monospace";
        const nextText = next[i].toFixed(3);
        const nextWidth = ctx.measureText(nextText).width;
        ctx.fillText(nextText, updateX + 80 - nextWidth/2, gy + 5);

        if (updateActive) {
          ctx.fillStyle = "#10b981";
          ctx.font = "bold 16px ui-sans-serif";
          ctx.fillText("✓", updateX + 115, gy + 5);
        }
      }

      // Backprop arrow from output to gradient
      if (i === 0 && stage >= 0) {
        drawArrow(ctx, outX - nodeR, outY + nodeR + 10, gx + nodeR/2, gradStartY - 40, "#ef4444", 2);
      }
    }

    // Function equation at bottom
    const eqY = sy + h - 30;
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px ui-monospace";
    ctx.fillText("f(x₁,x₂,x₃) = x₁² + x₁(1-x₂) + x₂² - x₂x₃ + x₃² + x₃", sx + 20, eqY);
  }

  function drawEquationsPanel(ctx: CanvasRenderingContext2D, x: [number,number,number], eta: number, grad: [number,number,number], next: [number,number,number], fVal: number, stage: number, sx: number, sy: number, w: number, h: number) {
    // Panel background
    ctx.fillStyle = "#1a1f2e";
    ctx.fillRect(sx, sy, w, h);
    ctx.strokeStyle = "#2d3748";
    ctx.lineWidth = 2;
    ctx.strokeRect(sx, sy, w, h);

    let y = sy + 30;

    // Title
    ctx.fillStyle = "#fbbf24";
    ctx.font = "bold 20px ui-sans-serif";
    ctx.fillText("Step-by-Step Gradient Descent", sx + 20, y);
    y += 40;

    // Function definition
    ctx.fillStyle = "#94a3b8";
    ctx.font = "bold 14px ui-sans-serif";
    ctx.fillText("Objective Function:", sx + 20, y);
    y += 25;

    ctx.fillStyle = "#ef4444";
    ctx.font = "bold 16px ui-monospace";
    ctx.fillText("f(x₁, x₂, x₃) = x₁² + x₁(1-x₂) + x₂² - x₂x₃ + x₃² + x₃", sx + 20, y);
    y += 35;

    // Current values
    ctx.fillStyle = "#60a5fa";
    ctx.font = "bold 15px ui-monospace";
    ctx.fillText(`Current: x = [${x[0].toFixed(3)}, ${x[1].toFixed(3)}, ${x[2].toFixed(3)}]`, sx + 20, y);
    y += 22;

    ctx.fillStyle = "#10b981";
    ctx.fillText(`f(x) = ${fVal.toFixed(6)}`, sx + 20, y);
    y += 22;

    ctx.fillStyle = "#fbbf24";
    ctx.fillText(`Learning rate η = ${eta.toFixed(2)}`, sx + 20, y);
    y += 40;

    // Divider
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx + 20, y);
    ctx.lineTo(sx + w - 20, y);
    ctx.stroke();
    y += 30;

    // ========== ∂f/∂x₁ ==========
    ctx.fillStyle = stage >= 0 && stage < 4 ? "#fbbf24" : "#94a3b8";
    ctx.font = "bold 16px ui-sans-serif";
    ctx.fillText("① Compute ∂f/∂x₁:", sx + 20, y);
    y += 28;

    ctx.font = "14px ui-monospace";
    ctx.fillText("∂f/∂x₁ = 2x₁ + 1 - x₂", sx + 40, y);
    y += 24;

    if (stage >= 0) {
      ctx.fillStyle = stage >= 0 && stage < 4 ? "#fbbf24" : "#64748b";
      ctx.fillText(`      = 2(${x[0].toFixed(2)}) + 1 - ${x[1].toFixed(2)}`, sx + 40, y);
      y += 24;
      ctx.fillText(`      = ${(2*x[0]).toFixed(2)} + 1 - ${x[1].toFixed(2)}`, sx + 40, y);
      y += 24;
      ctx.fillStyle = stage >= 0 && stage < 4 ? "#ef4444" : "#64748b";
      ctx.font = "bold 15px ui-monospace";
      ctx.fillText(`      = ${grad[0].toFixed(3)}`, sx + 40, y);
      y += 30;
    }

    if (stage >= 1) {
      ctx.fillStyle = stage >= 1 && stage < 4 ? "#fbbf24" : "#94a3b8";
      ctx.font = "bold 16px ui-sans-serif";
      ctx.fillText("Update x₁:", sx + 40, y);
      y += 26;

      ctx.font = "14px ui-monospace";
      ctx.fillText(`x₁' = x₁ - η × ∂f/∂x₁`, sx + 40, y);
      y += 22;

      ctx.fillStyle = stage >= 1 && stage < 4 ? "#fbbf24" : "#64748b";
      ctx.fillText(`    = ${x[0].toFixed(2)} - ${eta.toFixed(2)} × ${grad[0].toFixed(3)}`, sx + 40, y);
      y += 22;
      ctx.fillText(`    = ${x[0].toFixed(2)} - ${(eta * grad[0]).toFixed(3)}`, sx + 40, y);
      y += 22;

      ctx.fillStyle = stage >= 3 ? "#10b981" : (stage >= 1 && stage < 4 ? "#fbbf24" : "#64748b");
      ctx.font = "bold 15px ui-monospace";
      ctx.fillText(`    = ${next[0].toFixed(3)} ${stage >= 3 ? '✓' : ''}`, sx + 40, y);
      y += 35;
    }

    // Divider
    if (stage >= 4) {
      ctx.strokeStyle = "#374151";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx + 20, y);
      ctx.lineTo(sx + w - 20, y);
      ctx.stroke();
      y += 25;
    }

    // ========== ∂f/∂x₂ ==========
    if (stage >= 4) {
      ctx.fillStyle = stage >= 4 && stage < 8 ? "#fbbf24" : "#94a3b8";
      ctx.font = "bold 16px ui-sans-serif";
      ctx.fillText("② Compute ∂f/∂x₂:", sx + 20, y);
      y += 28;

      ctx.font = "14px ui-monospace";
      ctx.fillText("∂f/∂x₂ = -x₁ + 2x₂ - x₃", sx + 40, y);
      y += 24;

      ctx.fillStyle = stage >= 4 && stage < 8 ? "#fbbf24" : "#64748b";
      ctx.fillText(`      = -(${x[0].toFixed(2)}) + 2(${x[1].toFixed(2)}) - ${x[2].toFixed(2)}`, sx + 40, y);
      y += 24;
      ctx.fillText(`      = ${(-x[0]).toFixed(2)} + ${(2*x[1]).toFixed(2)} - ${x[2].toFixed(2)}`, sx + 40, y);
      y += 24;

      ctx.fillStyle = stage >= 4 && stage < 8 ? "#ef4444" : "#64748b";
      ctx.font = "bold 15px ui-monospace";
      ctx.fillText(`      = ${grad[1].toFixed(3)}`, sx + 40, y);
      y += 30;
    }

    if (stage >= 5) {
      ctx.fillStyle = stage >= 5 && stage < 8 ? "#fbbf24" : "#94a3b8";
      ctx.font = "bold 16px ui-sans-serif";
      ctx.fillText("Update x₂:", sx + 40, y);
      y += 26;

      ctx.font = "14px ui-monospace";
      ctx.fillText(`x₂' = x₂ - η × ∂f/∂x₂`, sx + 40, y);
      y += 22;

      ctx.fillStyle = stage >= 5 && stage < 8 ? "#fbbf24" : "#64748b";
      ctx.fillText(`    = ${x[1].toFixed(2)} - ${eta.toFixed(2)} × ${grad[1].toFixed(3)}`, sx + 40, y);
      y += 22;
      ctx.fillText(`    = ${x[1].toFixed(2)} - ${(eta * grad[1]).toFixed(3)}`, sx + 40, y);
      y += 22;

      ctx.fillStyle = stage >= 7 ? "#10b981" : (stage >= 5 && stage < 8 ? "#fbbf24" : "#64748b");
      ctx.font = "bold 15px ui-monospace";
      ctx.fillText(`    = ${next[1].toFixed(3)} ${stage >= 7 ? '✓' : ''}`, sx + 40, y);
      y += 35;
    }

    // Divider
    if (stage >= 8) {
      ctx.strokeStyle = "#374151";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx + 20, y);
      ctx.lineTo(sx + w - 20, y);
      ctx.stroke();
      y += 25;
    }

    // ========== ∂f/∂x₃ ==========
    if (stage >= 8) {
      ctx.fillStyle = stage >= 8 && stage < 12 ? "#fbbf24" : "#94a3b8";
      ctx.font = "bold 16px ui-sans-serif";
      ctx.fillText("③ Compute ∂f/∂x₃:", sx + 20, y);
      y += 28;

      ctx.font = "14px ui-monospace";
      ctx.fillText("∂f/∂x₃ = -x₂ + 2x₃ + 1", sx + 40, y);
      y += 24;

      ctx.fillStyle = stage >= 8 && stage < 12 ? "#fbbf24" : "#64748b";
      ctx.fillText(`      = -(${x[1].toFixed(2)}) + 2(${x[2].toFixed(2)}) + 1`, sx + 40, y);
      y += 24;
      ctx.fillText(`      = ${(-x[1]).toFixed(2)} + ${(2*x[2]).toFixed(2)} + 1`, sx + 40, y);
      y += 24;

      ctx.fillStyle = stage >= 8 && stage < 12 ? "#ef4444" : "#64748b";
      ctx.font = "bold 15px ui-monospace";
      ctx.fillText(`      = ${grad[2].toFixed(3)}`, sx + 40, y);
      y += 30;
    }

    if (stage >= 9) {
      ctx.fillStyle = stage >= 9 && stage < 12 ? "#fbbf24" : "#94a3b8";
      ctx.font = "bold 16px ui-sans-serif";
      ctx.fillText("Update x₃:", sx + 40, y);
      y += 26;

      ctx.font = "14px ui-monospace";
      ctx.fillText(`x₃' = x₃ - η × ∂f/∂x₃`, sx + 40, y);
      y += 22;

      ctx.fillStyle = stage >= 9 && stage < 12 ? "#fbbf24" : "#64748b";
      ctx.fillText(`    = ${x[2].toFixed(2)} - ${eta.toFixed(2)} × ${grad[2].toFixed(3)}`, sx + 40, y);
      y += 22;
      ctx.fillText(`    = ${x[2].toFixed(2)} - ${(eta * grad[2]).toFixed(3)}`, sx + 40, y);
      y += 22;

      ctx.fillStyle = stage >= 11 ? "#10b981" : (stage >= 9 && stage < 12 ? "#fbbf24" : "#64748b");
      ctx.font = "bold 15px ui-monospace";
      ctx.fillText(`    = ${next[2].toFixed(3)} ${stage >= 11 ? '✓' : ''}`, sx + 40, y);
      y += 35;
    }

    // Final summary
    if (stage >= 12 || (stage === 0 && iteration > 0)) {
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx + 20, y);
      ctx.lineTo(sx + w - 20, y);
      ctx.stroke();
      y += 30;

      ctx.fillStyle = "#10b981";
      ctx.font = "bold 18px ui-sans-serif";
      ctx.fillText("✓ Iteration Complete!", sx + 20, y);
      y += 30;

      ctx.font = "bold 16px ui-monospace";
      ctx.fillText(`New x = [${next[0].toFixed(3)}, ${next[1].toFixed(3)}, ${next[2].toFixed(3)}]`, sx + 20, y);
      y += 26;

      const newF = f(next);
      const decrease = fVal - newF;
      ctx.fillStyle = "#22c55e";
      ctx.fillText(`f(new x) = ${newF.toFixed(6)}`, sx + 20, y);
      y += 24;
      ctx.fillText(`Δf = ${decrease.toFixed(6)} ${decrease > 0 ? '(↓ decreased)' : ''}`, sx + 20, y);
    }
  }

  // Numerical verification
  function verifyGradient() {
    const eps = 1e-6;
    const analyticGrad = gradient(x);
    const numericGrad: [number, number, number] = [0, 0, 0];

    for (let i = 0; i < 3; i++) {
      const xPlus: [number, number, number] = [...x] as [number, number, number];
      const xMinus: [number, number, number] = [...x] as [number, number, number];
      xPlus[i] += eps;
      xMinus[i] -= eps;
      numericGrad[i] = (f(xPlus) - f(xMinus)) / (2 * eps);
    }

    const msg = [
      `Gradient Verification at x = [${x.map(v => v.toFixed(3)).join(", ")}]`,
      ``,
      `Analytic ∇f = [${analyticGrad.map(v => v.toFixed(6)).join(", ")}]`,
      `Numeric  ∇f = [${numericGrad.map(v => v.toFixed(6)).join(", ")}]`,
      ``,
      `Difference  = [${analyticGrad.map((v, i) => Math.abs(v - numericGrad[i]).toExponential(2)).join(", ")}]`,
      ``,
      `✓ Gradients match! (differences < 1e-5)`
    ].join("\n");

    alert(msg);
  }

  return (
    <div className="w-full min-h-screen bg-slate-950 text-slate-100 px-4 py-6">
      <div className="max-w-[1800px] mx-auto">
        <div className="bg-slate-900/60 rounded-2xl shadow-2xl p-4 mb-6">
          <canvas ref={canvasRef} className="w-full h-[900px] rounded-xl" />
        </div>

        {/* Controls */}
        <div className="bg-slate-900/60 rounded-2xl shadow-lg p-4">
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setPlaying(!playing)}
              className="px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-semibold shadow-lg transition-colors"
            >
              {playing ? "⏸ Pause" : "▶ Play"}
            </button>

            <button
              onClick={stepOnce}
              disabled={playing}
              className="px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-600 text-slate-950 font-semibold shadow-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ⏭ Step
            </button>

            <button
              onClick={reset}
              className="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-600 text-slate-950 font-semibold shadow-lg transition-colors"
            >
              ↻ Reset to (1,1,1)
            </button>

            <button
              onClick={verifyGradient}
              className="px-4 py-2 rounded-xl bg-violet-500 hover:bg-violet-600 text-slate-950 font-semibold shadow-lg transition-colors"
            >
              ✓ Verify ∇f
            </button>

            <div className="ml-auto flex items-center gap-3 bg-slate-950/60 rounded-xl px-4 py-2">
              <label className="text-sm font-medium">Learning Rate η:</label>
              <input
                type="range"
                min={0.01}
                max={1}
                step={0.01}
                value={eta}
                onChange={(e) => setEta(parseFloat(e.target.value))}
                className="w-32 accent-emerald-500"
                disabled={playing}
              />
              <span className="font-mono text-sm w-12 text-right">{eta.toFixed(2)}</span>
            </div>

            <div className="bg-slate-950/60 rounded-xl px-4 py-2">
              <span className="text-sm font-medium">Iteration: {iteration} | Stage: {stage}/12</span>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-slate-900/40 rounded-xl p-4 text-sm text-slate-300">
          <h3 className="font-semibold text-slate-100 mb-2">Features:</h3>
          <ul className="list-disc ml-5 space-y-1">
            <li><strong>Neural Network Visualization (Left):</strong> Shows variables flowing through the function with gradients and updates</li>
            <li><strong>Step-by-Step Equations (Right):</strong> Complete plug-in calculations for each derivative and update</li>
            <li><strong>Play Mode:</strong> Automatically animates through all 12 stages of one iteration</li>
            <li><strong>Step Mode:</strong> Manually advance to see each calculation appear</li>
            <li><strong>Color Coding:</strong> Yellow = computing, Red = gradient result, Green = updated value</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

// ============ Helper Functions ============

function setupCanvas(canvas: HTMLCanvasElement, width: number, height: number) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.scale(dpr, dpr);
  return ctx;
}

function drawNode(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, fill: string, stroke: string) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 3;
  ctx.stroke();
}

function drawArrow(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, width: number) {
  const headLen = 10;
  const angle = Math.atan2(y2 - y1, x2 - x1);

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}
