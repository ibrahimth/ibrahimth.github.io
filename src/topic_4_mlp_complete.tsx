import React, { useState, useEffect, useMemo, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Play, Pause, RotateCcw, StepForward, Brain, Layers, Zap, GitBranch, TrendingDown, Eye } from "lucide-react";

// ==========================================
// Topic 4: Complete MLP & Neural Network Learning
// Part 1: Multilayer Perceptron Architecture
// Part 2: Neural Network Learning (Backpropagation)
// ==========================================

// Activation functions and derivatives
const activations = {
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
  tanh: (x: number) => Math.tanh(x),
  relu: (x: number) => Math.max(0, x),
};

const activationDerivatives = {
  sigmoid: (x: number) => {
    const s = activations.sigmoid(x);
    return s * (1 - s);
  },
  tanh: (x: number) => {
    const t = activations.tanh(x);
    return 1 - t * t;
  },
  relu: (x: number) => (x > 0 ? 1 : 0),
};

// ==========================================
// Component 1: MLP Architecture Visualizer
// ==========================================
function MLPArchitectureVisualizer() {
  const [layers, setLayers] = useState([2, 3, 1]); // Input, Hidden, Output
  const [activationFn, setActivationFn] = useState<keyof typeof activations>('sigmoid');

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Example weights for visualization (normally random)
  const sampleWeights = useMemo(() => {
    const weights: number[][][] = [];
    for (let l = 0; l < layers.length - 1; l++) {
      const layerWeights: number[][] = [];
      for (let i = 0; i < layers[l]; i++) {
        const neuronWeights: number[] = [];
        for (let j = 0; j < layers[l + 1]; j++) {
          neuronWeights.push(Math.random() * 2 - 1);
        }
        layerWeights.push(neuronWeights);
      }
      weights.push(layerWeights);
    }
    return weights;
  }, [layers]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Calculate positions
    const layerSpacing = w / (layers.length + 1);
    const positions: Array<Array<{ x: number; y: number }>> = [];

    layers.forEach((count, layerIdx) => {
      const layerPos: Array<{ x: number; y: number }> = [];
      const x = layerSpacing * (layerIdx + 1);
      const neuronSpacing = h / (count + 1);

      for (let i = 0; i < count; i++) {
        const y = neuronSpacing * (i + 1);
        layerPos.push({ x, y });
      }
      positions.push(layerPos);
    });

    // Draw connections (weights)
    for (let l = 0; l < positions.length - 1; l++) {
      const currentLayer = positions[l];
      const nextLayer = positions[l + 1];

      currentLayer.forEach((from, i) => {
        nextLayer.forEach((to, j) => {
          const weight = sampleWeights[l][i][j];
          const normalized = (weight + 1) / 2; // Normalize to [0, 1]

          ctx.strokeStyle = weight > 0
            ? `rgba(34, 197, 94, ${Math.abs(weight) * 0.5})`
            : `rgba(239, 68, 68, ${Math.abs(weight) * 0.5})`;
          ctx.lineWidth = Math.abs(weight) * 3 + 1;

          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        });
      });
    }

    // Draw neurons
    positions.forEach((layerPos, layerIdx) => {
      layerPos.forEach((pos, neuronIdx) => {
        // Neuron circle
        ctx.fillStyle = layerIdx === 0 ? '#3b82f6' : layerIdx === layers.length - 1 ? '#10b981' : '#8b5cf6';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 20, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 12px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        if (layerIdx === 0) {
          ctx.fillText(`x${neuronIdx + 1}`, pos.x, pos.y);
        } else if (layerIdx === layers.length - 1) {
          ctx.fillText('y', pos.x, pos.y);
        } else {
          ctx.fillText(`h${neuronIdx + 1}`, pos.x, pos.y);
        }
      });

      // Layer label
      ctx.fillStyle = '#64748b';
      ctx.font = '14px system-ui';
      ctx.textAlign = 'center';

      const layerX = layerSpacing * (layerIdx + 1);
      const labelY = h - 20;

      if (layerIdx === 0) {
        ctx.fillText('Input Layer', layerX, labelY);
      } else if (layerIdx === layers.length - 1) {
        ctx.fillText('Output Layer', layerX, labelY);
      } else {
        ctx.fillText(`Hidden Layer ${layerIdx}`, layerX, labelY);
      }
    });

  }, [layers, sampleWeights]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Layers className="w-5 h-5" />
          MLP Architecture Visualizer
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          A <strong>Multilayer Perceptron (MLP)</strong> consists of multiple layers of neurons. Each connection
          has a weight (green = positive, red = negative), and each neuron applies an activation function.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Architecture</label>
              <div className="flex gap-2 items-center mt-1">
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={layers[0]}
                  onChange={(e) => setLayers([Number(e.target.value), layers[1], layers[2]])}
                  className="w-16 px-2 py-1 border rounded"
                />
                <span className="text-sm text-muted-foreground">→</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={layers[1]}
                  onChange={(e) => setLayers([layers[0], Number(e.target.value), layers[2]])}
                  className="w-16 px-2 py-1 border rounded"
                />
                <span className="text-sm text-muted-foreground">→</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={layers[2]}
                  onChange={(e) => setLayers([layers[0], layers[1], Number(e.target.value)])}
                  className="w-16 px-2 py-1 border rounded"
                />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Input → Hidden → Output neurons
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">Activation Function</label>
              <div className="flex gap-2 mt-1">
                {(['sigmoid', 'tanh', 'relu'] as const).map(fn => (
                  <Button
                    key={fn}
                    variant={activationFn === fn ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setActivationFn(fn)}
                  >
                    {fn}
                  </Button>
                ))}
              </div>
            </div>

            <div className="text-sm space-y-2 bg-blue-50 border border-blue-200 rounded-lg p-3">
              <h4 className="font-semibold text-blue-900">Key Concepts:</h4>
              <ul className="list-disc pl-5 space-y-1 text-blue-800 text-xs">
                <li><strong>Input layer:</strong> Receives input features (x₁, x₂, ...)</li>
                <li><strong>Hidden layers:</strong> Learn intermediate representations</li>
                <li><strong>Output layer:</strong> Produces final prediction</li>
                <li><strong>Weights:</strong> Connection strengths (learned via training)</li>
                <li><strong>Activation:</strong> Non-linear transformation (enables complex patterns)</li>
              </ul>
            </div>

            <div className="text-xs text-muted-foreground space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-8 h-1 bg-green-500 rounded"></div>
                <span>Positive weight</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-1 bg-red-500 rounded"></div>
                <span>Negative weight</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-blue-500 rounded-full border-2 border-white"></div>
                <span>Input neuron</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-purple-500 rounded-full border-2 border-white"></div>
                <span>Hidden neuron</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded-full border-2 border-white"></div>
                <span>Output neuron</span>
              </div>
            </div>
          </div>

          <div>
            <canvas
              ref={canvasRef}
              width={500}
              height={400}
              className="w-full h-auto border border-slate-200 rounded bg-white"
            />
          </div>
        </div>

        <div className="text-sm space-y-2 bg-purple-50 border border-purple-200 rounded-lg p-3">
          <h4 className="font-semibold text-purple-900">Total Parameters:</h4>
          <div className="font-mono text-purple-800">
            {layers.slice(0, -1).reduce((sum, curr, idx) => {
              const weights = curr * layers[idx + 1];
              const biases = layers[idx + 1];
              return sum + weights + biases;
            }, 0)} parameters
          </div>
          <p className="text-xs text-purple-700">
            (weights between layers + biases for each neuron)
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Component 2: Forward Propagation Animator
// ==========================================
function ForwardPropagationAnimator() {
  const [input, setInput] = useState([0.5, 0.3]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Simple 2-2-1 network
  const weights1 = useMemo(() => [[0.5, -0.3], [0.2, 0.4]], []);
  const bias1 = useMemo(() => [0.1, -0.1], []);
  const weights2 = useMemo(() => [[0.6], [-0.4]], []);
  const bias2 = useMemo(() => [0.2], []);

  const [activationFn] = useState<keyof typeof activations>('sigmoid');

  // Calculate forward pass
  const forward = useMemo(() => {
    const act = activations[activationFn];

    // Layer 1
    const z1 = weights1.map((w, i) => w[0] * input[0] + w[1] * input[1] + bias1[i]);
    const a1 = z1.map(act);

    // Layer 2
    const z2 = [weights2[0][0] * a1[0] + weights2[1][0] * a1[1] + bias2[0]];
    const a2 = z2.map(act);

    return {
      z1,
      a1,
      z2,
      a2,
      steps: [
        { desc: 'Input received', values: input },
        { desc: 'Hidden layer: z₁ = W₁·x + b₁', values: z1 },
        { desc: `Hidden layer: a₁ = ${activationFn}(z₁)`, values: a1 },
        { desc: 'Output layer: z₂ = W₂·a₁ + b₂', values: z2 },
        { desc: `Output: y = ${activationFn}(z₂)`, values: a2 },
      ]
    };
  }, [input, weights1, bias1, weights2, bias2, activationFn]);

  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(() => {
      setStep(s => (s + 1) % forward.steps.length);
    }, 1500);
    return () => clearInterval(interval);
  }, [playing, forward.steps.length]);

  const currentStep = forward.steps[step];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitBranch className="w-5 h-5" />
          Forward Propagation Step-by-Step
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          <strong>Forward propagation</strong> is how the network makes predictions: input flows through
          layers, each applying weights, biases, and activation functions.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Input Values</label>
              <div className="flex gap-2 mt-1">
                <input
                  type="number"
                  step="0.1"
                  value={input[0]}
                  onChange={(e) => setInput([parseFloat(e.target.value) || 0, input[1]])}
                  className="flex-1 px-2 py-1 border rounded"
                  placeholder="x₁"
                />
                <input
                  type="number"
                  step="0.1"
                  value={input[1]}
                  onChange={(e) => setInput([input[0], parseFloat(e.target.value) || 0])}
                  className="flex-1 px-2 py-1 border rounded"
                  placeholder="x₂"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => setPlaying(!playing)}
                size="sm"
              >
                {playing ? <><Pause className="w-4 h-4 mr-1" /> Pause</> : <><Play className="w-4 h-4 mr-1" /> Play</>}
              </Button>
              <Button
                onClick={() => setStep((s) => (s + 1) % forward.steps.length)}
                size="sm"
                variant="outline"
              >
                <StepForward className="w-4 h-4 mr-1" /> Next
              </Button>
              <Button
                onClick={() => setStep(0)}
                size="sm"
                variant="outline"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-semibold">Computation Steps:</h4>
              {forward.steps.map((s, i) => (
                <div
                  key={i}
                  className={`p-2 rounded border text-sm ${
                    i === step ? 'bg-blue-100 border-blue-300 font-semibold' : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      i === step ? 'bg-blue-500 text-white' : 'bg-slate-300 text-slate-600'
                    }`}>
                      {i + 1}
                    </div>
                    <span>{s.desc}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <AnimatePresence mode="wait">
              <motion.div
                key={step}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-blue-300 rounded-xl p-4"
              >
                <h3 className="font-semibold text-lg mb-2">{currentStep.desc}</h3>
                <div className="flex flex-wrap gap-2">
                  {currentStep.values.map((val, i) => (
                    <div key={i} className="bg-white px-4 py-2 rounded-lg shadow">
                      <div className="text-xs text-muted-foreground">
                        {step === 0 ? `x${i+1}` : step <= 2 ? `h${i+1}` : 'y'}
                      </div>
                      <div className="font-mono text-lg font-bold">{val.toFixed(4)}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>

            <div className="text-sm space-y-2 bg-amber-50 border border-amber-200 rounded-lg p-3">
              <h4 className="font-semibold text-amber-900">Formula Breakdown:</h4>
              <div className="space-y-1 text-xs text-amber-800 font-mono">
                <div>z = W · x + b (weighted sum + bias)</div>
                <div>a = σ(z) (activation function)</div>
                <div className="mt-2 text-sm">Where:</div>
                <div>• W = weight matrix</div>
                <div>• x = input or previous layer activations</div>
                <div>• b = bias vector</div>
                <div>• σ = activation function ({activationFn})</div>
              </div>
            </div>

            <div className="text-xs text-muted-foreground bg-slate-50 p-3 rounded">
              <strong>Final Output:</strong> {forward.a2[0].toFixed(4)}
              <div className="mt-1">
                This is the network's prediction for input [{input[0]}, {input[1]}]
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Component 3: XOR Problem Solver
// ==========================================
function XORProblemSolver() {
  const xorData = useMemo(() => [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ], []);

  const [weights1, setWeights1] = useState([[0.5, -0.3], [0.2, 0.4]]);
  const [bias1, setBias1] = useState([0.1, -0.1]);
  const [weights2, setWeights2] = useState([[0.6], [-0.4]]);
  const [bias2, setBias2] = useState([0.2]);

  const [epoch, setEpoch] = useState(0);
  const [training, setTraining] = useState(false);
  const [lossHistory, setLossHistory] = useState<number[]>([]);

  const learningRate = 0.5;

  // Forward pass
  const forward = (x: number[]) => {
    const z1 = weights1.map((w, i) => w[0] * x[0] + w[1] * x[1] + bias1[i]);
    const a1 = z1.map(z => activations.sigmoid(z));

    const z2 = [weights2[0][0] * a1[0] + weights2[1][0] * a1[1] + bias2[0]];
    const a2 = z2.map(z => activations.sigmoid(z));

    return { z1, a1, z2, a2: a2[0] };
  };

  // Backpropagation
  const backward = (x: number[], y: number) => {
    const { z1, a1, z2, a2 } = forward(x);

    // Output layer
    const dZ2 = a2 - y;
    const dW2 = a1.map(a => dZ2 * a);
    const dB2 = dZ2;

    // Hidden layer
    const dA1 = [dZ2 * weights2[0][0], dZ2 * weights2[1][0]];
    const dZ1 = dA1.map((da, i) => da * activationDerivatives.sigmoid(z1[i]));
    const dW1 = dZ1.map(dz => [dz * x[0], dz * x[1]]);
    const dB1 = dZ1;

    return { dW1, dB1, dW2, dB2 };
  };

  // Training step
  const trainStep = () => {
    let totalLoss = 0;

    xorData.forEach(({ x, y }) => {
      const { dW1, dB1, dW2, dB2 } = backward(x, y);

      // Update weights and biases
      setWeights1(w => w.map((row, i) => row.map((val, j) => val - learningRate * dW1[i][j])));
      setBias1(b => b.map((val, i) => val - learningRate * dB1[i]));
      setWeights2(w => w.map((row, i) => row.map((val, j) => val - learningRate * dW2[i])));
      setBias2(b => [b[0] - learningRate * dB2]);

      const pred = forward(x).a2;
      totalLoss += Math.pow(pred - y, 2);
    });

    setLossHistory(h => [...h, totalLoss / xorData.length].slice(-100));
    setEpoch(e => e + 1);
  };

  useEffect(() => {
    if (!training) return;
    const interval = setInterval(trainStep, 100);
    return () => clearInterval(interval);
  }, [training, weights1, bias1, weights2, bias2]);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw loss curve
    if (lossHistory.length > 1) {
      const maxLoss = Math.max(...lossHistory, 1);

      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();

      lossHistory.forEach((loss, i) => {
        const x = (i / 100) * w;
        const y = h - (loss / maxLoss) * h * 0.9;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.stroke();
    }

    // Labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '12px system-ui';
    ctx.fillText('Loss', 10, 20);
    ctx.fillText('Epoch →', w - 60, h - 10);

  }, [lossHistory]);

  const predictions = xorData.map(d => ({ ...d, pred: forward(d.x).a2 }));
  const currentLoss = lossHistory[lossHistory.length - 1] || 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          XOR Problem: The Classic Challenge
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          <strong>XOR (exclusive OR)</strong> is not linearly separable—a single perceptron cannot solve it.
          But a simple MLP with one hidden layer can! Watch the network learn this classic problem.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="flex gap-2">
              <Button
                onClick={() => setTraining(!training)}
                size="sm"
              >
                {training ? <><Pause className="w-4 h-4 mr-1" /> Pause</> : <><Play className="w-4 h-4 mr-1" /> Train</>}
              </Button>
              <Button
                onClick={trainStep}
                size="sm"
                variant="outline"
              >
                <StepForward className="w-4 h-4 mr-1" /> Step
              </Button>
              <Button
                onClick={() => {
                  setWeights1([[0.5, -0.3], [0.2, 0.4]]);
                  setBias1([0.1, -0.1]);
                  setWeights2([[0.6], [-0.4]]);
                  setBias2([0.2]);
                  setEpoch(0);
                  setLossHistory([]);
                  setTraining(false);
                }}
                size="sm"
                variant="outline"
              >
                <RotateCcw className="w-4 h-4 mr-1" /> Reset
              </Button>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between bg-slate-50 p-2 rounded">
                <span className="text-muted-foreground">Epoch:</span>
                <span className="font-mono">{epoch}</span>
              </div>
              <div className="flex justify-between bg-slate-50 p-2 rounded">
                <span className="text-muted-foreground">Loss (MSE):</span>
                <span className="font-mono">{currentLoss.toFixed(6)}</span>
              </div>
              <div className="flex justify-between bg-slate-50 p-2 rounded">
                <span className="text-muted-foreground">Learning Rate:</span>
                <span className="font-mono">{learningRate}</span>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold mb-2">Predictions:</h4>
              <div className="space-y-2">
                {predictions.map((p, i) => {
                  const isCorrect = Math.abs(p.pred - p.y) < 0.1;
                  return (
                    <div
                      key={i}
                      className={`p-2 rounded flex justify-between items-center ${
                        isCorrect ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
                      }`}
                    >
                      <span className="font-mono text-sm">
                        [{p.x[0]}, {p.x[1]}] XOR
                      </span>
                      <div className="flex gap-2 items-center">
                        <span className="text-xs text-muted-foreground">→</span>
                        <span className="font-mono">{p.pred.toFixed(3)}</span>
                        <span className="text-xs text-muted-foreground">(target: {p.y})</span>
                        {isCorrect && <span className="text-green-600">✓</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <h4 className="text-sm font-semibold mb-2">Training Loss Curve</h4>
              <canvas
                ref={canvasRef}
                width={400}
                height={200}
                className="w-full h-auto border border-slate-200 rounded bg-white"
              />
            </div>

            <div className="text-sm space-y-2 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <h4 className="font-semibold text-yellow-900">Why MLP Solves XOR:</h4>
              <ul className="list-disc pl-5 space-y-1 text-yellow-800 text-xs">
                <li><strong>Hidden layer</strong> learns intermediate features</li>
                <li>First neuron might learn "x₁ OR x₂"</li>
                <li>Second neuron might learn "x₁ AND x₂"</li>
                <li>Output combines these: (x₁ OR x₂) AND NOT(x₁ AND x₂)</li>
                <li>This creates a non-linear decision boundary!</li>
              </ul>
            </div>

            <div className="text-xs text-muted-foreground bg-slate-50 p-3 rounded">
              <strong>Truth Table:</strong>
              <table className="w-full mt-2">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-1">x₁</th>
                    <th className="text-left py-1">x₂</th>
                    <th className="text-left py-1">XOR</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  <tr><td>0</td><td>0</td><td>0</td></tr>
                  <tr><td>0</td><td>1</td><td>1</td></tr>
                  <tr><td>1</td><td>0</td><td>1</td></tr>
                  <tr><td>1</td><td>1</td><td>0</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Component 4: Backpropagation Visualizer
// ==========================================
function BackpropagationVisualizer() {
  const [showForward, setShowForward] = useState(true);

  // Simple example: single training sample
  const input = [0.5, 0.3];
  const target = 0.8;

  // Network parameters
  const w1 = [[0.5, -0.3], [0.2, 0.4]];
  const b1 = [0.1, -0.1];
  const w2 = [[0.6], [-0.4]];
  const b2 = [0.2];

  // Forward pass
  const z1_0 = w1[0][0] * input[0] + w1[0][1] * input[1] + b1[0];
  const z1_1 = w1[1][0] * input[0] + w1[1][1] * input[1] + b1[1];
  const a1_0 = activations.sigmoid(z1_0);
  const a1_1 = activations.sigmoid(z1_1);

  const z2 = w2[0][0] * a1_0 + w2[1][0] * a1_1 + b2[0];
  const output = activations.sigmoid(z2);

  const loss = Math.pow(output - target, 2) / 2;

  // Backward pass (gradients)
  const dL_dOutput = output - target;
  const dOutput_dZ2 = activationDerivatives.sigmoid(z2);
  const dL_dZ2 = dL_dOutput * dOutput_dZ2;

  const dL_dW2_0 = dL_dZ2 * a1_0;
  const dL_dW2_1 = dL_dZ2 * a1_1;
  const dL_dB2 = dL_dZ2;

  const dL_dA1_0 = dL_dZ2 * w2[0][0];
  const dL_dA1_1 = dL_dZ2 * w2[1][0];

  const dA1_dZ1_0 = activationDerivatives.sigmoid(z1_0);
  const dA1_dZ1_1 = activationDerivatives.sigmoid(z1_1);

  const dL_dZ1_0 = dL_dA1_0 * dA1_dZ1_0;
  const dL_dZ1_1 = dL_dA1_1 * dA1_dZ1_1;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingDown className="w-5 h-5" />
          Backpropagation: The Learning Algorithm
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          <strong>Backpropagation</strong> is how neural networks learn. It computes gradients (how much each
          weight affects the loss) by using the chain rule, flowing error backward through the network.
        </p>

        <div className="flex gap-2 mb-4">
          <Button
            variant={showForward ? 'default' : 'outline'}
            size="sm"
            onClick={() => setShowForward(true)}
          >
            <Eye className="w-4 h-4 mr-1" /> Forward Pass
          </Button>
          <Button
            variant={!showForward ? 'default' : 'outline'}
            size="sm"
            onClick={() => setShowForward(false)}
          >
            <TrendingDown className="w-4 h-4 mr-1" /> Backward Pass
          </Button>
        </div>

        {showForward ? (
          <div className="space-y-3">
            <h3 className="font-semibold">Forward Pass (Prediction)</h3>
            <div className="space-y-2 font-mono text-sm bg-blue-50 p-4 rounded-lg">
              <div>Input: [{input[0]}, {input[1]}]</div>
              <div className="text-xs text-muted-foreground">↓ Hidden layer computation</div>
              <div>z₁₀ = {z1_0.toFixed(4)}, a₁₀ = σ(z₁₀) = {a1_0.toFixed(4)}</div>
              <div>z₁₁ = {z1_1.toFixed(4)}, a₁₁ = σ(z₁₁) = {a1_1.toFixed(4)}</div>
              <div className="text-xs text-muted-foreground">↓ Output layer computation</div>
              <div>z₂ = {z2.toFixed(4)}</div>
              <div className="text-lg font-bold">ŷ = σ(z₂) = {output.toFixed(4)}</div>
              <div className="border-t pt-2 mt-2">
                <div>Target: y = {target}</div>
                <div className="text-lg font-bold text-red-600">Loss = (ŷ - y)² / 2 = {loss.toFixed(6)}</div>
              </div>
            </div>

            <div className="text-sm space-y-2 bg-green-50 border border-green-200 rounded-lg p-3">
              <h4 className="font-semibold text-green-900">Forward Pass Steps:</h4>
              <ol className="list-decimal pl-5 space-y-1 text-green-800 text-xs">
                <li>Compute weighted sum: z = W·x + b</li>
                <li>Apply activation: a = σ(z)</li>
                <li>Repeat for each layer</li>
                <li>Calculate loss: L = (ŷ - y)²</li>
              </ol>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <h3 className="font-semibold">Backward Pass (Learning)</h3>
            <div className="space-y-2 font-mono text-sm bg-red-50 p-4 rounded-lg">
              <div className="text-lg font-bold">∂L/∂ŷ = ŷ - y = {dL_dOutput.toFixed(4)}</div>
              <div className="text-xs text-muted-foreground">↓ Chain rule through output activation</div>
              <div>∂L/∂z₂ = {dL_dZ2.toFixed(4)}</div>
              <div className="text-xs text-muted-foreground">↓ Gradients for output weights/bias</div>
              <div>∂L/∂w₂₀ = {dL_dW2_0.toFixed(4)}</div>
              <div>∂L/∂w₂₁ = {dL_dW2_1.toFixed(4)}</div>
              <div>∂L/∂b₂ = {dL_dB2.toFixed(4)}</div>
              <div className="text-xs text-muted-foreground">↓ Propagate error to hidden layer</div>
              <div>∂L/∂a₁₀ = {dL_dA1_0.toFixed(4)}</div>
              <div>∂L/∂a₁₁ = {dL_dA1_1.toFixed(4)}</div>
              <div className="text-xs text-muted-foreground">↓ Chain rule through hidden activation</div>
              <div>∂L/∂z₁₀ = {dL_dZ1_0.toFixed(4)}</div>
              <div>∂L/∂z₁₁ = {dL_dZ1_1.toFixed(4)}</div>
            </div>

            <div className="text-sm space-y-2 bg-purple-50 border border-purple-200 rounded-lg p-3">
              <h4 className="font-semibold text-purple-900">Backpropagation Steps:</h4>
              <ol className="list-decimal pl-5 space-y-1 text-purple-800 text-xs">
                <li>Calculate output error: ∂L/∂ŷ</li>
                <li>Compute gradient at output layer</li>
                <li>Propagate error backward using chain rule</li>
                <li>Compute gradient at each hidden layer</li>
                <li>Update all weights: w_new = w_old - learning_rate × gradient</li>
              </ol>
            </div>

            <div className="text-xs text-muted-foreground bg-slate-50 p-3 rounded">
              <strong>The Chain Rule:</strong> To find how much w₁ affects L, we multiply all the derivatives
              along the path from L back to w₁. This is the essence of backpropagation!
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ==========================================
// Main Component with Tabs
// ==========================================
export default function Topic4MLPComplete() {
  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-2"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-slate-900 flex items-center justify-center gap-3">
            <Brain className="w-8 h-8 text-blue-600" />
            Topic 4: Multilayer Perceptron & Neural Network Learning
          </h1>
          <p className="text-slate-600 max-w-3xl mx-auto">
            Complete interactive guide: from MLP architecture to backpropagation—see how neural networks
            learn from the inside out!
          </p>
        </motion.div>

        <Tabs defaultValue="architecture" className="w-full">
          <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 gap-2">
            <TabsTrigger value="architecture">Architecture</TabsTrigger>
            <TabsTrigger value="forward">Forward Pass</TabsTrigger>
            <TabsTrigger value="xor">XOR Problem</TabsTrigger>
            <TabsTrigger value="backprop">Backpropagation</TabsTrigger>
          </TabsList>

          <TabsContent value="architecture" className="mt-6">
            <MLPArchitectureVisualizer />
          </TabsContent>

          <TabsContent value="forward" className="mt-6">
            <ForwardPropagationAnimator />
          </TabsContent>

          <TabsContent value="xor" className="mt-6">
            <XORProblemSolver />
          </TabsContent>

          <TabsContent value="backprop" className="mt-6">
            <BackpropagationVisualizer />
          </TabsContent>
        </Tabs>

        <footer className="text-center text-sm text-slate-500 pt-8 border-t">
          <p>
            COE 292 — Introduction to Artificial Intelligence •
            Part 1: Multilayer Perceptron • Part 2: Neural Network Learning
          </p>
        </footer>
      </div>
    </div>
  );
}
