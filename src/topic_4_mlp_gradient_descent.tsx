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
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Play, Pause, RotateCcw, Brain, Zap, Network, TrendingDown } from "lucide-react";

// ==========================================
// Topic 4: Neural Networks & Gradient Descent
// Based on CMU 11-785 Introduction to Deep Learning
// ==========================================

// Activation Functions
const activations = {
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
  tanh: (x: number) => Math.tanh(x),
  relu: (x: number) => Math.max(0, x),
  leakyRelu: (x: number) => x > 0 ? x : 0.01 * x,
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
  leakyRelu: (x: number) => (x > 0 ? 1 : 0.01),
};

// Utility functions
function linspace(start: number, end: number, n: number): number[] {
  const step = (end - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + i * step);
}

function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x));
}

// Generate synthetic dataset
function generateData(n: number, pattern: 'linear' | 'xor' | 'circle' | 'spiral') {
  const data: Array<{ x: number; y: number; label: number }> = [];

  for (let i = 0; i < n; i++) {
    let x: number, y: number, label: number;

    switch (pattern) {
      case 'linear':
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        label = x + y > 0 ? 1 : 0;
        break;

      case 'xor':
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        label = (x > 0) !== (y > 0) ? 1 : 0;
        break;

      case 'circle':
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        const r = Math.sqrt(x * x + y * y);
        label = r < 0.6 ? 1 : 0;
        break;

      case 'spiral':
        const theta = Math.random() * 4 * Math.PI;
        const radius = Math.random() * 0.8;
        x = radius * Math.cos(theta) * (1 + Math.random() * 0.1);
        y = radius * Math.sin(theta) * (1 + Math.random() * 0.1);
        label = Math.floor(theta / (2 * Math.PI)) % 2;
        break;

      default:
        x = 0;
        y = 0;
        label = 0;
    }

    data.push({ x, y, label });
  }

  return data;
}

// Simple MLP implementation
class MLP {
  weights1: number[][];
  bias1: number[];
  weights2: number[][];
  bias2: number[];
  hiddenSize: number;
  activation: keyof typeof activations;

  constructor(hiddenSize: number, activation: keyof typeof activations = 'sigmoid') {
    this.hiddenSize = hiddenSize;
    this.activation = activation;

    // Xavier initialization
    const limit1 = Math.sqrt(6 / (2 + hiddenSize));
    this.weights1 = Array(hiddenSize).fill(0).map(() =>
      Array(2).fill(0).map(() => (Math.random() * 2 - 1) * limit1)
    );
    this.bias1 = Array(hiddenSize).fill(0);

    const limit2 = Math.sqrt(6 / (hiddenSize + 1));
    this.weights2 = Array(1).fill(0).map(() =>
      Array(hiddenSize).fill(0).map(() => (Math.random() * 2 - 1) * limit2)
    );
    this.bias2 = [0];
  }

  forward(x: number[]): { output: number; hidden: number[]; z1: number[]; z2: number } {
    const act = activations[this.activation];

    // Hidden layer
    const z1 = this.weights1.map((w, i) =>
      w[0] * x[0] + w[1] * x[1] + this.bias1[i]
    );
    const hidden = z1.map(act);

    // Output layer
    const z2 = this.weights2[0].reduce((sum, w, i) => sum + w * hidden[i], 0) + this.bias2[0];
    const output = activations.sigmoid(z2); // Always sigmoid for binary classification

    return { output, hidden, z1, z2 };
  }

  backward(
    x: number[],
    y: number,
    learningRate: number,
    dropout: number = 0,
    dropoutMask?: boolean[]
  ): number {
    const { output, hidden, z1 } = this.forward(x);

    // Loss (binary cross-entropy)
    const loss = -y * Math.log(output + 1e-10) - (1 - y) * Math.log(1 - output + 1e-10);

    // Output layer gradients
    const dOutput = output - y;
    const dWeights2 = hidden.map(h => dOutput * h);
    const dBias2 = dOutput;

    // Hidden layer gradients
    const actDeriv = activationDerivatives[this.activation];
    const dHidden = hidden.map((h, i) => {
      const grad = dOutput * this.weights2[0][i] * actDeriv(z1[i]);
      // Apply dropout mask during training
      return (dropoutMask && !dropoutMask[i]) ? 0 : grad;
    });

    const dWeights1 = dHidden.map(dh => [dh * x[0], dh * x[1]]);
    const dBias1 = dHidden;

    // Update weights
    this.weights2[0] = this.weights2[0].map((w, i) => w - learningRate * dWeights2[i]);
    this.bias2[0] -= learningRate * dBias2;

    this.weights1 = this.weights1.map((w, i) => [
      w[0] - learningRate * dWeights1[i][0],
      w[1] - learningRate * dWeights1[i][1]
    ]);
    this.bias1 = this.bias1.map((b, i) => b - learningRate * dBias1[i]);

    return loss;
  }

  predict(x: number[]): number {
    return this.forward(x).output;
  }
}

// ==========================================
// Component 1: Activation Functions
// ==========================================
function ActivationDemo() {
  const [selectedAct, setSelectedAct] = useState<keyof typeof activations>('sigmoid');

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw axes
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.stroke();

    // Draw grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(0, h * i / 4);
      ctx.lineTo(w, h * i / 4);
      ctx.moveTo(w * i / 4, 0);
      ctx.lineTo(w * i / 4, h);
      ctx.stroke();
    }

    // Draw activation function
    const act = activations[selectedAct];
    const actDeriv = activationDerivatives[selectedAct];

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();

    const points = linspace(-5, 5, 200);
    points.forEach((x, i) => {
      const y = act(x);
      const px = (x / 10 + 0.5) * w;
      const py = (0.5 - y / 2) * h;

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });

    ctx.stroke();

    // Draw derivative
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();

    points.forEach((x, i) => {
      const y = actDeriv(x);
      const px = (x / 10 + 0.5) * w;
      const py = (0.5 - y / 2) * h;

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });

    ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('x', w - 10, h / 2 - 10);

    // Legend
    ctx.textAlign = 'left';
    ctx.fillStyle = '#3b82f6';
    ctx.fillText('σ(x) - Activation', 10, 30);
    ctx.fillStyle = '#10b981';
    ctx.fillText("σ'(x) - Derivative", 10, 50);

    // Add equation annotations on the canvas near the curves
    ctx.font = 'bold 14px system-ui';

    // Position annotations based on activation function type
    let actEquation = '';
    let derivEquation = '';
    let actX = 0, actY = 0, derivX = 0, derivY = 0;

    switch (selectedAct) {
      case 'sigmoid':
        actEquation = 'σ(x) = 1/(1+e⁻ˣ)';
        derivEquation = "σ'(x) = σ(x)(1-σ(x))";
        actX = w * 0.7;
        actY = h * 0.2;
        derivX = w * 0.5;
        derivY = h * 0.55;
        break;
      case 'tanh':
        actEquation = 'σ(x) = tanh(x)';
        derivEquation = "σ'(x) = 1-tanh²(x)";
        actX = w * 0.7;
        actY = h * 0.35;
        derivX = w * 0.5;
        derivY = h * 0.55;
        break;
      case 'relu':
        actEquation = 'σ(x) = max(0,x)';
        derivEquation = "σ'(x) = 1 if x>0 else 0";
        actX = w * 0.65;
        actY = h * 0.25;
        derivX = w * 0.55;
        derivY = h * 0.48;
        break;
      case 'leakyRelu':
        actEquation = 'σ(x) = max(0.01x, x)';
        derivEquation = "σ'(x) = 1 if x>0 else 0.01";
        actX = w * 0.6;
        actY = h * 0.25;
        derivX = w * 0.55;
        derivY = h * 0.48;
        break;
    }

    // Draw activation function equation (blue)
    ctx.fillStyle = '#3b82f6';
    ctx.textAlign = 'left';
    ctx.fillText(actEquation, actX, actY);

    // Draw derivative equation (green)
    ctx.fillStyle = '#10b981';
    ctx.fillText(derivEquation, derivX, derivY);

  }, [selectedAct]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Activation Functions - The "Soft" Perceptron
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Unlike hard perceptrons (step functions), neural networks use smooth, differentiable activation
          functions. This allows gradient-based optimization. The <strong className="text-blue-600">solid line</strong> shows the activation function σ(x),
          and the <strong className="text-green-600">dashed line</strong> shows its derivative σ'(x) used in backpropagation.
        </p>

        <div className="flex flex-wrap gap-2">
          {(Object.keys(activations) as Array<keyof typeof activations>).map(act => (
            <Button
              key={act}
              variant={selectedAct === act ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedAct(act)}
            >
              {act}
            </Button>
          ))}
        </div>

        <div className="bg-slate-50 rounded-lg p-4">
          <canvas
            ref={canvasRef}
            width={600}
            height={300}
            className="w-full h-auto border border-slate-200 rounded"
          />
        </div>

        <div className="text-sm space-y-2">
          <h4 className="font-semibold">Key Properties:</h4>
          <ul className="list-disc pl-5 space-y-1 text-muted-foreground text-xs">
            <li><strong>Sigmoid:</strong> σ(x) = 1/(1+e⁻ˣ), σ'(x) = σ(x)(1-σ(x)). Outputs ∈ (0,1), derivative saturates at extremes → vanishing gradients</li>
            <li><strong>Tanh:</strong> σ(x) = tanh(x), σ'(x) = 1-tanh²(x). Outputs ∈ (-1,1), zero-centered, derivative still saturates</li>
            <li><strong>ReLU:</strong> σ(x) = max(0,x), σ'(x) = 1 if x&gt;0 else 0. Most popular, no saturation for x&gt;0, but "dead neurons" if x≤0</li>
            <li><strong>Leaky ReLU:</strong> σ(x) = max(0.01x, x), σ'(x) = 1 if x&gt;0 else 0.01. Fixes dying ReLU with small negative slope</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Component 2: MLP as Universal Classifier
// ==========================================
function UniversalClassifier() {
  const [pattern, setPattern] = useState<'linear' | 'xor' | 'circle' | 'spiral'>('xor');
  const [hiddenSize, setHiddenSize] = useState(4);
  const [activation, setActivation] = useState<keyof typeof activations>('relu');
  const [data, setData] = useState(() => generateData(200, 'xor'));
  const [model, setModel] = useState<MLP | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [training, setTraining] = useState(false);
  const [loss, setLoss] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize model when parameters change
  useEffect(() => {
    setModel(new MLP(hiddenSize, activation));
    setEpoch(0);
    setLoss(0);
  }, [hiddenSize, activation]);

  // Generate new data when pattern changes
  useEffect(() => {
    setData(generateData(200, pattern));
    setModel(new MLP(hiddenSize, activation));
    setEpoch(0);
    setLoss(0);
  }, [pattern]);

  // Training loop
  useEffect(() => {
    if (!training || !model) return;

    const interval = setInterval(() => {
      let totalLoss = 0;

      // One epoch: shuffle and train on all data
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      shuffled.forEach(point => {
        const l = model.backward([point.x, point.y], point.label, 0.1);
        totalLoss += l;
      });

      setLoss(totalLoss / data.length);
      setEpoch(e => e + 1);
    }, 50);

    return () => clearInterval(interval);
  }, [training, model, data]);

  // Render decision boundary
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !model) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw decision boundary heatmap
    const resolution = 50;
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = (i / resolution) * 2 - 1;
        const y = (j / resolution) * 2 - 1;
        const pred = model.predict([x, y]);

        const px = ((x + 1) / 2) * w;
        const py = ((1 - y) / 2) * h;

        ctx.fillStyle = pred > 0.5
          ? `rgba(59, 130, 246, ${pred * 0.3})`
          : `rgba(239, 68, 68, ${(1 - pred) * 0.3})`;
        ctx.fillRect(px, py, w / resolution + 1, h / resolution + 1);
      }
    }

    // Draw data points
    data.forEach(point => {
      const px = ((point.x + 1) / 2) * w;
      const py = ((1 - point.y) / 2) * h;

      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fillStyle = point.label === 1 ? '#3b82f6' : '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

  }, [model, data, epoch]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="w-5 h-5" />
          MLPs as Universal Classifiers
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          <strong>Universal Approximation Theorem:</strong> A neural network with even a single hidden layer
          can approximate any continuous function, given enough hidden units.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Dataset Pattern</label>
              <div className="flex flex-wrap gap-2 mt-1">
                {(['linear', 'xor', 'circle', 'spiral'] as const).map(p => (
                  <Button
                    key={p}
                    variant={pattern === p ? "default" : "outline"}
                    size="sm"
                    onClick={() => setPattern(p)}
                  >
                    {p.toUpperCase()}
                  </Button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-sm font-medium">Hidden Units: {hiddenSize}</label>
              <input
                type="range"
                min={2}
                max={16}
                value={hiddenSize}
                onChange={(e) => setHiddenSize(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-sm font-medium">Activation</label>
              <div className="flex flex-wrap gap-2 mt-1">
                {(['sigmoid', 'tanh', 'relu'] as const).map(act => (
                  <Button
                    key={act}
                    variant={activation === act ? "default" : "outline"}
                    size="sm"
                    onClick={() => setActivation(act)}
                  >
                    {act}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => setTraining(!training)}
                className="flex-1"
              >
                {training ? <><Pause className="w-4 h-4 mr-2" /> Pause</> : <><Play className="w-4 h-4 mr-2" /> Train</>}
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setModel(new MLP(hiddenSize, activation));
                  setEpoch(0);
                  setLoss(0);
                  setTraining(false);
                }}
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            <div className="text-sm space-y-1 bg-slate-50 p-3 rounded">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Epoch:</span>
                <span className="font-mono">{epoch}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Loss:</span>
                <span className="font-mono">{loss.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Architecture:</span>
                <span className="font-mono text-xs">2 → {hiddenSize} → 1</span>
              </div>
            </div>
          </div>

          <div>
            <canvas
              ref={canvasRef}
              width={400}
              height={400}
              className="w-full h-auto border border-slate-200 rounded bg-white"
            />
            <p className="text-xs text-muted-foreground mt-2">
              Blue regions = predicted class 1, Red regions = predicted class 0.
              Points show true labels.
            </p>
          </div>
        </div>

        <div className="text-sm space-y-2 bg-blue-50 border border-blue-200 rounded-lg p-3">
          <h4 className="font-semibold text-blue-900">Observations:</h4>
          <ul className="list-disc pl-5 space-y-1 text-blue-800">
            <li><strong>Linear:</strong> Solvable even without hidden layer</li>
            <li><strong>XOR:</strong> Classic non-linearly separable problem, needs hidden layer</li>
            <li><strong>Circle:</strong> Demonstrates curved decision boundaries</li>
            <li><strong>Spiral:</strong> Highly non-linear, needs more hidden units or deeper network</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Component 3: Gradient Descent Variants
// ==========================================
function GradientDescentVariants() {
  const [mode, setMode] = useState<'bgd' | 'sgd' | 'minibatch'>('minibatch');
  const [batchSize, setBatchSize] = useState(16);
  const [data] = useState(() => generateData(100, 'circle'));

  const [modelBGD, setModelBGD] = useState<MLP>(() => new MLP(4, 'relu'));
  const [modelSGD, setModelSGD] = useState<MLP>(() => new MLP(4, 'relu'));
  const [modelMB, setModelMB] = useState<MLP>(() => new MLP(4, 'relu'));

  const [lossBGD, setLossBGD] = useState<number[]>([]);
  const [lossSGD, setLossSGD] = useState<number[]>([]);
  const [lossMB, setLossMB] = useState<number[]>([]);

  const [training, setTraining] = useState(false);
  const [iteration, setIteration] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Training loop
  useEffect(() => {
    if (!training) return;

    const interval = setInterval(() => {
      // Batch Gradient Descent
      let totalLossBGD = 0;
      data.forEach(point => {
        totalLossBGD += modelBGD.backward([point.x, point.y], point.label, 0.1);
      });
      setLossBGD(prev => [...prev, totalLossBGD / data.length].slice(-100));

      // Stochastic Gradient Descent
      const randomPoint = data[Math.floor(Math.random() * data.length)];
      const lossSGDVal = modelSGD.backward([randomPoint.x, randomPoint.y], randomPoint.label, 0.01);
      setLossSGD(prev => [...prev, lossSGDVal].slice(-100));

      // Mini-batch Gradient Descent
      let totalLossMB = 0;
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      const batch = shuffled.slice(0, batchSize);
      batch.forEach(point => {
        totalLossMB += modelMB.backward([point.x, point.y], point.label, 0.05);
      });
      setLossMB(prev => [...prev, totalLossMB / batch.length].slice(-100));

      setIteration(i => i + 1);
    }, 100);

    return () => clearInterval(interval);
  }, [training, data, batchSize]);

  // Draw loss curves
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i / 5) * h);
      ctx.lineTo(w, (i / 5) * h);
      ctx.stroke();
    }

    // Draw loss curves
    const drawCurve = (losses: number[], color: string) => {
      if (losses.length < 2) return;

      const maxLoss = Math.max(...losses, 1);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      losses.forEach((loss, i) => {
        const x = (i / 100) * w;
        const y = h - (loss / maxLoss) * h * 0.9;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.stroke();
    };

    drawCurve(lossBGD, '#3b82f6'); // Blue - BGD
    drawCurve(lossSGD, '#ef4444'); // Red - SGD
    drawCurve(lossMB, '#10b981');  // Green - Mini-batch

    // Labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '12px system-ui';
    ctx.fillText('Loss', 10, 20);
    ctx.fillText('Iteration →', w - 80, h - 10);

  }, [lossBGD, lossSGD, lossMB]);

  const reset = () => {
    const seed = Math.random();
    Math.random = (() => {
      let s = seed;
      return () => {
        s = (s * 9301 + 49297) % 233280;
        return s / 233280;
      };
    })();

    setModelBGD(new MLP(4, 'relu'));
    setModelSGD(new MLP(4, 'relu'));
    setModelMB(new MLP(4, 'relu'));
    setLossBGD([]);
    setLossSGD([]);
    setLossMB([]);
    setIteration(0);
    setTraining(false);

    Math.random = () => Math.random();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingDown className="w-5 h-5" />
          Gradient Descent Variants
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Compare how different gradient descent algorithms converge. All start with identical weights.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Mini-batch Size: {batchSize}</label>
              <input
                type="range"
                min={1}
                max={50}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full"
                disabled={training}
              />
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => setTraining(!training)}
                className="flex-1"
              >
                {training ? <><Pause className="w-4 h-4 mr-2" /> Pause</> : <><Play className="w-4 h-4 mr-2" /> Train</>}
              </Button>
              <Button variant="outline" onClick={reset}>
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-blue-500 rounded"></div>
                <div>
                  <strong>Batch GD:</strong> Uses entire dataset per update
                  <div className="text-xs text-muted-foreground">
                    Smooth, stable, but slow for large datasets
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <div>
                  <strong>Stochastic GD:</strong> One sample per update
                  <div className="text-xs text-muted-foreground">
                    Noisy, fast, can escape local minima
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <div>
                  <strong>Mini-batch GD:</strong> Small batch per update
                  <div className="text-xs text-muted-foreground">
                    Best trade-off: stable + efficient + parallelizable
                  </div>
                </div>
              </div>
            </div>

            <div className="text-sm space-y-1 bg-slate-50 p-3 rounded">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Iteration:</span>
                <span className="font-mono">{iteration}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Dataset size:</span>
                <span className="font-mono">{data.length}</span>
              </div>
            </div>
          </div>

          <div>
            <canvas
              ref={canvasRef}
              width={500}
              height={300}
              className="w-full h-auto border border-slate-200 rounded bg-white"
            />
            <p className="text-xs text-muted-foreground mt-2">
              Training loss over iterations. Notice SGD's noise vs BGD's smoothness.
            </p>
          </div>
        </div>

        <div className="text-sm space-y-2 bg-amber-50 border border-amber-200 rounded-lg p-3">
          <h4 className="font-semibold text-amber-900">In Practice (Deep Learning):</h4>
          <ul className="list-disc pl-5 space-y-1 text-amber-800">
            <li><strong>Mini-batch GD</strong> is the standard (batch sizes: 32, 64, 128, 256)</li>
            <li>Combined with momentum, Adam, or other optimizers</li>
            <li>Batch size affects GPU memory usage and convergence</li>
            <li>Larger batches → more stable but may generalize worse</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Component 4: Overfitting & Dropout
// ==========================================
function OverfittingDropout() {
  const [data] = useState(() => generateData(50, 'circle'));
  const [testData] = useState(() => generateData(100, 'circle'));

  const [modelNoReg, setModelNoReg] = useState<MLP>(() => new MLP(16, 'relu'));
  const [modelDropout, setModelDropout] = useState<MLP>(() => new MLP(16, 'relu'));

  const [dropoutRate, setDropoutRate] = useState(0.5);
  const [training, setTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);

  const [trainLossNoReg, setTrainLossNoReg] = useState<number[]>([]);
  const [testLossNoReg, setTestLossNoReg] = useState<number[]>([]);
  const [trainLossDropout, setTrainLossDropout] = useState<number[]>([]);
  const [testLossDropout, setTestLossDropout] = useState<number[]>([]);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Training loop
  useEffect(() => {
    if (!training) return;

    const interval = setInterval(() => {
      // Train without regularization
      let trainLoss1 = 0;
      data.forEach(point => {
        trainLoss1 += modelNoReg.backward([point.x, point.y], point.label, 0.1);
      });

      // Train with dropout
      let trainLoss2 = 0;
      data.forEach(point => {
        const dropoutMask = Array(16).fill(0).map(() => Math.random() > dropoutRate);
        trainLoss2 += modelDropout.backward([point.x, point.y], point.label, 0.1, dropoutRate, dropoutMask);
      });

      // Evaluate on test set
      let testLoss1 = 0;
      let testLoss2 = 0;
      testData.forEach(point => {
        const pred1 = modelNoReg.predict([point.x, point.y]);
        const pred2 = modelDropout.predict([point.x, point.y]);

        testLoss1 += -point.label * Math.log(pred1 + 1e-10) - (1 - point.label) * Math.log(1 - pred1 + 1e-10);
        testLoss2 += -point.label * Math.log(pred2 + 1e-10) - (1 - point.label) * Math.log(1 - pred2 + 1e-10);
      });

      setTrainLossNoReg(prev => [...prev, trainLoss1 / data.length].slice(-100));
      setTestLossNoReg(prev => [...prev, testLoss1 / testData.length].slice(-100));
      setTrainLossDropout(prev => [...prev, trainLoss2 / data.length].slice(-100));
      setTestLossDropout(prev => [...prev, testLoss2 / testData.length].slice(-100));

      setEpoch(e => e + 1);
    }, 100);

    return () => clearInterval(interval);
  }, [training, data, testData, dropoutRate]);

  // Draw loss curves
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i / 5) * h);
      ctx.lineTo(w, (i / 5) * h);
      ctx.stroke();
    }

    // Draw curves
    const drawCurve = (losses: number[], color: string, dashed: boolean = false) => {
      if (losses.length < 2) return;

      const maxLoss = Math.max(...trainLossNoReg, ...testLossNoReg, ...trainLossDropout, ...testLossDropout, 1);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) ctx.setLineDash([5, 5]);
      else ctx.setLineDash([]);

      ctx.beginPath();

      losses.forEach((loss, i) => {
        const x = (i / 100) * w;
        const y = h - (loss / maxLoss) * h * 0.9;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.stroke();
    };

    drawCurve(trainLossNoReg, '#3b82f6', false);    // Solid blue - train no reg
    drawCurve(testLossNoReg, '#3b82f6', true);     // Dashed blue - test no reg
    drawCurve(trainLossDropout, '#10b981', false); // Solid green - train dropout
    drawCurve(testLossDropout, '#10b981', true);   // Dashed green - test dropout

    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '12px system-ui';
    ctx.fillText('Loss', 10, 20);
    ctx.fillText('Epoch →', w - 60, h - 10);

  }, [trainLossNoReg, testLossNoReg, trainLossDropout, testLossDropout]);

  const reset = () => {
    setModelNoReg(new MLP(16, 'relu'));
    setModelDropout(new MLP(16, 'relu'));
    setTrainLossNoReg([]);
    setTestLossNoReg([]);
    setTrainLossDropout([]);
    setTestLossDropout([]);
    setEpoch(0);
    setTraining(false);
  };

  const gapNoReg = testLossNoReg.length > 0 && trainLossNoReg.length > 0
    ? testLossNoReg[testLossNoReg.length - 1] - trainLossNoReg[trainLossNoReg.length - 1]
    : 0;

  const gapDropout = testLossDropout.length > 0 && trainLossDropout.length > 0
    ? testLossDropout[testLossDropout.length - 1] - trainLossDropout[trainLossDropout.length - 1]
    : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Overfitting & Dropout Regularization
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          <strong>Overfitting</strong> occurs when a model learns training data too well, including noise.
          <strong> Dropout</strong> randomly disables neurons during training to prevent co-adaptation.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Dropout Rate: {dropoutRate.toFixed(2)}</label>
              <input
                type="range"
                min={0}
                max={0.8}
                step={0.1}
                value={dropoutRate}
                onChange={(e) => setDropoutRate(Number(e.target.value))}
                className="w-full"
                disabled={training}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Probability of dropping each neuron (0 = no dropout, 0.5 typical)
              </p>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => setTraining(!training)}
                className="flex-1"
              >
                {training ? <><Pause className="w-4 h-4 mr-2" /> Pause</> : <><Play className="w-4 h-4 mr-2" /> Train</>}
              </Button>
              <Button variant="outline" onClick={reset}>
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-12 h-0.5 bg-blue-500"></div>
                <span>No Regularization (Train)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-12 h-0.5 bg-blue-500" style={{ borderTop: '2px dashed #3b82f6', background: 'none' }}></div>
                <span>No Regularization (Test)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-12 h-0.5 bg-green-500"></div>
                <span>With Dropout (Train)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-12 h-0.5 bg-green-500" style={{ borderTop: '2px dashed #10b981', background: 'none' }}></div>
                <span>With Dropout (Test)</span>
              </div>
            </div>

            <div className="text-sm space-y-1 bg-slate-50 p-3 rounded">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Epoch:</span>
                <span className="font-mono">{epoch}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Train size:</span>
                <span className="font-mono">{data.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Test size:</span>
                <span className="font-mono">{testData.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Gap (No Reg):</span>
                <span className={`font-mono ${gapNoReg > 0.2 ? 'text-red-600' : ''}`}>
                  {gapNoReg.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Gap (Dropout):</span>
                <span className={`font-mono ${gapDropout > 0.2 ? 'text-red-600' : 'text-green-600'}`}>
                  {gapDropout.toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          <div>
            <canvas
              ref={canvasRef}
              width={500}
              height={300}
              className="w-full h-auto border border-slate-200 rounded bg-white"
            />
            <p className="text-xs text-muted-foreground mt-2">
              <strong>Overfitting signature:</strong> Train loss ↓ but test loss ↑ or stagnates.
              Dropout reduces this gap.
            </p>
          </div>
        </div>

        <div className="text-sm space-y-2 bg-purple-50 border border-purple-200 rounded-lg p-3">
          <h4 className="font-semibold text-purple-900">Why Dropout Works:</h4>
          <ul className="list-disc pl-5 space-y-1 text-purple-800">
            <li>Forces network to learn redundant representations</li>
            <li>Prevents complex co-adaptations between neurons</li>
            <li>Equivalent to training an ensemble of networks</li>
            <li>At test time, use all neurons but scale weights by (1 - dropout_rate)</li>
            <li>Modern alternatives: Batch Normalization, Layer Normalization</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Main Component
// ==========================================
export default function Topic4MLPGradientDescent() {
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
            Topic 4: Neural Networks & Gradient Descent
          </h1>
          <p className="text-slate-600 max-w-3xl mx-auto">
            Based on CMU 11-785: Introduction to Deep Learning — Explore activation functions,
            universal approximation, gradient descent variants, and regularization techniques
          </p>
        </motion.div>

        <Tabs defaultValue="activation" className="w-full">
          <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 gap-2">
            <TabsTrigger value="activation">Activation Functions</TabsTrigger>
            <TabsTrigger value="universal">Universal Approximation</TabsTrigger>
            <TabsTrigger value="gradient">Gradient Descent</TabsTrigger>
            <TabsTrigger value="overfitting">Overfitting & Dropout</TabsTrigger>
          </TabsList>

          <TabsContent value="activation" className="mt-6">
            <ActivationDemo />
          </TabsContent>

          <TabsContent value="universal" className="mt-6">
            <UniversalClassifier />
          </TabsContent>

          <TabsContent value="gradient" className="mt-6">
            <GradientDescentVariants />
          </TabsContent>

          <TabsContent value="overfitting" className="mt-6">
            <OverfittingDropout />
          </TabsContent>
        </Tabs>

        <footer className="text-center text-sm text-slate-500 pt-8 border-t">
          <p>
            COE 292 — Introduction to Artificial Intelligence •
            Inspired by CMU 11-785 (Bhiksha Raj & Rita Singh)
          </p>
        </footer>
      </div>
    </div>
  );
}