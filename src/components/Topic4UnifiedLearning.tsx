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
import { Play, Pause, RotateCcw, StepForward, Brain, Target, TrendingDown, Zap, Link2, BookOpen } from "lucide-react";

// ==========================================
// Topic 4: Unified Learning Journey
// From MLP Architecture → Backpropagation → Gradient Descent
// ==========================================

// Activation functions
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
// Level 1: Single Weight - The Simplest Case
// ==========================================
function Level1SingleWeight() {
  const [weight, setWeight] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<Array<{ weight: number; loss: number }>>([]);

  // Simple training data: x=2, target=6 (so ideal weight is 3)
  const x = 2;
  const target = 6;
  const learningRate = 0.1;

  // Forward pass
  const prediction = weight * x;
  const error = prediction - target;
  const loss = (error * error); // Cost function: (ŷ - y)² (following 3Blue1Brown notation)

  // Gradient (partial derivative using chain rule)
  const gradient = 2 * error * x; // ∂C/∂w = ∂C/∂ŷ × ∂ŷ/∂w = 2(ŷ-y) × x

  // Training step
  const trainStep = () => {
    const newWeight = weight - learningRate * gradient;
    setWeight(newWeight);
    setStep(s => s + 1);
    setHistory(h => [...h, { weight: newWeight, loss: (newWeight * x - target) ** 2 / 2 }].slice(-50));
  };

  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(trainStep, 300);
    return () => clearInterval(interval);
  }, [playing, weight]);

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

    // Draw quadratic loss function
    const weights = [];
    const losses = [];
    for (let testW = -1; testW <= 5; testW += 0.1) {
      weights.push(testW);
      const testLoss = ((testW * x - target) ** 2) / 2;
      losses.push(testLoss);
    }

    const maxLoss = Math.max(...losses);
    const minW = Math.min(...weights);
    const maxW = Math.max(...weights);

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();

    weights.forEach((testW, i) => {
      const px = ((testW - minW) / (maxW - minW)) * w;
      const py = h - (losses[i] / maxLoss) * h * 0.9;

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });

    ctx.stroke();

    // Draw current position
    const cx = ((weight - minW) / (maxW - minW)) * w;
    const cy = h - (loss / maxLoss) * h * 0.9;

    ctx.fillStyle = '#ef4444';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Draw descent direction arrow (−∇C)
    // The negative gradient points downhill
    const arrowScale = 20;
    const arrowDx = -gradient * arrowScale; // Negative gradient direction
    const arrowDy = Math.abs(gradient) * arrowScale * 2; // Visual scaling for slope

    // Only draw arrow if gradient is significant
    if (Math.abs(gradient) > 0.01) {
      ctx.strokeStyle = '#10b981'; // Green descent arrow
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + arrowDx, cy + arrowDy);
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(arrowDy, arrowDx);
      const headSize = 10;
      ctx.fillStyle = '#10b981'; // Green arrowhead
      ctx.beginPath();
      ctx.moveTo(cx + arrowDx, cy + arrowDy);
      ctx.lineTo(
        cx + arrowDx - headSize * Math.cos(angle - Math.PI / 6),
        cy + arrowDy - headSize * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        cx + arrowDx - headSize * Math.cos(angle + Math.PI / 6),
        cy + arrowDy - headSize * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '14px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Weight →', w / 2, h - 10);
    ctx.fillText('Loss', 30, 20);

  }, [weight, loss, gradient]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="w-5 h-5" />
          Level 1: Single Weight - Understanding the Basics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          <strong>The simplest case:</strong> One weight, one input, one output. This is where gradient descent begins!
          We want to find the weight that makes <code>weight × 2 = 6</code>.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-900 mb-3">The Problem:</h4>
              <div className="space-y-2 font-mono text-sm">
                <div>Input: x = {x}</div>
                <div>Target: y = {target}</div>
                <div className="text-lg font-bold text-blue-700">Find: w such that w × x = y</div>
              </div>
            </div>

            <div className="space-y-2 bg-slate-50 p-3 rounded">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Current weight:</span>
                <span className="font-mono font-bold">{weight.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Prediction (w×x):</span>
                <span className="font-mono">{prediction.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Error (pred-target):</span>
                <span className="font-mono text-red-600">{error.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Cost C = (ŷ-y)²:</span>
                <span className="font-mono font-bold text-red-600">{loss.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm border-t pt-2">
                <span className="text-muted-foreground">Gradient (∂C/∂w):</span>
                <span className="font-mono text-green-600 font-bold">{gradient.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Steps taken:</span>
                <span className="font-mono">{step}</span>
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={() => setPlaying(!playing)} size="sm">
                {playing ? <><Pause className="w-4 h-4 mr-1" /> Pause</> : <><Play className="w-4 h-4 mr-1" /> Play</>}
              </Button>
              <Button onClick={trainStep} size="sm" variant="outline">
                <StepForward className="w-4 h-4 mr-1" /> Step
              </Button>
              <Button
                onClick={() => {
                  setWeight(0.5);
                  setStep(0);
                  setHistory([]);
                  setPlaying(false);
                }}
                size="sm"
                variant="outline"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            <div className="text-sm space-y-2 bg-green-50 border border-green-200 rounded-lg p-3">
              <h4 className="font-semibold text-green-900">Gradient Descent Update Rule:</h4>
              <div className="font-mono text-xs text-green-800 space-y-1">
                <div className="font-bold mb-1">w<sub>new</sub> = w<sub>old</sub> - η × ∂C/∂w</div>
                <div className="text-muted-foreground">(η = learning rate, controls step size)</div>
                <div className="mt-2">w<sub>new</sub> = {weight.toFixed(3)} - 0.1 × {gradient.toFixed(3)}</div>
                <div className="text-sm font-bold">w<sub>new</sub> = {(weight - learningRate * gradient).toFixed(3)}</div>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <h4 className="text-sm font-semibold">Loss Landscape (Parabola)</h4>
            <canvas
              ref={canvasRef}
              width={400}
              height={300}
              className="w-full h-auto border border-slate-200 rounded bg-white"
            />
            <p className="text-xs text-muted-foreground">
              <strong>Red ball</strong> = current weight on the curve, <strong>Green arrow</strong> = descent direction (−∇C).
              The arrow points downhill toward the minimum!
            </p>
          </div>
        </div>

        <div className="text-sm space-y-2 bg-amber-50 border border-amber-200 rounded-lg p-3">
          <h4 className="font-semibold text-amber-900">Key Insights:</h4>
          <ul className="list-disc pl-5 space-y-1 text-amber-800">
            <li><strong>Gradient = slope:</strong> Tells us the direction and steepness of the loss curve</li>
            <li><strong>Positive gradient:</strong> Moving right increases loss → go left (subtract)</li>
            <li><strong>Negative gradient:</strong> Moving left increases loss → go right (add)</li>
            <li><strong>Gradient → 0:</strong> We've reached the minimum! (optimal weight ≈ 3)</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Level 2: Bridge - From Backprop to Gradient Descent
// ==========================================
function Level2Bridge() {
  const [input] = useState([0.5, 0.3]);
  const [target] = useState(0.8);
  const [step, setStep] = useState(0);

  // Network parameters
  const [w1, setW1] = useState([[0.5, -0.3], [0.2, 0.4]]);
  const [b1, setB1] = useState([0.1, -0.1]);
  const [w2, setW2] = useState([[0.6], [-0.4]]);
  const [b2, setB2] = useState([0.2]);

  const learningRate = 0.5;

  // Forward pass
  const z1_0 = w1[0][0] * input[0] + w1[0][1] * input[1] + b1[0];
  const z1_1 = w1[1][0] * input[0] + w1[1][1] * input[1] + b1[1];
  const a1_0 = activations.sigmoid(z1_0);
  const a1_1 = activations.sigmoid(z1_1);

  const z2 = w2[0][0] * a1_0 + w2[1][0] * a1_1 + b2[0];
  const output = activations.sigmoid(z2);

  const loss = Math.pow(output - target, 2); // Cost for single example: C₀ = (a⁽ᴸ⁾ - y)²

  // Backward pass (gradients) - Following 3Blue1Brown chain rule notation
  const dL_dOutput = 2 * (output - target); // ∂C₀/∂a⁽ᴸ⁾ = 2(a⁽ᴸ⁾ - y)
  const dOutput_dZ2 = activationDerivatives.sigmoid(z2); // ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = σ'(z⁽ᴸ⁾)
  const dL_dZ2 = dL_dOutput * dOutput_dZ2; // Chain rule: ∂C₀/∂z⁽ᴸ⁾

  const dL_dW2_0 = dL_dZ2 * a1_0; // ∂C₀/∂w⁽ᴸ⁾ = ∂z⁽ᴸ⁾/∂w⁽ᴸ⁾ × ∂C₀/∂z⁽ᴸ⁾ = a⁽ᴸ⁻¹⁾ × ∂C₀/∂z⁽ᴸ⁾
  const dL_dW2_1 = dL_dZ2 * a1_1; // Same for other weight
  const dL_dB2 = dL_dZ2; // ∂C₀/∂b⁽ᴸ⁾ = ∂C₀/∂z⁽ᴸ⁾

  // Training step
  const trainStep = () => {
    // Update output layer (just showing w2[0][0] for clarity)
    const newW2_0_0 = w2[0][0] - learningRate * dL_dW2_0;
    const newW2_1_0 = w2[1][0] - learningRate * dL_dW2_1;
    const newB2 = b2[0] - learningRate * dL_dB2;

    setW2([[newW2_0_0], [newW2_1_0]]);
    setB2([newB2]);
    setStep(s => s + 1);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Link2 className="w-5 h-5" />
          Level 2: The Connection - Backpropagation IS Gradient Descent!
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
          <h3 className="text-lg font-bold text-purple-900 mb-2 flex items-center gap-2">
            <Zap className="w-5 h-5" />
            The Big Revelation
          </h3>
          <p className="text-purple-800">
            <strong>Backpropagation is just gradient descent applied to neural networks!</strong>
            The "backward pass" computes gradients for ALL weights at once, then we use the SAME update rule.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <h4 className="font-semibold">Forward Pass (Make Prediction)</h4>
            <div className="bg-blue-50 border border-blue-200 rounded p-3 font-mono text-xs space-y-1">
              <div>Input: [{input[0]}, {input[1]}]</div>
              <div className="text-muted-foreground">↓ Hidden layer</div>
              <div>z₁ = [{z1_0.toFixed(3)}, {z1_1.toFixed(3)}]</div>
              <div>a₁ = σ(z₁) = [{a1_0.toFixed(3)}, {a1_1.toFixed(3)}]</div>
              <div className="text-muted-foreground">↓ Output layer</div>
              <div>z₂ = {z2.toFixed(3)}</div>
              <div className="text-lg font-bold">ŷ = σ(z₂) = {output.toFixed(3)}</div>
              <div className="border-t pt-1 mt-1">
                <div>Target: y = {target}</div>
                <div className="font-bold text-red-600">Loss = {loss.toFixed(4)}</div>
              </div>
            </div>

            <h4 className="font-semibold mt-4">Backward Pass (Compute Gradients)</h4>
            <div className="bg-red-50 border border-red-200 rounded p-3 font-mono text-xs space-y-1">
              <div className="font-bold">∂Loss/∂ŷ = {dL_dOutput.toFixed(4)}</div>
              <div className="text-muted-foreground">↓ Chain rule</div>
              <div>∂Loss/∂z₂ = {dL_dZ2.toFixed(4)}</div>
              <div className="text-muted-foreground">↓ Compute gradients for each weight</div>
              <div className="text-green-600 font-bold">∂Loss/∂w₂[0][0] = {dL_dW2_0.toFixed(4)}</div>
              <div className="text-green-600 font-bold">∂Loss/∂w₂[1][0] = {dL_dW2_1.toFixed(4)}</div>
              <div className="text-green-600 font-bold">∂Loss/∂b₂ = {dL_dB2.toFixed(4)}</div>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="font-semibold">Gradient Descent Update (Same as Level 1!)</h4>
            <div className="bg-green-50 border-2 border-green-400 rounded-lg p-4 space-y-3">
              <div className="text-sm font-mono space-y-2">
                <div className="font-bold text-lg mb-2">For EACH weight:</div>
                <div className="bg-white p-2 rounded">
                  w_new = w_old - α × gradient
                </div>
              </div>

              <div className="space-y-2 text-xs font-mono bg-white p-3 rounded">
                <div className="font-semibold mb-1">Current values:</div>
                <div>w₂[0][0] = {w2[0][0].toFixed(4)}</div>
                <div>w₂[1][0] = {w2[1][0].toFixed(4)}</div>
                <div>b₂ = {b2[0].toFixed(4)}</div>
              </div>

              <div className="space-y-2 text-xs font-mono bg-green-100 p-3 rounded border border-green-300">
                <div className="font-semibold mb-1">After update (α=0.5):</div>
                <div>w₂[0][0] = {w2[0][0].toFixed(4)} - 0.5×{dL_dW2_0.toFixed(3)} = <strong>{(w2[0][0] - learningRate * dL_dW2_0).toFixed(4)}</strong></div>
                <div>w₂[1][0] = {w2[1][0].toFixed(4)} - 0.5×{dL_dW2_1.toFixed(3)} = <strong>{(w2[1][0] - learningRate * dL_dW2_1).toFixed(4)}</strong></div>
                <div>b₂ = {b2[0].toFixed(4)} - 0.5×{dL_dB2.toFixed(3)} = <strong>{(b2[0] - learningRate * dL_dB2).toFixed(4)}</strong></div>
              </div>

              <Button onClick={trainStep} className="w-full mt-2">
                <StepForward className="w-4 h-4 mr-2" />
                Apply Gradient Descent Step {step > 0 && `(${step})`}
              </Button>
            </div>

            <div className="text-xs text-muted-foreground bg-slate-50 p-3 rounded">
              <strong>Important:</strong> In a full training step, we update ALL weights (including hidden layer)
              using their respective gradients. Here we're just showing the output layer for clarity.
            </div>
          </div>
        </div>

        <div className="text-sm space-y-2 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <h4 className="font-semibold text-yellow-900">The Unified Picture:</h4>
          <ul className="list-disc pl-5 space-y-1 text-yellow-800">
            <li><strong>Single weight:</strong> Compute gradient, update with w = w - α×gradient</li>
            <li><strong>Neural network:</strong> Compute gradients for ALL weights (backprop), then update EACH weight with the same rule!</li>
            <li><strong>Backpropagation</strong> is just an efficient algorithm to compute all gradients at once using the chain rule</li>
            <li><strong>Gradient descent</strong> is the update rule that moves parameters toward lower loss</li>
            <li>They work together: <strong>Backprop finds the gradients, Gradient Descent uses them to update</strong></li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Level 3: Visual Journey Guide
// ==========================================
function LearningGuide() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="w-5 h-5" />
          Learning Journey Guide
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Follow this progressive path to master neural network learning from the ground up.
        </p>

        <div className="space-y-4">
          {/* Level 1 */}
          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
            <h3 className="font-bold text-blue-900 mb-2 flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">1</div>
              Start: Single Weight
            </h3>
            <p className="text-sm text-blue-800 mb-2">
              Understand gradient descent with the simplest possible case: one weight, one input, one output.
            </p>
            <ul className="list-disc pl-5 text-xs text-blue-700 space-y-1">
              <li>See how the gradient points "downhill" on the loss curve</li>
              <li>Watch the weight converge to the optimal value</li>
              <li>Understand the update rule: w = w - learning_rate × gradient</li>
            </ul>
          </div>

          {/* Level 2 */}
          <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
            <h3 className="font-bold text-purple-900 mb-2 flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center font-bold">2</div>
              Bridge: Backprop = Gradient Descent
            </h3>
            <p className="text-sm text-purple-800 mb-2">
              Make the crucial connection: backpropagation in neural networks IS gradient descent!
            </p>
            <ul className="list-disc pl-5 text-xs text-purple-700 space-y-1">
              <li>See the SAME update rule applied to neural network weights</li>
              <li>Understand that backprop computes gradients for ALL weights</li>
              <li>Realize that the math is identical—just scaled up</li>
            </ul>
          </div>

          {/* Level 3 */}
          <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
            <h3 className="font-bold text-green-900 mb-2 flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center font-bold">3</div>
              Advanced: Full MLP Training
            </h3>
            <p className="text-sm text-green-800 mb-2">
              Ready to explore the complete MLP with architecture, XOR problem, and backpropagation details.
            </p>
            <ul className="list-disc pl-5 text-xs text-green-700 space-y-1">
              <li>Build intuition for network architecture</li>
              <li>See forward and backward propagation in action</li>
              <li>Solve the XOR problem—impossible for single neurons!</li>
            </ul>
          </div>

          {/* Level 4 */}
          <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded">
            <h3 className="font-bold text-amber-900 mb-2 flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-amber-500 text-white flex items-center justify-center font-bold">4</div>
              Deep Dive: Gradient Descent Variants
            </h3>
            <p className="text-sm text-amber-800 mb-2">
              Explore activation functions, universal approximation, GD variants, and regularization.
            </p>
            <ul className="list-disc pl-5 text-xs text-amber-700 space-y-1">
              <li>Compare Batch, Stochastic, and Mini-batch gradient descent</li>
              <li>Understand overfitting and dropout regularization</li>
              <li>See MLPs as universal function approximators</li>
            </ul>
          </div>

          {/* Level 5 */}
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
            <h3 className="font-bold text-red-900 mb-2 flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-red-500 text-white flex items-center justify-center font-bold">5</div>
              Mastery: Visual Gradient Descent
            </h3>
            <p className="text-sm text-red-800 mb-2">
              Visualize gradient descent on real problems with dual-panel cost landscapes.
            </p>
            <ul className="list-disc pl-5 text-xs text-red-700 space-y-1">
              <li>See the cost landscape for linear regression</li>
              <li>Watch the "ball" roll downhill to the minimum</li>
              <li>Understand high-dimensional gradient vectors</li>
            </ul>
          </div>
        </div>

        <div className="bg-slate-100 border border-slate-300 rounded-lg p-4 mt-6">
          <h4 className="font-semibold mb-2 flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Pedagogical Approach
          </h4>
          <p className="text-xs text-muted-foreground">
            This learning path follows the <strong>concrete → abstract</strong> principle. We start with the simplest
            case (one weight) where everything is visible and intuitive, then gradually build up complexity. Each level
            reinforces that the SAME fundamental idea (gradient descent) applies at every scale—from one weight to
            millions of parameters.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// ==========================================
// Main Component
// ==========================================
export default function Topic4UnifiedLearning() {
  return (
    <div className="w-full bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-2"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-slate-900 flex items-center justify-center gap-3">
            <Brain className="w-8 h-8 text-blue-600" />
            Topic 4: From MLP to Gradient Descent - A Unified Journey
          </h1>
          <p className="text-slate-600 max-w-3xl mx-auto">
            Build intuition step-by-step: Start with one weight, discover the connection to backpropagation,
            then master neural network learning. Everything is the same gradient descent algorithm!
          </p>
        </motion.div>

        <Tabs defaultValue="guide" className="w-full">
          <TabsList className="grid w-full grid-cols-2 md:grid-cols-3 gap-2">
            <TabsTrigger value="guide">📚 Learning Guide</TabsTrigger>
            <TabsTrigger value="level1">1️⃣ Single Weight</TabsTrigger>
            <TabsTrigger value="level2">🔗 The Bridge</TabsTrigger>
          </TabsList>

          <TabsContent value="guide" className="mt-6">
            <LearningGuide />
          </TabsContent>

          <TabsContent value="level1" className="mt-6">
            <Level1SingleWeight />
          </TabsContent>

          <TabsContent value="level2" className="mt-6">
            <Level2Bridge />
          </TabsContent>
        </Tabs>

        <footer className="text-center text-sm text-slate-500 pt-8 border-t">
          <p>
            COE 292 — Introduction to Artificial Intelligence •
            Progressive Learning: Concrete → Abstract
          </p>
        </footer>
      </div>
    </div>
  );
}
