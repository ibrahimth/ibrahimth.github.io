import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Pause, RotateCcw, SkipForward, SkipBack } from "lucide-react";

// The actual data from the table
const iterations = [
  { x1: 1.00, x2: 1.00, x3: 1.00, F: 3.00, grad1: 2.00, grad2: 0.00, grad3: 2.00 },
  { x1: 0.40, x2: 1.00, x3: 0.40, F: 1.32, grad1: 0.80, grad2: 1.20, grad3: 0.80 },
  { x1: 0.16, x2: 0.64, x3: 0.16, F: 0.58, grad1: 0.68, grad2: 0.96, grad3: 0.68 },
  { x1: -0.04, x2: 0.35, x3: -0.04, F: 0.07, grad1: 0.56, grad2: 0.79, grad3: 0.56 },
  { x1: -0.21, x2: 0.11, x3: -0.21, F: -0.27, grad1: 0.46, grad2: 0.65, grad3: 0.46 },
  { x1: -0.35, x2: -0.08, x3: -0.35, F: -0.51, grad1: 0.38, grad2: 0.54, grad3: 0.38 },
  { x1: -0.46, x2: -0.24, x3: -0.46, F: -0.66, grad1: 0.31, grad2: 0.44, grad3: 0.31 },
  { x1: -0.56, x2: -0.38, x3: -0.56, F: -0.77, grad1: 0.26, grad2: 0.37, grad3: 0.26 },
  { x1: -0.64, x2: -0.49, x3: -0.64, F: -0.84, grad1: 0.21, grad2: 0.30, grad3: 0.21 },
  { x1: -0.70, x2: -0.58, x3: -0.70, F: -0.89, grad1: 0.18, grad2: 0.25, grad3: 0.18 },
  { x1: -0.75, x2: -0.65, x3: -0.75, F: -0.93, grad1: 0.14, grad2: 0.20, grad3: 0.14 },
  { x1: -0.80, x2: -0.71, x3: -0.80, F: -0.95, grad1: 0.12, grad2: 0.17, grad3: 0.12 },
  { x1: -0.83, x2: -0.76, x3: -0.83, F: -0.97, grad1: 0.10, grad2: 0.14, grad3: 0.10 },
  { x1: -0.86, x2: -0.80, x3: -0.86, F: -0.98, grad1: 0.08, grad2: 0.11, grad3: 0.08 },
  { x1: -0.89, x2: -0.84, x3: -0.89, F: -0.98, grad1: 0.07, grad2: 0.09, grad3: 0.07 },
  { x1: -0.91, x2: -0.87, x3: -0.91, F: -0.99, grad1: 0.06, grad2: 0.08, grad3: 0.06 },
  { x1: -0.92, x2: -0.89, x3: -0.92, F: -0.99, grad1: 0.05, grad2: 0.06, grad3: 0.05 },
  { x1: -1.00, x2: -1.00, x3: -1.00, F: -1.00, grad1: 0.00, grad2: 0.00, grad3: 0.00 }
];

export default function GradientDescentComplete() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const current = iterations[currentStep];
  const learningRate = 0.3;

  // Animation control
  useEffect(() => {
    if (!isPlaying) return;

    const timer = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= iterations.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isPlaying]);

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-4">
      <div className="max-w-6xl mx-auto space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl">
              Complete Gradient Descent Visualization
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-2">
              f(x₁, x₂, x₃) = (x₁)² + x₁(1-x₂) + (x₂)² - x₂x₃ + (x₃)² + x₃
            </p>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Iteration Info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className={`${currentStep === iterations.length - 1 ? 'bg-gradient-to-br from-green-50 to-green-100 border-2 border-green-300' : 'bg-gradient-to-br from-blue-50 to-blue-100 border-2 border-blue-200'} p-4 rounded-lg`}>
                <div className={`text-sm font-semibold ${currentStep === iterations.length - 1 ? 'text-green-700' : 'text-blue-700'}`}>
                  {currentStep === iterations.length - 1 ? '✓ Converged' : '⚡ Training'}
                </div>
                <div className={`text-3xl font-bold ${currentStep === iterations.length - 1 ? 'text-green-900' : 'text-blue-900'}`}>
                  Step {currentStep}
                </div>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border-2 border-green-200">
                <div className="text-sm text-green-700 font-semibold">Function Value (F)</div>
                <div className="text-3xl font-bold text-green-900">{current.F.toFixed(3)}</div>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border-2 border-purple-200">
                <div className="text-sm text-purple-700 font-semibold">Learning Rate (η)</div>
                <div className="text-3xl font-bold text-purple-900">{learningRate}</div>
              </div>
              <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg border-2 border-orange-200">
                <div className="text-sm text-orange-700 font-semibold">Gradient Norm</div>
                <div className="text-2xl font-bold text-orange-900">
                  {Math.sqrt(current.grad1**2 + current.grad2**2 + current.grad3**2).toFixed(3)}
                </div>
              </div>
            </div>

            {/* Partial Derivatives Formulas */}
            <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3 text-blue-900">Partial Derivatives</h3>
              <div className="space-y-2 font-mono text-sm">
                <div className="flex items-center justify-between bg-white p-3 rounded border border-blue-200">
                  <span>∂f/∂x₁ = 2x₁ + 1 - x₂</span>
                  <span className="text-red-600 font-bold">= {current.grad1.toFixed(3)}</span>
                </div>
                <div className="flex items-center justify-between bg-white p-3 rounded border border-blue-200">
                  <span>∂f/∂x₂ = -x₁ + 2x₂ - x₃</span>
                  <span className="text-red-600 font-bold">= {current.grad2.toFixed(3)}</span>
                </div>
                <div className="flex items-center justify-between bg-white p-3 rounded border border-blue-200">
                  <span>∂f/∂x₃ = -x₂ + 2x₃ + 1</span>
                  <span className="text-red-600 font-bold">= {current.grad3.toFixed(3)}</span>
                </div>
              </div>
            </div>

            {/* Current Values */}
            <div className="bg-slate-50 border-2 border-slate-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3">Current Values</h3>
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-white p-3 rounded border-2 border-blue-300">
                  <div className="text-sm text-muted-foreground">x₁</div>
                  <div className="text-2xl font-bold text-blue-600">{current.x1.toFixed(3)}</div>
                </div>
                <div className="bg-white p-3 rounded border-2 border-blue-300">
                  <div className="text-sm text-muted-foreground">x₂</div>
                  <div className="text-2xl font-bold text-blue-600">{current.x2.toFixed(3)}</div>
                </div>
                <div className="bg-white p-3 rounded border-2 border-blue-300">
                  <div className="text-sm text-muted-foreground">x₃</div>
                  <div className="text-2xl font-bold text-blue-600">{current.x3.toFixed(3)}</div>
                </div>
              </div>
            </div>

            {/* Update Rule */}
            {currentStep < iterations.length - 1 && (
              <div className="bg-green-50 border-2 border-green-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-green-900">Next Step Update Rule</h3>
                <div className="space-y-2 font-mono text-sm">
                  <div className="bg-white p-3 rounded border border-green-200">
                    x₁ₙₑw = {current.x1.toFixed(2)} - {learningRate} × {current.grad1.toFixed(2)} = {(current.x1 - learningRate * current.grad1).toFixed(3)}
                  </div>
                  <div className="bg-white p-3 rounded border border-green-200">
                    x₂ₙₑw = {current.x2.toFixed(2)} - {learningRate} × {current.grad2.toFixed(2)} = {(current.x2 - learningRate * current.grad2).toFixed(3)}
                  </div>
                  <div className="bg-white p-3 rounded border border-green-200">
                    x₃ₙₑw = {current.x3.toFixed(2)} - {learningRate} × {current.grad3.toFixed(2)} = {(current.x3 - learningRate * current.grad3).toFixed(3)}
                  </div>
                </div>
              </div>
            )}

            {/* Controls */}
            <div className="flex items-center justify-center gap-3">
              <Button
                onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                disabled={currentStep === 0}
                variant="outline"
              >
                <SkipBack className="w-4 h-4" />
              </Button>
              <Button
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={currentStep === iterations.length - 1}
                size="lg"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4 mr-2" /> Pause
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" /> Play
                  </>
                )}
              </Button>
              <Button
                onClick={() => setCurrentStep(Math.min(iterations.length - 1, currentStep + 1))}
                disabled={currentStep === iterations.length - 1}
                variant="outline"
              >
                <SkipForward className="w-4 h-4" />
              </Button>
              <Button
                onClick={() => {
                  setCurrentStep(0);
                  setIsPlaying(false);
                }}
                variant="outline"
              >
                <RotateCcw className="w-4 h-4 mr-2" /> Reset
              </Button>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Progress</span>
                <span>{currentStep} / {iterations.length - 1}</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-blue-500 to-green-500 h-full transition-all duration-300"
                  style={{ width: `${(currentStep / (iterations.length - 1)) * 100}%` }}
                />
              </div>
            </div>

            {/* Convergence Status */}
            {currentStep === iterations.length - 1 && (
              <div className="bg-green-100 border-2 border-green-300 rounded-lg p-4 text-center">
                <h3 className="text-lg font-bold text-green-800 mb-2">🎯 Convergence Achieved!</h3>
                <p className="text-green-700">
                  The gradient descent has converged to the minimum at approximately (-1, -1, -1) with F ≈ -1
                </p>
                <p className="text-sm text-green-600 mt-2">
                  All gradients are now ≈ 0, indicating we've reached the optimal solution.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Neural Network Explanation */}
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">🧠 How the Equations Come from Neural Networks</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Computational Graph */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Step 1: Computational Graph (Forward Pass)</h3>
              <div className="bg-slate-50 border-2 border-slate-200 rounded-lg p-6">
                <div className="space-y-4 text-sm">
                  <p className="font-semibold text-lg mb-3">Breaking down f(x₁, x₂, x₃) = (x₁)² + x₁(1-x₂) + (x₂)² - x₂x₃ + (x₃)² + x₃</p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-white p-4 rounded border-2 border-blue-200">
                      <h4 className="font-bold text-blue-900 mb-2">Layer 1: Basic Operations</h4>
                      <div className="space-y-1 font-mono text-xs">
                        <div>t₁ = x₁²</div>
                        <div>t₂ = 1 - x₂</div>
                        <div>t₃ = x₁ × t₂</div>
                        <div>t₄ = x₂²</div>
                        <div>t₅ = x₂ × x₃</div>
                        <div>t₆ = x₃²</div>
                      </div>
                    </div>

                    <div className="bg-white p-4 rounded border-2 border-green-200">
                      <h4 className="font-bold text-green-900 mb-2">Layer 2: Aggregation</h4>
                      <div className="space-y-1 font-mono text-xs">
                        <div>f = t₁ + t₃ + t₄ - t₅ + t₆ + x₃</div>
                        <div className="text-muted-foreground mt-2">
                          Each term contributes to the final output
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg mt-4">
                    <p className="text-sm">
                      <strong>🔵 Forward Pass:</strong> Data flows from inputs (x₁, x₂, x₃) → intermediate computations (t₁...t₆) → final output (f)
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Backpropagation */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Step 2: Backpropagation (Backward Pass)</h3>
              <div className="bg-slate-50 border-2 border-slate-200 rounded-lg p-6">
                <p className="text-sm mb-4">
                  <strong>The Chain Rule:</strong> Gradients flow backward from output to inputs. Each node passes gradients to its parents.
                </p>

                <div className="space-y-4">
                  {/* For x1 */}
                  <div className="bg-white p-4 rounded border-l-4 border-red-500">
                    <h4 className="font-bold text-red-900 mb-2">∂f/∂x₁ Derivation</h4>
                    <div className="space-y-2 text-sm">
                      <p className="font-mono">x₁ affects f through two paths:</p>
                      <ol className="list-decimal pl-6 space-y-1">
                        <li><strong>Path 1 (through t₁):</strong> x₁ → t₁ = x₁² → f
                          <div className="ml-4 text-blue-600">∂f/∂t₁ × ∂t₁/∂x₁ = 1 × 2x₁ = 2x₁</div>
                        </li>
                        <li><strong>Path 2 (through t₃):</strong> x₁ → t₃ = x₁(1-x₂) → f
                          <div className="ml-4 text-blue-600">∂f/∂t₃ × ∂t₃/∂x₁ = 1 × (1-x₂) = 1-x₂</div>
                        </li>
                      </ol>
                      <div className="bg-red-50 p-3 rounded mt-3 font-bold">
                        ∂f/∂x₁ = 2x₁ + (1-x₂) = 2x₁ + 1 - x₂ ✓
                      </div>
                    </div>
                  </div>

                  {/* For x2 */}
                  <div className="bg-white p-4 rounded border-l-4 border-green-500">
                    <h4 className="font-bold text-green-900 mb-2">∂f/∂x₂ Derivation</h4>
                    <div className="space-y-2 text-sm">
                      <p className="font-mono">x₂ affects f through three paths:</p>
                      <ol className="list-decimal pl-6 space-y-1">
                        <li><strong>Path 1 (through t₂→t₃):</strong> x₂ → t₂ = (1-x₂) → t₃ = x₁t₂ → f
                          <div className="ml-4 text-blue-600">∂f/∂t₃ × ∂t₃/∂t₂ × ∂t₂/∂x₂ = 1 × x₁ × (-1) = -x₁</div>
                        </li>
                        <li><strong>Path 2 (through t₄):</strong> x₂ → t₄ = x₂² → f
                          <div className="ml-4 text-blue-600">∂f/∂t₄ × ∂t₄/∂x₂ = 1 × 2x₂ = 2x₂</div>
                        </li>
                        <li><strong>Path 3 (through t₅):</strong> x₂ → t₅ = x₂x₃ → f (note: f has -t₅)
                          <div className="ml-4 text-blue-600">∂f/∂t₅ × ∂t₅/∂x₂ = (-1) × x₃ = -x₃</div>
                        </li>
                      </ol>
                      <div className="bg-green-50 p-3 rounded mt-3 font-bold">
                        ∂f/∂x₂ = -x₁ + 2x₂ + (-x₃) = -x₁ + 2x₂ - x₃ ✓
                      </div>
                    </div>
                  </div>

                  {/* For x3 */}
                  <div className="bg-white p-4 rounded border-l-4 border-purple-500">
                    <h4 className="font-bold text-purple-900 mb-2">∂f/∂x₃ Derivation</h4>
                    <div className="space-y-2 text-sm">
                      <p className="font-mono">x₃ affects f through three paths:</p>
                      <ol className="list-decimal pl-6 space-y-1">
                        <li><strong>Path 1 (through t₅):</strong> x₃ → t₅ = x₂x₃ → f (note: f has -t₅)
                          <div className="ml-4 text-blue-600">∂f/∂t₅ × ∂t₅/∂x₃ = (-1) × x₂ = -x₂</div>
                        </li>
                        <li><strong>Path 2 (through t₆):</strong> x₃ → t₆ = x₃² → f
                          <div className="ml-4 text-blue-600">∂f/∂t₆ × ∂t₆/∂x₃ = 1 × 2x₃ = 2x₃</div>
                        </li>
                        <li><strong>Path 3 (direct):</strong> x₃ → f (linear term)
                          <div className="ml-4 text-blue-600">∂f/∂x₃ = 1</div>
                        </li>
                      </ol>
                      <div className="bg-purple-50 p-3 rounded mt-3 font-bold">
                        ∂f/∂x₃ = -x₂ + 2x₃ + 1 ✓
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Key Insights */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-300 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-3">🎯 Key Neural Network Insights</h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">1️⃣</span>
                  <div>
                    <strong>Computational Graph:</strong> Every mathematical function can be broken into a network of simple operations (additions, multiplications, squares, etc.)
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">2️⃣</span>
                  <div>
                    <strong>Chain Rule = Message Passing:</strong> Gradients flow backward through the graph. Each node receives gradient from children and sends gradient to parents.
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">3️⃣</span>
                  <div>
                    <strong>Multiple Paths:</strong> If a variable affects the output through multiple paths, we sum all the gradients (like x₂ having 3 paths).
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">4️⃣</span>
                  <div>
                    <strong>Automatic Differentiation:</strong> Neural network frameworks (PyTorch, TensorFlow) automatically build this computational graph and compute gradients using backpropagation—you never calculate derivatives by hand!
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="text-2xl">5️⃣</span>
                  <div>
                    <strong>Efficiency:</strong> Backpropagation computes all partial derivatives in one backward pass through the network—much more efficient than computing each derivative separately.
                  </div>
                </div>
              </div>
            </div>

            {/* Visual Summary */}
            <div className="bg-slate-900 text-white rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-3">📊 The Big Picture</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="bg-slate-800 p-4 rounded">
                  <div className="text-blue-400 font-bold mb-2">Forward Pass →</div>
                  <div className="text-slate-300">Inputs flow through network operations to produce output f(x₁, x₂, x₃)</div>
                </div>
                <div className="bg-slate-800 p-4 rounded">
                  <div className="text-red-400 font-bold mb-2">← Backward Pass</div>
                  <div className="text-slate-300">Gradients flow backward using chain rule to compute ∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃</div>
                </div>
                <div className="bg-slate-800 p-4 rounded">
                  <div className="text-green-400 font-bold mb-2">Update Step</div>
                  <div className="text-slate-300">Use gradients to update parameters: x_new = x_old - η × ∇f</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}