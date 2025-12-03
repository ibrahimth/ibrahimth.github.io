import React, { useState, useEffect, useRef } from 'react';
import Topic4UnifiedLearning from './components/Topic4UnifiedLearning';
import GradientDescentComplete from './components/Topic4MultivariateGD';

/**
 * Topic 4: Neural Networks & Gradient Descent (Complete)
 * Based on CMU 11-785, 3Blue1Brown, and StatQuest
 *
 * Sections:
 * 1. Activation Functions
 * 2. Single Weight Gradient Descent
 * 3. Linear Regression with Dual-Panel Visualization
 * 4. MLP Architecture
 * 5. Unified Learning Journey
 * 6. Multivariate Gradient Descent
 */

type ActivationFunction = 'sigmoid' | 'tanh' | 'relu' | 'leakyRelu';
type Section = 'activations' | 'singleWeight' | 'linearRegression' | 'mlp' | 'journey' | 'multivariate';

// Activation functions
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const sigmoidDerivative = (x: number) => {
  const s = sigmoid(x);
  return s * (1 - s);
};

const tanh = (x: number) => Math.tanh(x);
const tanhDerivative = (x: number) => 1 - Math.pow(Math.tanh(x), 2);

const relu = (x: number) => Math.max(0, x);
const reluDerivative = (x: number) => x > 0 ? 1 : 0;

const leakyRelu = (x: number) => Math.max(0.01 * x, x);
const leakyReluDerivative = (x: number) => x > 0 ? 1 : 0.01;

const activationFunctions = {
  sigmoid: { fn: sigmoid, derivative: sigmoidDerivative, name: 'Sigmoid', formula: 'σ(x) = 1/(1+e⁻ˣ)' },
  tanh: { fn: tanh, derivative: tanhDerivative, name: 'Tanh', formula: 'σ(x) = tanh(x)' },
  relu: { fn: relu, derivative: reluDerivative, name: 'ReLU', formula: 'σ(x) = max(0,x)' },
  leakyRelu: { fn: leakyRelu, derivative: leakyReluDerivative, name: 'Leaky ReLU', formula: 'σ(x) = max(0.01x,x)' },
};

export default function NeuralNetworksComplete() {
  const [activeSection, setActiveSection] = useState<Section>('activations');

  // Activation function state
  const [selectedActivation, setSelectedActivation] = useState<ActivationFunction>('sigmoid');

  // Single weight GD state
  const [weight, setWeight] = useState(0.5);
  const [learningRate, setLearningRate] = useState(0.1);
  const [steps, setSteps] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Linear regression state
  const [intercept, setIntercept] = useState(-2);
  const [slope, setSlope] = useState(-1);
  const [lrLearningRate, setLrLearningRate] = useState(0.01);
  const [lrSteps, setLrSteps] = useState(0);
  const [lrIsPlaying, setLrIsPlaying] = useState(false);

  // Generate sample data for linear regression
  const dataPoints = [
    { x: 1, y: 3 }, { x: 2, y: 5 }, { x: 3, y: 7 },
    { x: 4, y: 9 }, { x: 5, y: 11 }, { x: 6, y: 13 },
    { x: 7, y: 15 }, { x: 8, y: 17 }, { x: 9, y: 19 }
  ];

  // MLP state
  const [mlpLayers, setMlpLayers] = useState([2, 3, 1]);
  const [mlpActivation, setMlpActivation] = useState<ActivationFunction>('sigmoid');

  // Single weight gradient descent calculations
  const input = 2;
  const target = 6;
  const prediction = weight * input;
  const error = prediction - target;
  const cost = error * error;
  const gradient = 2 * error * input; // ∂C/∂w = 2(ŷ-y)x

  const stepSingleWeight = () => {
    const newWeight = weight - learningRate * gradient;
    setWeight(newWeight);
    setSteps(s => s + 1);
  };

  // Linear regression calculations
  const calculateSSE = () => {
    return dataPoints.reduce((sum, point) => {
      const predicted = intercept + slope * point.x;
      const residual = predicted - point.y;
      return sum + residual * residual;
    }, 0);
  };

  const calculateGradients = () => {
    let dIntercept = 0;
    let dSlope = 0;

    dataPoints.forEach(point => {
      const predicted = intercept + slope * point.x;
      const residual = predicted - point.y;
      dIntercept += 2 * residual;
      dSlope += 2 * residual * point.x;
    });

    return {
      dIntercept: dIntercept / dataPoints.length,
      dSlope: dSlope / dataPoints.length
    };
  };

  const sse = calculateSSE();
  const { dIntercept, dSlope } = calculateGradients();

  const stepLinearRegression = () => {
    const grads = calculateGradients();
    setIntercept(intercept - lrLearningRate * grads.dIntercept);
    setSlope(slope - lrLearningRate * grads.dSlope);
    setLrSteps(s => s + 1);
  };

  // Auto-play effect for single weight
  useEffect(() => {
    if (isPlaying && Math.abs(gradient) > 0.01) {
      const timer = setTimeout(stepSingleWeight, 200);
      return () => clearTimeout(timer);
    } else if (isPlaying && Math.abs(gradient) <= 0.01) {
      setIsPlaying(false);
    }
  }, [isPlaying, weight]);

  // Auto-play effect for linear regression
  useEffect(() => {
    if (lrIsPlaying && (Math.abs(dIntercept) > 0.1 || Math.abs(dSlope) > 0.1)) {
      const timer = setTimeout(stepLinearRegression, 200);
      return () => clearTimeout(timer);
    } else if (lrIsPlaying && Math.abs(dIntercept) <= 0.1 && Math.abs(dSlope) <= 0.1) {
      setLrIsPlaying(false);
    }
  }, [lrIsPlaying, intercept, slope]);

  // Render activation function plot
  const renderActivationPlot = () => {
    const width = 400;
    const height = 300;
    const xMin = -5;
    const xMax = 5;
    const xScale = width / (xMax - xMin);
    const yScale = height / 6; // -3 to 3 y-axis
    const toX = (x: number) => (x - xMin) * xScale;
    const toY = (y: number) => height / 2 - y * yScale;

    const { fn, derivative } = activationFunctions[selectedActivation];

    const points: string[] = [];
    const derivPoints: string[] = [];

    for (let x = xMin; x <= xMax; x += 0.1) {
      const y = fn(x);
      const dy = derivative(x);
      points.push(`${toX(x)},${toY(y)}`);
      derivPoints.push(`${toX(x)},${toY(dy)}`);
    }

    return (
      <svg width={width} height={height} style={{ border: '1px solid #d1d5db', borderRadius: '8px', background: 'white' }}>
        {/* Grid */}
        <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke="#e5e7eb" strokeWidth={1} />
        <line x1={toX(0)} y1={0} x2={toX(0)} y2={height} stroke="#e5e7eb" strokeWidth={1} />

        {/* Derivative (dashed) */}
        <polyline
          points={derivPoints.join(' ')}
          fill="none"
          stroke="#9ca3af"
          strokeWidth={2}
          strokeDasharray="5,5"
        />

        {/* Activation function (solid) */}
        <polyline
          points={points.join(' ')}
          fill="none"
          stroke="#3b82f6"
          strokeWidth={3}
        />

        {/* Labels */}
        <text x={width - 60} y={20} fill="#3b82f6" fontSize="14" fontWeight="600">σ(x)</text>
        <text x={width - 60} y={40} fill="#9ca3af" fontSize="14" fontWeight="600">σ'(x)</text>
      </svg>
    );
  };

  // Render single weight loss landscape
  const renderSingleWeightLoss = () => {
    const width = 400;
    const height = 200;
    const wMin = -2;
    const wMax = 8;
    const wScale = width / (wMax - wMin);
    const toX = (w: number) => (w - wMin) * wScale;

    // Calculate parabola (cost for different weights)
    const costs: { w: number; c: number }[] = [];
    for (let w = wMin; w <= wMax; w += 0.2) {
      const pred = w * input;
      const err = pred - target;
      const c = err * err;
      costs.push({ w, c });
    }

    const maxCost = Math.max(...costs.map(p => p.c));
    const toY = (c: number) => height - (c / maxCost) * (height - 20);

    const points = costs.map(p => `${toX(p.w)},${toY(p.c)}`).join(' ');

    return (
      <svg width={width} height={height} style={{ border: '1px solid #d1d5db', borderRadius: '8px', background: 'white' }}>
        {/* Parabola */}
        <polyline points={points} fill="none" stroke="#6366f1" strokeWidth={2} />

        {/* Current weight */}
        <circle cx={toX(weight)} cy={toY(cost)} r={6} fill="#ef4444" />

        {/* Gradient arrow */}
        {Math.abs(gradient) > 0.1 && (
          <>
            <line
              x1={toX(weight)}
              y1={toY(cost)}
              x2={toX(weight - gradient * 0.05)}
              y2={toY(cost)}
              stroke="#16a34a"
              strokeWidth={2}
              markerEnd="url(#arrowhead)"
            />
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#16a34a" />
              </marker>
            </defs>
          </>
        )}

        {/* Optimal weight line */}
        <line x1={toX(3)} y1={0} x2={toX(3)} y2={height} stroke="#22c55e" strokeWidth={1} strokeDasharray="3,3" />
        <text x={toX(3) + 5} y={15} fill="#22c55e" fontSize="12" fontWeight="600">optimal</text>
      </svg>
    );
  };

  // Render linear regression dual panel
  const renderLinearRegressionPanel = () => {
    const width = 350;
    const height = 300;

    // Problem space (left panel)
    const xMin = 0;
    const xMax = 10;
    const yMin = 0;
    const yMax = 20;
    const xScale = width / (xMax - xMin);
    const yScale = height / (yMax - yMin);
    const toX = (x: number) => x * xScale;
    const toY = (y: number) => height - y * yScale;

    // Cost landscape (right panel)
    const renderCostLandscape = () => {
      const iMin = -5;
      const iMax = 5;
      const sMin = -3;
      const sMax = 5;

      return (
        <svg width={width} height={height} style={{ border: '1px solid #d1d5db', borderRadius: '8px', background: '#f9fafb' }}>
          {/* Simplified contour representation */}
          {[50, 100, 200, 500, 1000, 2000].map((level, idx) => {
            const radius = Math.sqrt(level) * 2;
            return (
              <circle
                key={idx}
                cx={width / 2}
                cy={height / 2}
                r={radius}
                fill="none"
                stroke="#cbd5e1"
                strokeWidth={1}
              />
            );
          })}

          {/* Current position */}
          <circle
            cx={width / 2 + (intercept - 1) * 20}
            cy={height / 2 - (slope - 2) * 20}
            r={8}
            fill="#ef4444"
          />

          {/* Gradient arrow */}
          {(Math.abs(dIntercept) > 0.5 || Math.abs(dSlope) > 0.5) && (
            <line
              x1={width / 2 + (intercept - 1) * 20}
              y1={height / 2 - (slope - 2) * 20}
              x2={width / 2 + (intercept - dIntercept * 5 - 1) * 20}
              y2={height / 2 - (slope - dSlope * 5 - 2) * 20}
              stroke="white"
              strokeWidth={3}
              markerEnd="url(#arrowhead2)"
            />
          )}

          <defs>
            <marker id="arrowhead2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="white" />
            </marker>
          </defs>

          {/* Label */}
          <text x={10} y={20} fill="#374151" fontSize="14" fontWeight="700">Cost Landscape (SSE)</text>
          <text x={10} y={40} fill="#6b7280" fontSize="12">Red ball = current params</text>
        </svg>
      );
    };

    return (
      <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
        {/* Problem Space */}
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.95rem', fontWeight: '700' }}>
            Panel 1: Problem Space
          </h4>
          <svg width={width} height={height} style={{ border: '1px solid #d1d5db', borderRadius: '8px', background: 'white' }}>
            {/* Data points */}
            {dataPoints.map((point, idx) => {
              const predicted = intercept + slope * point.x;
              return (
                <g key={idx}>
                  {/* Residual line */}
                  <line
                    x1={toX(point.x)}
                    y1={toY(point.y)}
                    x2={toX(point.x)}
                    y2={toY(predicted)}
                    stroke="#ef4444"
                    strokeWidth={1}
                    strokeDasharray="3,3"
                  />
                  {/* Data point */}
                  <circle cx={toX(point.x)} cy={toY(point.y)} r={5} fill="#3b82f6" />
                </g>
              );
            })}

            {/* Regression line */}
            <line
              x1={toX(0)}
              y1={toY(intercept)}
              x2={toX(10)}
              y2={toY(intercept + slope * 10)}
              stroke="#22c55e"
              strokeWidth={2}
            />

            {/* Axes */}
            <line x1={0} y1={height} x2={width} y2={height} stroke="#9ca3af" strokeWidth={1} />
            <line x1={0} y1={0} x2={0} y2={height} stroke="#9ca3af" strokeWidth={1} />
          </svg>
          <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.75rem', color: '#6b7280' }}>
            Blue dots = data, Green line = model, Red = residuals
          </p>
        </div>

        {/* Cost Landscape */}
        <div>
          <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.95rem', fontWeight: '700' }}>
            Panel 2: Cost Landscape
          </h4>
          {renderCostLandscape()}
          <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.75rem', color: '#6b7280' }}>
            Contours = cost levels, Arrow = -∇SSE
          </p>
        </div>
      </div>
    );
  };

  // Render MLP architecture
  const renderMLPArchitecture = () => {
    const width = 600;
    const height = 350;
    const layerSpacing = width / (mlpLayers.length + 1);

    return (
      <svg width={width} height={height} style={{ background: '#f9fafb', borderRadius: '8px' }}>
        {/* Draw connections */}
        {mlpLayers.map((layerSize, layerIdx) => {
          if (layerIdx === 0) return null;

          const prevLayerSize = mlpLayers[layerIdx - 1];
          const prevX = layerSpacing * layerIdx;
          const currX = layerSpacing * (layerIdx + 1);
          const prevSpacing = height / (prevLayerSize + 1);
          const currSpacing = height / (layerSize + 1);

          const connections = [];
          for (let i = 0; i < prevLayerSize; i++) {
            for (let j = 0; j < layerSize; j++) {
              const weight = Math.random() * 2 - 1;
              connections.push(
                <line
                  key={`${layerIdx}-${i}-${j}`}
                  x1={prevX}
                  y1={prevSpacing * (i + 1)}
                  x2={currX}
                  y2={currSpacing * (j + 1)}
                  stroke={weight > 0 ? '#22c55e' : '#ef4444'}
                  strokeWidth={Math.abs(weight) * 2}
                  opacity={0.3}
                />
              );
            }
          }

          return <g key={`layer-${layerIdx}`}>{connections}</g>;
        })}

        {/* Draw neurons */}
        {mlpLayers.map((layerSize, layerIdx) => {
          const x = layerSpacing * (layerIdx + 1);
          const spacing = height / (layerSize + 1);

          return (
            <g key={`neurons-${layerIdx}`}>
              {Array.from({ length: layerSize }).map((_, neuronIdx) => {
                const y = spacing * (neuronIdx + 1);
                const isInput = layerIdx === 0;
                const isOutput = layerIdx === mlpLayers.length - 1;

                return (
                  <circle
                    key={neuronIdx}
                    cx={x}
                    cy={y}
                    r={15}
                    fill={isInput ? '#60a5fa' : isOutput ? '#f59e0b' : '#8b5cf6'}
                    stroke="#fff"
                    strokeWidth={2}
                  />
                );
              })}

              {/* Layer label */}
              <text
                x={x}
                y={height - 10}
                textAnchor="middle"
                fontSize="12"
                fontWeight="600"
                fill="#374151"
              >
                {layerIdx === 0 ? 'Input' : layerIdx === mlpLayers.length - 1 ? 'Output' : `Hidden`}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // Common styles
  const containerStyle: React.CSSProperties = {
    padding: '1.5rem',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  };

  const cardStyle: React.CSSProperties = {
    background: 'white',
    borderRadius: '12px',
    padding: '1.5rem',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    marginBottom: '1rem',
  };

  const buttonStyle: React.CSSProperties = {
    padding: '0.5rem 1rem',
    borderRadius: '6px',
    border: '1px solid #d1d5db',
    background: 'white',
    cursor: 'pointer',
    fontSize: '0.875rem',
    fontWeight: '500',
  };

  const primaryButtonStyle: React.CSSProperties = {
    ...buttonStyle,
    background: '#3b82f6',
    color: 'white',
    border: 'none',
  };

  const activeTabStyle: React.CSSProperties = {
    ...buttonStyle,
    background: '#3b82f6',
    color: 'white',
    border: 'none',
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem', fontWeight: '700' }}>
          🎓 Topic 4: Neural Networks & Gradient Descent
        </h2>
        <p style={{ margin: '0 0 1rem 0', color: '#6b7280', fontSize: '0.875rem' }}>
          Based on CMU 11-785, 3Blue1Brown, and StatQuest — Explore activation functions, gradient descent, and neural network learning
        </p>

        {/* Section tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <button
            onClick={() => setActiveSection('activations')}
            style={activeSection === 'activations' ? activeTabStyle : buttonStyle}
          >
            Activation Functions
          </button>
          <button
            onClick={() => setActiveSection('singleWeight')}
            style={activeSection === 'singleWeight' ? activeTabStyle : buttonStyle}
          >
            Single Weight GD
          </button>
          <button
            onClick={() => setActiveSection('linearRegression')}
            style={activeSection === 'linearRegression' ? activeTabStyle : buttonStyle}
          >
            Linear Regression
          </button>
          <button
            onClick={() => setActiveSection('mlp')}
            style={activeSection === 'mlp' ? activeTabStyle : buttonStyle}
          >
            MLP Architecture
          </button>
          <button
            onClick={() => setActiveSection('journey')}
            style={activeSection === 'journey' ? activeTabStyle : buttonStyle}
          >
            Unified Learning Journey
          </button>
          <button
            onClick={() => setActiveSection('multivariate')}
            style={activeSection === 'multivariate' ? activeTabStyle : buttonStyle}
          >
            Multivariate GD
          </button>
        </div>

        {/* Activation Functions Section */}
        {activeSection === 'activations' && (
          <div>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.2rem', fontWeight: '700' }}>
              Activation Functions Visualizer
            </h3>
            <p style={{ margin: '0 0 1rem 0', color: '#6b7280', fontSize: '0.875rem' }}>
              Activation functions introduce non-linearity into the network. Select a function to see its shape and derivative.
            </p>

            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
              {Object.keys(activationFunctions).map((key) => (
                <button
                  key={key}
                  onClick={() => setSelectedActivation(key as ActivationFunction)}
                  style={selectedActivation === key ? activeTabStyle : buttonStyle}
                >
                  {activationFunctions[key as ActivationFunction].name}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
              {renderActivationPlot()}
              <div style={{ maxWidth: '300px' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', fontWeight: '700' }}>
                  {activationFunctions[selectedActivation].name}
                </h4>
                <code style={{ display: 'block', background: '#f3f4f6', padding: '0.5rem', borderRadius: '4px', marginBottom: '1rem' }}>
                  {activationFunctions[selectedActivation].formula}
                </code>
                <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>
                  <strong>Blue Line:</strong> The activation function output.<br />
                  <strong>Dashed Line:</strong> The derivative (gradient).<br /><br />
                  Notice how the derivative behaves. For Sigmoid/Tanh, it vanishes at extremes (vanishing gradient problem). ReLU has a constant gradient for positive inputs.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Single Weight GD Section */}
        {activeSection === 'singleWeight' && (
          <div>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.2rem', fontWeight: '700' }}>
              Single Weight Gradient Descent
            </h3>
            <p style={{ margin: '0 0 1rem 0', color: '#6b7280', fontSize: '0.875rem' }}>
              Minimize Cost = (prediction - target)² by adjusting a single weight w.
            </p>

            <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem' }}>
              <div style={{ background: '#f9fafb', padding: '1rem', borderRadius: '8px', width: '200px' }}>
                <div style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Input (x): <strong>{input}</strong></div>
                <div style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Target (y): <strong>{target}</strong></div>
                <div style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Weight (w): <strong>{weight.toFixed(3)}</strong></div>
                <div style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Prediction (ŷ): <strong>{prediction.toFixed(3)}</strong></div>
                <div style={{ fontSize: '0.875rem', marginBottom: '0.5rem', color: '#ef4444' }}>Cost (C): <strong>{cost.toFixed(4)}</strong></div>
                <div style={{ fontSize: '0.875rem', color: '#16a34a' }}>Gradient (∂C/∂w): <strong>{gradient.toFixed(3)}</strong></div>
              </div>

              <div>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ fontSize: '0.875rem', fontWeight: '600', marginRight: '0.5rem' }}>Learning Rate (η):</label>
                  <input
                    type="number"
                    step="0.01"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    style={{ padding: '0.25rem', borderRadius: '4px', border: '1px solid #d1d5db' }}
                  />
                </div>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button onClick={stepSingleWeight} style={primaryButtonStyle}>
                    Step
                  </button>
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    style={buttonStyle}
                  >
                    {isPlaying ? 'Pause' : 'Auto Train'}
                  </button>
                  <button
                    onClick={() => { setWeight(0.5); setSteps(0); setIsPlaying(false); }}
                    style={buttonStyle}
                  >
                    Reset
                  </button>
                </div>
                <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280' }}>Steps: {steps}</p>
              </div>
            </div>

            {renderSingleWeightLoss()}
          </div>
        )}

        {/* Linear Regression Section */}
        {activeSection === 'linearRegression' && (
          <div>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.2rem', fontWeight: '700' }}>
              Linear Regression Gradient Descent
            </h3>
            <p style={{ margin: '0 0 1rem 0', color: '#6b7280', fontSize: '0.875rem' }}>
              Fitting a line y = mx + b. Visualizing the problem space vs the cost landscape.
            </p>

            <div style={{ marginBottom: '1.5rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div style={{ fontSize: '0.875rem' }}>
                Intercept (b): <strong>{intercept.toFixed(2)}</strong>
              </div>
              <div style={{ fontSize: '0.875rem' }}>
                Slope (m): <strong>{slope.toFixed(2)}</strong>
              </div>
              <div style={{ fontSize: '0.875rem', color: '#ef4444' }}>
                SSE Cost: <strong>{sse.toFixed(2)}</strong>
              </div>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
                <label style={{ fontSize: '0.875rem' }}>LR:</label>
                <input
                  type="number"
                  step="0.001"
                  value={lrLearningRate}
                  onChange={(e) => setLrLearningRate(parseFloat(e.target.value))}
                  style={{ width: '60px', padding: '0.25rem', borderRadius: '4px', border: '1px solid #d1d5db' }}
                />
                <button onClick={stepLinearRegression} style={primaryButtonStyle}>Step</button>
                <button onClick={() => setLrIsPlaying(!lrIsPlaying)} style={buttonStyle}>
                  {lrIsPlaying ? 'Pause' : 'Play'}
                </button>
                <button
                  onClick={() => { setIntercept(-2); setSlope(-1); setLrSteps(0); setLrIsPlaying(false); }}
                  style={buttonStyle}
                >
                  Reset
                </button>
              </div>
            </div>

            {renderLinearRegressionPanel()}
          </div>
        )}

        {/* MLP Architecture Section */}
        {activeSection === 'mlp' && (
          <div>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.2rem', fontWeight: '700' }}>
              MLP Architecture Visualizer
            </h3>
            <p style={{ margin: '0 0 1rem 0', color: '#6b7280', fontSize: '0.875rem' }}>
              A Multilayer Perceptron consists of layers of neurons. Green = positive weights, Red = negative weights.
            </p>

            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', fontWeight: '600' }}>
                Architecture: {mlpLayers.join(' → ')}
              </label>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <div>
                  <label style={{ fontSize: '0.75rem' }}>Input: </label>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    value={mlpLayers[0]}
                    onChange={(e) => {
                      const newLayers = [...mlpLayers];
                      newLayers[0] = parseInt(e.target.value) || 2;
                      setMlpLayers(newLayers);
                    }}
                    style={{ width: '60px', padding: '0.25rem', borderRadius: '4px', border: '1px solid #d1d5db' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.75rem' }}>Hidden: </label>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={mlpLayers[1]}
                    onChange={(e) => {
                      const newLayers = [...mlpLayers];
                      newLayers[1] = parseInt(e.target.value) || 3;
                      setMlpLayers(newLayers);
                    }}
                    style={{ width: '60px', padding: '0.25rem', borderRadius: '4px', border: '1px solid #d1d5db' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.75rem' }}>Output: </label>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    value={mlpLayers[2]}
                    onChange={(e) => {
                      const newLayers = [...mlpLayers];
                      newLayers[2] = parseInt(e.target.value) || 1;
                      setMlpLayers(newLayers);
                    }}
                    style={{ width: '60px', padding: '0.25rem', borderRadius: '4px', border: '1px solid #d1d5db' }}
                  />
                </div>
              </div>
            </div>

            {renderMLPArchitecture()}
          </div>
        )}

        {/* Unified Learning Journey Section */}
        {activeSection === 'journey' && (
          <Topic4UnifiedLearning />
        )}

        {/* Multivariate GD Section */}
        {activeSection === 'multivariate' && (
          <GradientDescentComplete />
        )}
      </div>
    </div>
  );
}
