import React, { useState, useMemo } from 'react';

/**
 * Topic 5 — CNN Visualizer (No Padding, No Bias)
 * Interactive visualization of Convolutional Neural Networks with:
 * - Convolution operations (no padding, no bias)
 * - Pooling operations (max and average)
 * - Feature map visualization
 * - Kernel/filter editing
 */

type PoolingType = 'max' | 'average';

interface ConvLayer {
  type: 'conv';
  kernelSize: number;
  numFilters: number;
  stride: number;
}

interface PoolLayer {
  type: 'pool';
  poolSize: number;
  poolType: PoolingType;
  stride: number;
}

type Layer = ConvLayer | PoolLayer;

// Simple 8x8 input image (grayscale values 0-255)
const createSampleImage = (): number[][] => {
  return [
    [50, 50, 50, 200, 200, 50, 50, 50],
    [50, 50, 200, 200, 200, 200, 50, 50],
    [50, 200, 200, 50, 50, 200, 200, 50],
    [200, 200, 50, 50, 50, 50, 200, 200],
    [200, 200, 50, 50, 50, 50, 200, 200],
    [50, 200, 200, 50, 50, 200, 200, 50],
    [50, 50, 200, 200, 200, 200, 50, 50],
    [50, 50, 50, 200, 200, 50, 50, 50],
  ];
};

// Convolution operation (no padding, no bias)
function convolve(
  input: number[][],
  kernel: number[][],
  stride: number = 1
): number[][] {
  const inputSize = input.length;
  const kernelSize = kernel.length;
  const outputSize = Math.floor((inputSize - kernelSize) / stride) + 1;
  const output: number[][] = [];

  for (let i = 0; i < outputSize; i++) {
    output[i] = [];
    for (let j = 0; j < outputSize; j++) {
      let sum = 0;
      for (let ki = 0; ki < kernelSize; ki++) {
        for (let kj = 0; kj < kernelSize; kj++) {
          const inputRow = i * stride + ki;
          const inputCol = j * stride + kj;
          sum += input[inputRow][inputCol] * kernel[ki][kj];
        }
      }
      // No bias added, just the convolution result
      output[i][j] = sum;
    }
  }

  return output;
}

// Max pooling
function maxPool(input: number[][], poolSize: number, stride: number): number[][] {
  const inputSize = input.length;
  const outputSize = Math.floor((inputSize - poolSize) / stride) + 1;
  const output: number[][] = [];

  for (let i = 0; i < outputSize; i++) {
    output[i] = [];
    for (let j = 0; j < outputSize; j++) {
      let max = -Infinity;
      for (let pi = 0; pi < poolSize; pi++) {
        for (let pj = 0; pj < poolSize; pj++) {
          const inputRow = i * stride + pi;
          const inputCol = j * stride + pj;
          if (inputRow < inputSize && inputCol < inputSize) {
            max = Math.max(max, input[inputRow][inputCol]);
          }
        }
      }
      output[i][j] = max;
    }
  }

  return output;
}

// Average pooling
function averagePool(input: number[][], poolSize: number, stride: number): number[][] {
  const inputSize = input.length;
  const outputSize = Math.floor((inputSize - poolSize) / stride) + 1;
  const output: number[][] = [];

  for (let i = 0; i < outputSize; i++) {
    output[i] = [];
    for (let j = 0; j < outputSize; j++) {
      let sum = 0;
      let count = 0;
      for (let pi = 0; pi < poolSize; pi++) {
        for (let pj = 0; pj < poolSize; pj++) {
          const inputRow = i * stride + pi;
          const inputCol = j * stride + pj;
          if (inputRow < inputSize && inputCol < inputSize) {
            sum += input[inputRow][inputCol];
            count++;
          }
        }
      }
      output[i][j] = sum / count;
    }
  }

  return output;
}

// Predefined kernels
const KERNELS: Record<string, { kernel: number[][], name: string }> = {
  identity: {
    name: 'Identity',
    kernel: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
  },
  edgeHorizontal: {
    name: 'Edge (Horizontal)',
    kernel: [
      [-1, -1, -1],
      [0, 0, 0],
      [1, 1, 1],
    ],
  },
  edgeVertical: {
    name: 'Edge (Vertical)',
    kernel: [
      [-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1],
    ],
  },
  sharpen: {
    name: 'Sharpen',
    kernel: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  blur: {
    name: 'Blur',
    kernel: [
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9],
      [1/9, 1/9, 1/9],
    ],
  },
};

export default function CNNVisualizer() {
  const [inputImage] = useState<number[][]>(createSampleImage());
  const [selectedKernel, setSelectedKernel] = useState<string>('edgeHorizontal');
  const [convStride, setConvStride] = useState(1);
  const [poolType, setPoolType] = useState<PoolingType>('max');
  const [poolSize, setPoolSize] = useState(2);
  const [poolStride, setPoolStride] = useState(2);
  const [showPooling, setShowPooling] = useState(true);

  const kernel = KERNELS[selectedKernel].kernel;

  // Apply convolution
  const convOutput = useMemo(() => {
    return convolve(inputImage, kernel, convStride);
  }, [inputImage, kernel, convStride]);

  // Apply pooling
  const poolOutput = useMemo(() => {
    if (!showPooling) return null;
    if (poolType === 'max') {
      return maxPool(convOutput, poolSize, poolStride);
    } else {
      return averagePool(convOutput, poolSize, poolStride);
    }
  }, [convOutput, poolType, poolSize, poolStride, showPooling]);

  const renderMatrix = (
    matrix: number[][],
    title: string,
    colorScale: boolean = true
  ) => {
    const size = matrix.length;
    const cellSize = Math.min(40, 320 / size);

    // Normalize values for color display
    const flatValues = matrix.flat();
    const min = Math.min(...flatValues);
    const max = Math.max(...flatValues);

    return (
      <div style={{ marginBottom: '1.5rem' }}>
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.875rem', fontWeight: '600' }}>
          {title} ({size}×{size})
        </h4>
        <div
          style={{
            display: 'inline-grid',
            gridTemplateColumns: `repeat(${size}, ${cellSize}px)`,
            gap: '2px',
            padding: '0.5rem',
            background: '#f3f4f6',
            borderRadius: '8px',
          }}
        >
          {matrix.map((row, i) =>
            row.map((val, j) => {
              const normalized = max > min ? (val - min) / (max - min) : 0.5;
              const colorValue = Math.round(normalized * 255);
              const bgColor = colorScale
                ? `rgb(${colorValue}, ${colorValue}, ${colorValue})`
                : '#fff';

              return (
                <div
                  key={`${i}-${j}`}
                  style={{
                    width: `${cellSize}px`,
                    height: `${cellSize}px`,
                    background: bgColor,
                    border: '1px solid #d1d5db',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: cellSize > 30 ? '0.65rem' : '0',
                    fontWeight: '600',
                    color: normalized > 0.5 ? '#000' : '#fff',
                  }}
                >
                  {cellSize > 30 ? Math.round(val) : ''}
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  };

  const renderKernel = (kernel: number[][], title: string) => {
    const size = kernel.length;
    const cellSize = 50;

    return (
      <div style={{ marginBottom: '1rem' }}>
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.875rem', fontWeight: '600' }}>
          {title} ({size}×{size})
        </h4>
        <div
          style={{
            display: 'inline-grid',
            gridTemplateColumns: `repeat(${size}, ${cellSize}px)`,
            gap: '2px',
            padding: '0.5rem',
            background: '#fef3c7',
            borderRadius: '8px',
            border: '2px solid #fbbf24',
          }}
        >
          {kernel.map((row, i) =>
            row.map((val, j) => (
              <div
                key={`${i}-${j}`}
                style={{
                  width: `${cellSize}px`,
                  height: `${cellSize}px`,
                  background: 'white',
                  border: '1px solid #fbbf24',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.75rem',
                  fontWeight: '700',
                  color: '#92400e',
                }}
              >
                {val.toFixed(2)}
              </div>
            ))
          )}
        </div>
      </div>
    );
  };

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

  const controlsStyle: React.CSSProperties = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '1rem',
    marginBottom: '1.5rem',
    padding: '1rem',
    background: '#f9fafb',
    borderRadius: '8px',
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem', fontWeight: '700' }}>
          🎓 Topic 5 — CNN (No Padding, No Bias)
        </h2>
        <p style={{ margin: '0 0 1rem 0', color: '#6b7280', fontSize: '0.875rem' }}>
          Explore Convolutional Neural Networks with convolution and pooling operations.
          <strong> No padding and no bias</strong> are used in this visualization.
        </p>

        {/* Controls */}
        <div style={controlsStyle}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Kernel Type:
            </span>
            <select
              value={selectedKernel}
              onChange={(e) => setSelectedKernel(e.target.value)}
              style={{
                padding: '0.5rem',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
                background: 'white',
                fontSize: '0.875rem',
              }}
            >
              {Object.entries(KERNELS).map(([key, { name }]) => (
                <option key={key} value={key}>
                  {name}
                </option>
              ))}
            </select>
          </label>

          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Conv Stride:
            </span>
            <input
              type="number"
              min={1}
              max={2}
              value={convStride}
              onChange={(e) => setConvStride(parseInt(e.target.value) || 1)}
              style={{
                width: '80px',
                padding: '0.5rem',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
                fontSize: '0.875rem',
              }}
            />
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={showPooling}
              onChange={(e) => setShowPooling(e.target.checked)}
              style={{ width: '16px', height: '16px' }}
            />
            <span style={{ fontSize: '0.875rem', fontWeight: '600' }}>Enable Pooling</span>
          </label>

          {showPooling && (
            <>
              <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
                  Pool Type:
                </span>
                <select
                  value={poolType}
                  onChange={(e) => setPoolType(e.target.value as PoolingType)}
                  style={{
                    padding: '0.5rem',
                    borderRadius: '6px',
                    border: '1px solid #d1d5db',
                    background: 'white',
                    fontSize: '0.875rem',
                  }}
                >
                  <option value="max">Max Pooling</option>
                  <option value="average">Average Pooling</option>
                </select>
              </label>

              <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
                  Pool Size:
                </span>
                <input
                  type="number"
                  min={2}
                  max={3}
                  value={poolSize}
                  onChange={(e) => setPoolSize(parseInt(e.target.value) || 2)}
                  style={{
                    width: '80px',
                    padding: '0.5rem',
                    borderRadius: '6px',
                    border: '1px solid #d1d5db',
                    fontSize: '0.875rem',
                  }}
                />
              </label>

              <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
                  Pool Stride:
                </span>
                <input
                  type="number"
                  min={1}
                  max={3}
                  value={poolStride}
                  onChange={(e) => setPoolStride(parseInt(e.target.value) || 2)}
                  style={{
                    width: '80px',
                    padding: '0.5rem',
                    borderRadius: '6px',
                    border: '1px solid #d1d5db',
                    fontSize: '0.875rem',
                  }}
                />
              </label>
            </>
          )}
        </div>

        {/* Visualization */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '2rem',
            marginTop: '1.5rem',
          }}
        >
          <div>
            {renderMatrix(inputImage, 'Input Image', true)}
          </div>

          <div>
            {renderKernel(kernel, 'Convolution Kernel')}
          </div>

          <div>
            {renderMatrix(convOutput, 'After Convolution', true)}
          </div>

          {showPooling && poolOutput && (
            <div>
              {renderMatrix(
                poolOutput,
                `After ${poolType === 'max' ? 'Max' : 'Average'} Pooling`,
                true
              )}
            </div>
          )}
        </div>

        {/* Info */}
        <div style={{ marginTop: '2rem', padding: '1rem', background: '#eff6ff', borderRadius: '8px', border: '1px solid #bfdbfe' }}>
          <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.875rem', fontWeight: '600', color: '#1e40af' }}>
            ℹ️ Key Points:
          </h4>
          <ul style={{ margin: 0, paddingLeft: '1.5rem', color: '#1e40af', fontSize: '0.875rem', lineHeight: 1.6 }}>
            <li><strong>No Padding:</strong> Output size = (Input size - Kernel size) / Stride + 1</li>
            <li><strong>No Bias:</strong> Only convolution multiplication, no bias term added</li>
            <li><strong>Max Pooling:</strong> Takes maximum value from each pooling window</li>
            <li><strong>Average Pooling:</strong> Takes average of values in each pooling window</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
