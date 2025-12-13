import React, { useState, useMemo, useEffect } from 'react';

/**
 * Topic 5 — CNN Visualizer (Configurable)
 * Interactive visualization of Convolutional Neural Networks with:
 * - Multiple input channels
 * - Configurable input size, filter size, number of filters
 * - Convolution operations (with/without padding, with/without bias)
 * - Pooling operations (max and average)
 * - Parameter counting and visualization
 */

type PoolingType = 'max' | 'average' | 'none';

// Generate random channel data
const generateChannel = (size: number): number[][] => {
  const channel: number[][] = [];
  for (let i = 0; i < size; i++) {
    channel[i] = [];
    for (let j = 0; j < size; j++) {
      channel[i][j] = Math.floor(Math.random() * 9) + 1;
    }
  }
  return channel;
};

// Convolution operation
function convolve(
  input: number[][],
  kernel: number[][],
  stride: number = 1,
  padding: number = 0,
  bias: number = 0
): number[][] {
  const inputSize = input.length;
  const kernelSize = kernel.length;

  // Add padding if specified
  let paddedInput = input;
  if (padding > 0) {
    const paddedSize = inputSize + 2 * padding;
    paddedInput = Array(paddedSize).fill(0).map(() => Array(paddedSize).fill(0));
    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < inputSize; j++) {
        paddedInput[i + padding][j + padding] = input[i][j];
      }
    }
  }

  const paddedSize = paddedInput.length;
  const outputSize = Math.floor((paddedSize - kernelSize) / stride) + 1;
  const output: number[][] = [];

  for (let i = 0; i < outputSize; i++) {
    output[i] = [];
    for (let j = 0; j < outputSize; j++) {
      let sum = 0;
      for (let ki = 0; ki < kernelSize; ki++) {
        for (let kj = 0; kj < kernelSize; kj++) {
          const inputRow = i * stride + ki;
          const inputCol = j * stride + kj;
          sum += paddedInput[inputRow][inputCol] * kernel[ki][kj];
        }
      }
      output[i][j] = sum + bias;
    }
  }

  return output;
}

// ReLU activation
function relu(input: number[][]): number[][] {
  return input.map(row => row.map(val => Math.max(0, val)));
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

export default function CNNVisualizer() {
  // Configuration
  const [inputSize, setInputSize] = useState(6);
  const [numInputChannels, setNumInputChannels] = useState(3);
  const [filterSize, setFilterSize] = useState(4);
  const [numFilters, setNumFilters] = useState(2);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [useBias, setUseBias] = useState(false);
  const [useReLU, setUseReLU] = useState(true);
  const [poolType, setPoolType] = useState<PoolingType>('max');
  const [poolSize, setPoolSize] = useState(2);
  const [poolStride, setPoolStride] = useState(2);

  // Generate input channels
  const [inputChannels, setInputChannels] = useState<number[][][]>(() =>
    Array(numInputChannels).fill(0).map(() => generateChannel(inputSize))
  );

  // Generate random filters (one set per output channel)
  const [filters, setFilters] = useState<number[][][][]>(() =>
    Array(numFilters).fill(0).map(() =>
      Array(numInputChannels).fill(0).map(() =>
        Array(filterSize).fill(0).map(() =>
          Array(filterSize).fill(0).map(() => Math.random() * 2 - 0.5)
        )
      )
    )
  );

  const [biases] = useState<number[]>(() =>
    Array(numFilters).fill(0).map(() => Math.random() * 2 - 1)
  );

  // Regenerate data when dimensions change
  useEffect(() => {
    setInputChannels(Array(numInputChannels).fill(0).map(() => generateChannel(inputSize)));
  }, [inputSize, numInputChannels]);

  useEffect(() => {
    setFilters(Array(numFilters).fill(0).map(() =>
      Array(numInputChannels).fill(0).map(() =>
        Array(filterSize).fill(0).map(() =>
          Array(filterSize).fill(0).map(() => Math.random() * 2 - 0.5)
        )
      )
    ));
  }, [filterSize, numFilters, numInputChannels]);

  // Regenerate data when size changes
  const regenerateData = () => {
    setInputChannels(Array(numInputChannels).fill(0).map(() => generateChannel(inputSize)));
    setFilters(Array(numFilters).fill(0).map(() =>
      Array(numInputChannels).fill(0).map(() =>
        Array(filterSize).fill(0).map(() =>
          Array(filterSize).fill(0).map(() => Math.random() * 2 - 0.5)
        )
      )
    ));
  };

  // Calculate output size after convolution
  // Use actual data dimensions to prevent crash during state updates
  const currentInputSize = inputChannels[0]?.length || inputSize;
  const currentFilterSize = filters[0]?.[0]?.length || filterSize;
  const convOutputSize = Math.floor((currentInputSize + 2 * padding - currentFilterSize) / stride) + 1;

  // Calculate output size after pooling
  const poolOutputSize = poolType !== 'none'
    ? Math.floor((convOutputSize - poolSize) / poolStride) + 1
    : convOutputSize;

  // Calculate parameters
  const numFilterChannels = numInputChannels * numFilters;
  const paramsPerFilter = filterSize * filterSize;
  const numNeurons = numFilters;
  const inputsPerNeuron = filterSize * filterSize * numInputChannels;
  const totalParams = useBias
    ? (filterSize * filterSize * numInputChannels * numFilters) + numFilters
    : (filterSize * filterSize * numInputChannels * numFilters);

  const safeOutputSize = Math.max(1, convOutputSize);
  const totalNeurons = safeOutputSize * safeOutputSize * numFilters;
  const params = filterSize * filterSize * numInputChannels * numFilters;

  // Apply convolution for all filters
  const convOutputs = useMemo(() => {
    return filters.map((filterSet, filterIdx) => {
      // For each output channel, sum convolutions across all input channels
      const channelOutputs = filterSet.map((filter, channelIdx) => {
        return convolve(inputChannels[channelIdx], filter, stride, padding, 0);
      });

      // Sum all channel outputs
      const summedOutput: number[][] = Array(convOutputSize).fill(0).map(() =>
        Array(convOutputSize).fill(0)
      );

      channelOutputs.forEach(channelOutput => {
        for (let i = 0; i < convOutputSize; i++) {
          for (let j = 0; j < convOutputSize; j++) {
            summedOutput[i][j] += channelOutput[i][j];
          }
        }
      });

      // Add bias if enabled
      if (useBias) {
        const bias = biases[filterIdx];
        for (let i = 0; i < convOutputSize; i++) {
          for (let j = 0; j < convOutputSize; j++) {
            summedOutput[i][j] += bias;
          }
        }
      }

      return summedOutput;
    });
  }, [inputChannels, filters, stride, padding, useBias, biases, convOutputSize, filterSize]);

  // Apply ReLU
  const reluOutputs = useMemo(() => {
    return useReLU ? convOutputs.map(output => relu(output)) : convOutputs;
  }, [convOutputs, useReLU]);

  // Apply pooling
  const poolOutputs = useMemo(() => {
    if (poolType === 'none') return reluOutputs;

    return reluOutputs.map(output => {
      if (poolType === 'max') {
        return maxPool(output, poolSize, poolStride);
      } else {
        return averagePool(output, poolSize, poolStride);
      }
    });
  }, [reluOutputs, poolType, poolSize, poolStride]);

  const renderMatrix = (
    matrix: number[][],
    title: string,
    colorScale: boolean = true,
    maxCellSize: number = 40
  ) => {
    const size = matrix.length;
    // Responsive max width: smaller on mobile
    const baseWidth = window.innerWidth < 768 ? 300 : 600;
    const cellSize = Math.min(maxCellSize, baseWidth / size);

    const flatValues = matrix.flat();
    const min = Math.min(...flatValues);
    const max = Math.max(...flatValues);

    return (
      <div style={{ marginBottom: '1rem' }}>
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.75rem', fontWeight: '600', color: '#374151' }}>
          {title}
        </h4>
        <div
          style={{
            display: 'inline-grid',
            gridTemplateColumns: `repeat(${size}, ${cellSize}px)`,
            gap: '1px',
            padding: '0.25rem',
            background: '#e5e7eb',
            borderRadius: '4px',
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
                    // Lowered threshold from 25 to 20 for visibility
                    fontSize: cellSize > 20 ? '0.6rem' : '0',
                    fontWeight: '600',
                    color: normalized > 0.5 ? '#000' : '#fff',
                  }}
                >
                  {cellSize > 20 ? val.toFixed(1) : ''}
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  };

  const containerStyle: React.CSSProperties = {
    padding: window.innerWidth < 768 ? '0.5rem' : '1.5rem',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  };

  const cardStyle: React.CSSProperties = {
    background: 'white',
    borderRadius: '12px',
    padding: window.innerWidth < 768 ? '0.75rem' : '1.5rem',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    marginBottom: '1rem',
  };

  const controlGroupStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: window.innerWidth < 768
      ? 'repeat(auto-fit, minmax(140px, 1fr))'
      : 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: window.innerWidth < 768 ? '0.5rem' : '1rem',
    marginBottom: '1.5rem',
    padding: window.innerWidth < 768 ? '0.5rem' : '1rem',
    background: '#f9fafb',
    borderRadius: '8px',
  };

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '0.5rem',
    borderRadius: '6px',
    border: '1px solid #d1d5db',
    fontSize: '0.875rem',
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem', fontWeight: '700' }}>
          🎓 Topic 5 — CNN Visualizer
        </h2>
        <p style={{ margin: '0 0 1rem 0', color: '#6b7280', fontSize: '0.875rem' }}>
          Explore Convolutional Neural Networks with configurable parameters.
        </p>

        {/* Architecture Configuration */}
        <div style={controlGroupStyle}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Input Size (NxN):
            </span>
            <input
              type="number"
              min={3}
              max={12}
              value={inputSize}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) setInputSize(Math.min(Math.max(val, 3), 12));
              }}
              style={inputStyle}
            />
          </label>

          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Input Channels:
            </span>
            <input
              type="number"
              min={1}
              max={5}
              value={numInputChannels}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) setNumInputChannels(Math.min(Math.max(val, 1), 5));
              }}
              style={inputStyle}
            />
          </label>

          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Filter Size (KxK):
            </span>
            <input
              type="number"
              min={2}
              max={5}
              value={filterSize}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) setFilterSize(Math.min(Math.max(val, 2), 5));
              }}
              style={inputStyle}
            />
          </label>

          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Number of Filters:
            </span>
            <input
              type="number"
              min={1}
              max={4}
              value={numFilters}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) setNumFilters(Math.min(Math.max(val, 1), 4));
              }}
              style={inputStyle}
            />
          </label>

          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Stride:
            </span>
            <input
              type="number"
              min={1}
              max={3}
              value={stride}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) setStride(Math.min(Math.max(val, 1), 3));
              }}
              style={inputStyle}
            />
          </label>

          <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#6b7280' }}>
              Padding:
            </span>
            <input
              type="number"
              min={0}
              max={2}
              value={padding}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) setPadding(Math.min(Math.max(val, 0), 2));
              }}
              style={inputStyle}
            />
          </label>
        </div>

        {/* Options */}
        <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={useBias}
              onChange={(e) => setUseBias(e.target.checked)}
              style={{ width: '16px', height: '16px' }}
            />
            <span style={{ fontSize: '0.875rem', fontWeight: '600' }}>Use Bias</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input
              type="checkbox"
              checked={useReLU}
              onChange={(e) => setUseReLU(e.target.checked)}
              style={{ width: '16px', height: '16px' }}
            />
            <span style={{ fontSize: '0.875rem', fontWeight: '600' }}>Use ReLU</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.875rem', fontWeight: '600' }}>Pooling:</span>
            <select
              value={poolType}
              onChange={(e) => setPoolType(e.target.value as PoolingType)}
              style={{
                padding: '0.25rem 0.5rem',
                borderRadius: '4px',
                border: '1px solid #d1d5db',
                background: 'white',
                fontSize: '0.875rem',
              }}
            >
              <option value="none">None</option>
              <option value="max">Max</option>
              <option value="average">Average</option>
            </select>
          </label>

          {poolType !== 'none' && (
            <>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.875rem', fontWeight: '600' }}>Pool Size:</span>
                <input
                  type="number"
                  min={2}
                  max={3}
                  value={poolSize}
                  onChange={(e) => setPoolSize(parseInt(e.target.value) || 2)}
                  style={{ width: '60px', padding: '0.25rem', borderRadius: '4px', border: '1px solid #d1d5db' }}
                />
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.875rem', fontWeight: '600' }}>Pool Stride:</span>
                <input
                  type="number"
                  min={1}
                  max={3}
                  value={poolStride}
                  onChange={(e) => setPoolStride(parseInt(e.target.value) || 2)}
                  style={{ width: '60px', padding: '0.25rem', borderRadius: '4px', border: '1px solid #d1d5db' }}
                />
              </label>
            </>
          )}

          <button
            onClick={regenerateData}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '6px',
              border: 'none',
              background: '#3b82f6',
              color: 'white',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: '500',
            }}
          >
            Regenerate Data
          </button>
        </div>

        {/* Calculations Panel */}
        <div style={{
          marginTop: '2rem',
          padding: '1.5rem',
          background: '#f8fafc',
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <h3 style={{ margin: '0 0 1rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            📊 Calculations (like the example in the image):
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
            <div style={{ fontSize: '0.9rem', lineHeight: '1.6', color: '#334155' }}>
              <p><strong>The output size</strong> = ({inputSize} + 2×{padding} - {filterSize})/{stride} + 1 = <strong>{safeOutputSize}</strong></p>
              <p>➤ <strong>The number of needed filters is</strong> {numFilters}</p>
              <p>➤ <strong>The number of needed filter channels is</strong> {numInputChannels}×{numFilters} = {numInputChannels * numFilters}, each of size {filterSize}×{filterSize}</p>
              <p>➤ <strong>The number of neurons is</strong> {safeOutputSize}×{safeOutputSize}×{numFilters} = {totalNeurons}</p>
              <p>➤ <strong>The number of inputs to each neuron is</strong> {filterSize}×{filterSize}×{numInputChannels} = {inputsPerNeuron}</p>
              <p>➤ <strong>The number of network parameters to be trained (ignoring biases) is</strong> {filterSize}×{filterSize}×{numInputChannels}×{numFilters} = {params}</p>
              <p>➤ <strong>After max pooling (1×1, stride=2):</strong> output size = {Math.ceil(safeOutputSize / 2)}×{Math.ceil(safeOutputSize / 2)}</p>
            </div>
          </div>
        </div>

        {/* Visualization */}
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', fontWeight: '700' }}>
            Layer Visualization
          </h3>

          {/* Input Channels */}
          <div style={{ marginBottom: '2rem' }}>
            <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.95rem', fontWeight: '600', color: '#2563eb' }}>
              Input ({inputSize}×{inputSize}×{numInputChannels})
            </h4>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              {inputChannels.map((channel, idx) => (
                <div key={idx}>
                  {renderMatrix(channel, `Channel ${idx + 1}`, true, 35)}
                </div>
              ))}
            </div>
          </div>

          {/* Filters (Kernels) */}
          <div style={{ marginBottom: '2rem' }}>
            <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.95rem', fontWeight: '600', color: '#9333ea' }}>
              Filters (Kernels) ({filterSize}×{filterSize}×{numInputChannels} per filter)
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {filters.map((filterSet, filterIdx) => (
                <div key={filterIdx} style={{ padding: '1rem', background: '#f3e8ff', borderRadius: '8px' }}>
                  <h5 style={{ margin: '0 0 0.5rem 0', fontSize: '0.85rem', fontWeight: '600', color: '#7e22ce' }}>
                    Filter {filterIdx + 1}
                  </h5>
                  <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    {filterSet.map((kernel, channelIdx) => (
                      <div key={channelIdx}>
                        {renderMatrix(kernel, `Channel ${channelIdx + 1}`, true, 35)}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Convolution Output */}
          <div style={{ marginBottom: '2rem' }}>
            <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.95rem', fontWeight: '600', color: '#16a34a' }}>
              After Convolution {useReLU && '+ ReLU'} ({convOutputSize}×{convOutputSize}×{numFilters})
            </h4>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              {reluOutputs.map((output, idx) => (
                <div key={idx}>
                  {renderMatrix(output, `Filter ${idx + 1} Output`, true, 35)}
                </div>
              ))}
            </div>
          </div>

          {/* Pooling Output */}
          {poolType !== 'none' && (
            <div style={{ marginBottom: '2rem' }}>
              <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.95rem', fontWeight: '600', color: '#dc2626' }}>
                After {poolType === 'max' ? 'Max' : 'Average'} Pooling ({poolOutputSize}×{poolOutputSize}×{numFilters})
              </h4>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                {poolOutputs.map((output, idx) => (
                  <div key={idx}>
                    {renderMatrix(output, `Filter ${idx + 1} Pooled`, true, 35)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div >
    </div >
  );
}
