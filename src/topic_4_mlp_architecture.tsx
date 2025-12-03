import React, { useState, useEffect } from 'react';

/**
 * Topic 4 — MLP Architecture Visualizer
 * Interactive visualization of Multi-Layer Perceptron with:
 * - Adjustable network architecture (layers and neurons)
 * - Forward propagation visualization
 * - Weight visualization
 * - Activation function selection
 */

interface Layer {
  neurons: number;
  activation: 'relu' | 'sigmoid' | 'tanh' | 'linear';
}

export default function MLPArchitecture() {
  const [layers, setLayers] = useState<Layer[]>([
    { neurons: 3, activation: 'linear' },
    { neurons: 4, activation: 'relu' },
    { neurons: 4, activation: 'relu' },
    { neurons: 2, activation: 'sigmoid' }
  ]);

  const [highlightedConnection, setHighlightedConnection] = useState<{
    from: [number, number];
    to: [number, number];
  } | null>(null);

  const addLayer = () => {
    if (layers.length < 6) {
      const newLayers = [...layers];
      newLayers.splice(layers.length - 1, 0, { neurons: 4, activation: 'relu' });
      setLayers(newLayers);
    }
  };

  const removeLayer = (index: number) => {
    if (layers.length > 2 && index > 0 && index < layers.length - 1) {
      setLayers(layers.filter((_, i) => i !== index));
    }
  };

  const updateLayer = (index: number, field: keyof Layer, value: any) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    setLayers(newLayers);
  };

  const getActivationColor = (activation: string) => {
    switch (activation) {
      case 'relu': return '#ef4444';
      case 'sigmoid': return '#3b82f6';
      case 'tanh': return '#8b5cf6';
      case 'linear': return '#6b7280';
      default: return '#6b7280';
    }
  };

  const renderNetwork = () => {
    const width = 800;
    const height = 500;
    const layerSpacing = width / (layers.length + 1);
    const maxNeurons = Math.max(...layers.map(l => l.neurons));

    return (
      <svg width={width} height={height} style={{ background: '#f9fafb', borderRadius: '8px' }}>
        {/* Draw connections */}
        {layers.map((layer, layerIdx) => {
          if (layerIdx === 0) return null;

          const prevLayer = layers[layerIdx - 1];
          const prevX = layerSpacing * layerIdx;
          const currX = layerSpacing * (layerIdx + 1);
          const prevSpacing = height / (prevLayer.neurons + 1);
          const currSpacing = height / (layer.neurons + 1);

          return prevLayer.neurons <= 8 && layer.neurons <= 8 ? (
            <g key={`connections-${layerIdx}`}>
              {Array.from({ length: prevLayer.neurons }).map((_, prevNeuronIdx) => {
                const prevY = prevSpacing * (prevNeuronIdx + 1);

                return Array.from({ length: layer.neurons }).map((_, currNeuronIdx) => {
                  const currY = currSpacing * (currNeuronIdx + 1);
                  const isHighlighted =
                    highlightedConnection?.from[0] === layerIdx - 1 &&
                    highlightedConnection?.from[1] === prevNeuronIdx &&
                    highlightedConnection?.to[0] === layerIdx &&
                    highlightedConnection?.to[1] === currNeuronIdx;

                  return (
                    <line
                      key={`${prevNeuronIdx}-${currNeuronIdx}`}
                      x1={prevX}
                      y1={prevY}
                      x2={currX}
                      y2={currY}
                      stroke={isHighlighted ? '#f59e0b' : '#d1d5db'}
                      strokeWidth={isHighlighted ? 2 : 1}
                      opacity={isHighlighted ? 1 : 0.3}
                    />
                  );
                });
              })}
            </g>
          ) : null;
        })}

        {/* Draw neurons */}
        {layers.map((layer, layerIdx) => {
          const x = layerSpacing * (layerIdx + 1);
          const spacing = height / (layer.neurons + 1);

          return (
            <g key={`layer-${layerIdx}`}>
              {Array.from({ length: layer.neurons }).map((_, neuronIdx) => {
                const y = spacing * (neuronIdx + 1);

                return (
                  <g key={`neuron-${layerIdx}-${neuronIdx}`}>
                    <circle
                      cx={x}
                      cy={y}
                      r={layer.neurons > 10 ? 8 : 12}
                      fill={getActivationColor(layer.activation)}
                      stroke="#fff"
                      strokeWidth={2}
                      style={{ cursor: 'pointer' }}
                      onMouseEnter={() => {
                        if (layerIdx > 0) {
                          setHighlightedConnection({
                            from: [layerIdx - 1, Math.floor(neuronIdx * layers[layerIdx - 1].neurons / layer.neurons)],
                            to: [layerIdx, neuronIdx]
                          });
                        }
                      }}
                      onMouseLeave={() => setHighlightedConnection(null)}
                    />
                  </g>
                );
              })}

              {/* Layer label */}
              <text
                x={x}
                y={height - 20}
                textAnchor="middle"
                fontSize="12"
                fill="#6b7280"
                fontWeight="600"
              >
                {layerIdx === 0 ? 'Input' : layerIdx === layers.length - 1 ? 'Output' : `Hidden ${layerIdx}`}
              </text>
              <text
                x={x}
                y={height - 5}
                textAnchor="middle"
                fontSize="11"
                fill="#9ca3af"
              >
                {layer.neurons} neurons
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  const totalParams = layers.reduce((sum, layer, idx) => {
    if (idx === 0) return 0;
    return sum + (layers[idx - 1].neurons * layer.neurons + layer.neurons);
  }, 0);

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

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem', fontWeight: '700' }}>
          🎓 Topic 4 — MLP Architecture
        </h2>
        <p style={{ margin: '0 0 1rem 0', color: '#6b7280' }}>
          Explore Multi-Layer Perceptron architecture. Add layers, adjust neurons, and visualize connections.
        </p>

        <div style={{ marginBottom: '1rem' }}>
          <button onClick={addLayer} style={primaryButtonStyle} disabled={layers.length >= 6}>
            + Add Hidden Layer
          </button>
          <span style={{ marginLeft: '1rem', fontSize: '0.875rem', color: '#6b7280' }}>
            Total Parameters: <strong>{totalParams.toLocaleString()}</strong>
          </span>
        </div>

        {/* Network Visualization */}
        <div style={{ marginBottom: '1.5rem', overflow: 'auto' }}>
          {renderNetwork()}
        </div>

        {/* Layer Controls */}
        <div style={{ display: 'grid', gap: '0.75rem' }}>
          {layers.map((layer, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                padding: '0.75rem',
                background: '#f9fafb',
                borderRadius: '8px',
                border: '1px solid #e5e7eb',
              }}
            >
              <span style={{ fontWeight: '600', minWidth: '80px', fontSize: '0.875rem' }}>
                {idx === 0 ? 'Input' : idx === layers.length - 1 ? 'Output' : `Hidden ${idx}`}
              </span>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
                Neurons:
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={layer.neurons}
                  onChange={(e) => updateLayer(idx, 'neurons', parseInt(e.target.value) || 1)}
                  style={{
                    width: '60px',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '4px',
                    border: '1px solid #d1d5db',
                  }}
                  disabled={idx === 0 || idx === layers.length - 1}
                />
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
                Activation:
                <select
                  value={layer.activation}
                  onChange={(e) => updateLayer(idx, 'activation', e.target.value)}
                  style={{
                    padding: '0.25rem 0.5rem',
                    borderRadius: '4px',
                    border: '1px solid #d1d5db',
                    background: 'white',
                  }}
                  disabled={idx === 0}
                >
                  {idx === 0 ? (
                    <option value="linear">linear</option>
                  ) : (
                    <>
                      <option value="relu">ReLU</option>
                      <option value="sigmoid">Sigmoid</option>
                      <option value="tanh">Tanh</option>
                      <option value="linear">Linear</option>
                    </>
                  )}
                </select>
              </label>

              {idx > 0 && idx < layers.length - 1 && (
                <button
                  onClick={() => removeLayer(idx)}
                  style={{
                    ...buttonStyle,
                    marginLeft: 'auto',
                    color: '#ef4444',
                    borderColor: '#fecaca',
                  }}
                >
                  Remove
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Legend */}
        <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#f9fafb', borderRadius: '8px' }}>
          <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.875rem', fontWeight: '600' }}>Activation Functions:</h4>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', fontSize: '0.875rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ef4444' }} />
              <span>ReLU</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#3b82f6' }} />
              <span>Sigmoid</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#8b5cf6' }} />
              <span>Tanh</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#6b7280' }} />
              <span>Linear</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
