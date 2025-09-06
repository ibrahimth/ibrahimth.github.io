import { useState } from 'react'
import Topic1Studio from './topic_1_interactive_studio_coe_292.jsx'
import Topic2Studio from './topic_2_interactive_studio_coe_292.tsx'
import SVMPlayground from './topic_3_svm_playground.tsx'

function TabButton({ id, isActive, onClick, children }) {
  const base = {
    padding: '0.5rem 0.9rem',
    borderRadius: '9999px',
    border: '1px solid',
    fontSize: '0.9rem',
    cursor: 'pointer',
    transition: 'all 120ms ease',
    marginRight: '0.5rem',
    background: 'white',
  }
  const active = {
    background: '#1d4ed8',       // blue-700
    color: 'white',
    borderColor: '#1d4ed8',
    boxShadow: '0 1px 6px rgba(0,0,0,0.08)',
  }
  const inactive = {
    color: '#1e3a8a',            // blue-900-ish
    borderColor: '#bfdbfe',      // blue-200
  }
  const style = { ...base, ...(isActive ? active : inactive) }

  return (
    <button
      id={id}
      role="tab"
      aria-selected={isActive}
      aria-controls={`${id}-panel`}
      onClick={onClick}
      style={style}
    >
      {children}
    </button>
  )
}

export default function App() {
  const [active, setActive] = useState('topic3')

  return (
    <div style={{ padding: '1rem' }}>
      {/* Tab list */}
      <nav
        role="tablist"
        aria-label="COE 292 Interactive Studios"
        style={{
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.25rem',
          marginBottom: '1rem',
          background: '#f1f5f9',        // slate-100
          padding: '0.5rem',
          borderRadius: '9999px',
          border: '1px solid #e2e8f0',  // slate-200
        }}
      >
        <TabButton
          id="tab-topic1"
          isActive={active === 'topic1'}
          onClick={() => setActive('topic1')}
        >
          Topic 1 — Interactive Studio
        </TabButton>

        <TabButton
          id="tab-topic2"
          isActive={active === 'topic2'}
          onClick={() => setActive('topic2')}
        >
          Topic 2 — Playground
        </TabButton>

        <TabButton
          id="tab-topic3"
          isActive={active === 'topic3'}
          onClick={() => setActive('topic3')}
        >
          Topic 3 — k-NN Classification
        </TabButton>
      </nav>

      {/* Panels */}
      <section
        id="tab-topic1-panel"
        role="tabpanel"
        aria-labelledby="tab-topic1"
        hidden={active !== 'topic1'}
        style={{ display: active === 'topic1' ? 'block' : 'none' }}
      >
        <Topic1Studio />
      </section>

      <section
        id="tab-topic2-panel"
        role="tabpanel"
        aria-labelledby="tab-topic2"
        hidden={active !== 'topic2'}
        style={{ display: active === 'topic2' ? 'block' : 'none' }}
      >
        <Topic2Studio />
      </section>

      <section
        id="tab-topic3-panel"
        role="tabpanel"
        aria-labelledby="tab-topic3"
        hidden={active !== 'topic3'}
        style={{ display: active === 'topic3' ? 'block' : 'none' }}
      >
        <SVMPlayground />
      </section>
    </div>
  )
}
