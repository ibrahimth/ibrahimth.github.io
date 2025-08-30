import React, { useMemo, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ------------------------------------------------------------
// COE 292 — Topic 1 Interactive Studio
// Tabs: Key Ideas • Playgrounds • Quiz
// - Animated concept cards (What is AI? Agents, Utility, Turing Test)
// - River Crossing Simulator (Lion–Lamb–Grass)
// - Feature Picking Game (Good vs Bad features for fish ID)
// - Mini Turing Test Guesser
// - Click & Select Quiz with instant feedback
// ------------------------------------------------------------

// ---------- UI Helpers ----------
const TabButton = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    className={
      "px-4 py-2 rounded-2xl text-sm mx-1 transition-all border " +
      (active
        ? "bg-blue-600 text-white shadow-md border-blue-600"
        : "bg-white hover:bg-blue-50 text-blue-700 border-blue-200")
    }
  >
    {children}
  </button>
);

const Card = ({ title, children }) => (
  <div className="bg-white rounded-2xl shadow-md p-5 border border-slate-200">
    {title && <h3 className="text-lg font-semibold mb-3">{title}</h3>}
    {children}
  </div>
);

const Pill = ({ children }) => (
  <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-slate-100 text-slate-700 border border-slate-200">
    {children}
  </span>
);

// ---------- Animated Concept Cards ----------
const conceptCards = [
  {
    title: "What is AI?",
    pts: [
      "Systems that do tasks needing human intelligence (perception, language, decision-making).",
      "Definitions differ — some focus on 'thinking like humans,' others on 'acting rationally'.",
      "1950s → expert systems → ML → deep learning → generative AI",
      "In the 1960s, ELIZA (a simple chatbot) felt like real AI.",
      "Today, it looks trivial compared to ChatGPT."

    ]
  },
  {
    title: "Application Areas",
    pts: [
      "Natural Language Processing (NLP) (Automatic Speech Recognition (ASR), Text-to-Speech (TTS), translation, search, ChatGPT)",
      "Computer Vision (recognition, detection)",
      "Robotics & Autonomy (planning, assistive robots)",
      "Applied AI (scheduling, route planning, diagnosis, fraud, recommendations)"
    ]
  },
  {
    title: "Core Vocabulary",
    pts: [
      "Agent: perceives via sensors and acts via actuators.",
      "Utility: numeric preference for outcomes.",
      "Rational Agent: maximizes expected utility given uncertainty."
    ]
  },
{
  title: "Turing Test (and limits)",
  pts: [
    "Judge converses with human & machine; if indistinguishable, machine ‘passes’.",
    "Limits: subjective, language-only, incentivizes deception, ignores perception/motor skills.",
    "Deception = fooling someone into believing the machine is smart.",
    "Intelligence = actually having the ability to reason, understand, and learn.",
    { 
      text: "👉 A useful way to think about it:",
      sub: [
        "Passing the Turing Test ≠ being intelligent.",
        "It just means you’re good at faking intelligence."
      ]
    }
  ]
},

 {
   title: "Two Solution Strategies",
   pts: [
     "Use Perception + Representation to expose constraints (river crossing).",
  "Generate & Test with good features (fish ID)."
    ]
  }
];

const ConceptCarousel = () => {
  const [idx, setIdx] = useState(0);
  const [autoplayMs, setAutoplayMs] = useState(0); // 0 = Off (manual only)

  useEffect(() => {
    if (!autoplayMs) return; // stay manual unless explicitly enabled
    const t = setInterval(() => setIdx((i) => (i + 1) % conceptCards.length), autoplayMs);
    return () => clearInterval(t);
  }, [autoplayMs]);

  const prev = () => setIdx((i) => (i - 1 + conceptCards.length) % conceptCards.length);
  const next = () => setIdx((i) => (i + 1) % conceptCards.length);

  const card = conceptCards[idx];
  return (
    <div className="relative">
      <AnimatePresence mode="wait">
        <motion.div
          key={card.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.35 }}
          className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 rounded-2xl p-6"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold">{card.title}</h3>
            <Pill>
              {idx + 1} / {conceptCards.length}
            </Pill>
          </div>
          <ul className="mt-3 list-disc pl-6 space-y-1 text-slate-700">
  {Array.isArray(card.pts) &&
    card.pts.map((p, i) => {
      if (typeof p === "string") return <li key={i}>{p}</li>;
      // object with sub-points
      const text = p?.text ?? "";
      const sub = Array.isArray(p?.sub) ? p.sub : [];
      return (
        <li key={i}>
          {text}
          {sub.length > 0 && (
            <ul className="mt-1 list-[circle] pl-6 space-y-1">
              {sub.map((s, j) => (
                <li key={j}>{s}</li>
              ))}
            </ul>
          )}
        </li>
      );
    })}
</ul>
 

 
        </motion.div>
      </AnimatePresence>

      <div className="mt-3 flex items-center justify-between flex-wrap gap-2">
        {/* Dots */}
        <div className="flex justify-center gap-2">
          {conceptCards.map((_, i) => (
            <button
              key={i}
              onClick={() => setIdx(i)}
              className={
                "w-2.5 h-2.5 rounded-full border " +
                (i === idx ? "bg-blue-600 border-blue-600" : "bg-white border-blue-300")
              }
              aria-label={`Go to slide ${i + 1}`}
            />
          ))}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button onClick={prev} className="px-3 py-1.5 rounded-xl border bg-white hover:bg-slate-50">← Prev</button>
          <button onClick={next} className="px-3 py-1.5 rounded-xl border bg-white hover:bg-slate-50">Next →</button>
          <div className="flex items-center gap-1 text-xs">
            <span className="text-slate-600">Autoplay:</span>
            <select
              value={autoplayMs}
              onChange={(e) => setAutoplayMs(Number(e.target.value))}
              className="border rounded-lg px-2 py-1 text-xs bg-white"
            >
              <option value={0}>Off</option>
              <option value={12000}>Every 12s</option>
              <option value={20000}>Every 20s</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};

// ---------- River Crossing Simulator ----------
// Entities: Person (P), Lion (L), Lamb/Sheep (S), Grass (G)
// Bank: 'L' (left) or 'R' (right)
const RC_START = { boat: "L", P: "L", L: "L", S: "L", G: "L" };

function violates({ P, L, S, G }) {
  // If person absent on a bank, check predator/prey pairs
  const left = { P: P === "L", L: L === "L", S: S === "L", G: G === "L" };
  const right = { P: P === "R", L: L === "R", S: S === "R", G: G === "R" };
  // Lion eats Sheep OR Sheep eats Grass without P
  const badLeft = (!left.P && ((left.L && left.S) || (left.S && left.G)));
  const badRight = (!right.P && ((right.L && right.S) || (right.S && right.G)));
  return badLeft || badRight;
}

const RC_SOLN = [
  // One valid optimal sequence (7 crossings)
  ["S"],
  [],
  ["L"],
  ["S"],
  ["G"],
  [],
  ["S"]
];

const RiverCrossing = () => {
  const [state, setState] = useState(RC_START);
  const [history, setHistory] = useState([RC_START]);
  const [status, setStatus] = useState("Make a safe crossing.");
  const [auto, setAuto] = useState(false);

  const reset = () => {
    setState(RC_START);
    setHistory([RC_START]);
    setStatus("Make a safe crossing.");
    setAuto(false);
  };

  const canMove = (item) => state[item] === state.boat; // item must be on same bank as boat

  const move = (items = []) => {
    // Person always rides; optionally take one item (L/S/G)
    const next = { ...state };
    const to = state.boat === "L" ? "R" : "L";
    next.P = to;
    if (items[0]) next[items[0]] = to;
    next.boat = to;

    if (violates(next)) {
      setStatus("⛔ Unsafe: someone got eaten! Try a different move.");
    } else {
      setStatus("✅ Safe move.");
      setState(next);
      setHistory((h) => [...h, next]);
    }
  };

  useEffect(() => {
    if (!auto) return;
    // play the RC_SOLN from current position
    let i = 0;
    const id = setInterval(() => {
      if (i >= RC_SOLN.length) {
        setAuto(false);
        return clearInterval(id);
      }
      move(RC_SOLN[i]);
      i++;
    }, 900);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auto]);

  const win = state.L === "R" && state.S === "R" && state.G === "R" && state.P === "R";

  const Entity = ({ name, emoji }) => (
    <div className="flex items-center gap-1 text-sm">
      <span className="text-xl">{emoji}</span>
      <span className="font-medium">{name}</span>
    </div>
  );

  const MoveBtn = ({ label, payload, disabled }) => (
    <button
      onClick={() => move(payload)}
      disabled={disabled}
      className={
        "px-3 py-2 rounded-xl text-sm border transition-all " +
        (disabled
          ? "bg-slate-100 text-slate-400 border-slate-200"
          : "bg-emerald-600 text-white border-emerald-600 hover:brightness-110")
      }
    >
      {label}
    </button>
  );

  return (
    <Card title="River Crossing Simulator (Lion–Lamb–Grass)">
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <div className="flex items-center gap-3 mb-3">
            <Entity name="Person" emoji="🧍" />
            <Entity name="Lion" emoji="🦁" />
            <Entity name="Lamb" emoji="🐑" />
            <Entity name="Grass" emoji="🌿" />
          </div>
          <div className="relative h-40 md:h-44 rounded-2xl overflow-hidden border bg-blue-50">
            {/* River */}
            <div className="absolute inset-0 grid grid-cols-2">
              {/* Left bank */}
              <div className="flex flex-col items-center justify-center gap-2">
                {state.P === "L" && <motion.div layout>🧍</motion.div>}
                {state.L === "L" && <motion.div layout>🦁</motion.div>}
                {state.S === "L" && <motion.div layout>🐑</motion.div>}
                {state.G === "L" && <motion.div layout>🌿</motion.div>}
              </div>
              {/* Right bank */}
              <div className="flex flex-col items-center justify-center gap-2">
                {state.P === "R" && <motion.div layout>🧍</motion.div>}
                {state.L === "R" && <motion.div layout>🦁</motion.div>}
                {state.S === "R" && <motion.div layout>🐑</motion.div>}
                {state.G === "R" && <motion.div layout>🌿</motion.div>}
              </div>
            </div>
            {/* Boat */}
            <motion.div
              className="absolute bottom-3 w-20 text-center"
              animate={{ left: state.boat === "L" ? 20 : 220 }}
              transition={{ type: "spring", stiffness: 120, damping: 12 }}
            >
              <div className="text-2xl">🛶</div>
              <div className="text-[10px] text-slate-600">boat</div>
            </motion.div>
          </div>

          <div className="mt-3 text-sm flex flex-wrap gap-2">
            <MoveBtn label="Cross alone" payload={[]} disabled={false} />
            <MoveBtn label="Take Lion" payload={["L"]} disabled={!canMove("L")} />
            <MoveBtn label="Take Lamb" payload={["S"]} disabled={!canMove("S")} />
            <MoveBtn label="Take Grass" payload={["G"]} disabled={!canMove("G")} />
            <button
              onClick={() => setAuto(true)}
              className="px-3 py-2 rounded-xl text-sm border bg-indigo-600 text-white border-indigo-600 hover:brightness-110"
            >
              ▶ Show one optimal sequence
            </button>
            <button onClick={reset} className="px-3 py-2 rounded-xl text-sm border bg-white hover:bg-slate-50">
              ↺ Reset
            </button>
          </div>
          <div className="mt-2 text-sm">
            <span className="font-medium">Status:</span> {win ? "🎉 Goal reached!" : status}
          </div>
          <div className="mt-2 text-xs text-slate-500">Rule: Never leave 🦁 with 🐑 or 🐑 with 🌿 without 🧍 present.</div>
        </div>
        <div>
          <Card title="Why this models Topic 1 well">
            <ul className="list-disc pl-5 text-sm space-y-1 text-slate-700">
              <li><b>Representation</b>: model states (who’s on which bank) and actions (crossings).</li>
              <li><b>Constraints</b>: detect illegal states (predator–prey without the person).</li>
              <li><b>Perception → Reasoning</b>: visualize and prune bad states before acting.</li>
              <li><b>Rationality</b>: choose actions that maximize progress toward the goal.</li>
            </ul>
          </Card>
          <div className="mt-3 text-xs text-slate-500">Tip: Try to finish in 7 crossings.</div>
        </div>
      </div>
    </Card>
  );
};

// ---------- Feature Picking Game (Good vs Bad features) ----------
const featureBank = [
  { txt: "Square head shape", good: true, why: "Uncommon shape → discriminative." },
  { txt: "Lives in water", good: false, why: "All fish do → not discriminative." },
  { txt: "Mouth points downward", good: true, why: "Not typical → helpful." },
  { txt: "Swims near rocks", good: false, why: "Too common → weak signal." },
  { txt: "Side eyes close to head edge", good: true, why: "Distinct placement → useful feature." },
  { txt: "Only photographed in bright light", good: false, why: "Artifact of taking photos, not the fish." }
];

const FeaturePicker = () => {
  const [answers, setAnswers] = useState({});
  const [showWhy, setShowWhy] = useState(false);
  const correct = Object.entries(answers).filter(([i, v]) => featureBank[+i].good === (v === "good")).length;

  return (
    <Card title="Generate & Test: Pick GOOD features for fish ID">
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <ul className="space-y-2">
            {featureBank.map((f, i) => {
              const choice = answers[i];
              const isCorrect = choice && (f.good === (choice === "good"));
              return (
                <li key={i} className="p-3 rounded-xl border bg-white flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-medium">{f.txt}</div>
                    {showWhy && (
                      <div className={"text-xs mt-1 " + (f.good ? "text-emerald-600" : "text-rose-600")}>{f.why}</div>
                    )}
                  </div>
                  <div className="flex gap-1">
                    <button
                      onClick={() => setAnswers({ ...answers, [i]: "good" })}
                      className={
                        "px-2 py-1 rounded-lg border text-xs " +
                        (choice === "good" ? "bg-emerald-600 text-white border-emerald-600" : "bg-white hover:bg-emerald-50")
                      }
                    >
                      Good
                    </button>
                    <button
                      onClick={() => setAnswers({ ...answers, [i]: "bad" })}
                      className={
                        "px-2 py-1 rounded-lg border text-xs " +
                        (choice === "bad" ? "bg-rose-600 text-white border-rose-600" : "bg-white hover:bg-rose-50")
                      }
                    >
                      Bad
                    </button>
                    {choice && (
                      <Pill>{isCorrect ? "✓" : "✗"}</Pill>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
          <div className="mt-3 flex items-center gap-2 text-sm">
            <Pill>Score: {correct} / {featureBank.length}</Pill>
            <button onClick={() => setShowWhy((v) => !v)} className="px-3 py-1.5 rounded-xl border text-sm bg-white hover:bg-slate-50">
              {showWhy ? "Hide" : "Show"} explanations
            </button>
          </div>
        </div>
        <div>
          <Card title="Why features matter">
            <p className="text-sm text-slate-700">
              In <b>Generate & Test</b>, we hypothesize a label, then check it.
              The quality of features controls how quickly we converge on the right answer.
              Good features are <i>discriminative</i> — they separate the target class from others.
            </p>
            <ul className="list-disc pl-5 text-sm mt-2 text-slate-700 space-y-1">
              <li>Good: rare traits, shape cues, distinctive textures/patterns.</li>
              <li>Bad: universal facts ("lives in water"), photographer artifacts.</li>
            </ul>
          </Card>
        </div>
      </div>
    </Card>
  );
};

// ---------- Mini Turing Test Guesser ----------
const turingPairs = [
  {
    q: "Explain Wi‑Fi in one short sentence.",
    options: [
      "It's like invisible internet that lets your phone talk to a box (router) in your home.",
      "Wi‑Fi is an 802.11 wireless LAN standard using radio frequencies."
    ],
    human: 0
  },
  {
    q: "Describe how to make a peanut butter sandwich.",
    options: [
      "Spread peanut butter on one slice, jam on the other, put them together, and eat.",
      "Acquire materials; execute assembly procedure per food preparation protocol."
    ],
    human: 0
  },
  {
    q: "What did you do this weekend?",
    options: [
      "Met friends, watched a movie, and fell asleep halfway through the credits.",
      "I completed recreational activities and achieved intended relaxation goals."
    ],
    human: 0
  },
  {
    q: "Tell a simple joke.",
    options: [
      "Why did the math book look sad? Because it had too many problems.",
      "Initiating humor: semantic pun about numeric entities."
    ],
    human: 0
  }
];

const TuringGuesser = () => {
  const [choices, setChoices] = useState({});
  const [show, setShow] = useState(false);
  const score = Object.entries(choices).filter(([i, v]) => turingPairs[+i].human === v).length;

  return (
    <Card title="Mini Turing Test – Can you tell who’s human?">
      <ol className="space-y-3">
        {turingPairs.map((p, i) => (
          <li key={i} className="p-3 rounded-xl border bg-white">
            <div className="text-sm font-medium mb-2">Q{i + 1}. {p.q}</div>
            <div className="grid md:grid-cols-2 gap-2">
              {p.options.map((opt, j) => (
                <button
                  key={j}
                  onClick={() => setChoices({ ...choices, [i]: j })}
                  className={
                    "text-left p-3 rounded-xl border text-sm " +
                    (choices[i] === j ? "bg-blue-600 text-white border-blue-600" : "bg-white hover:bg-blue-50")
                  }
                >
                  {opt}
                </button>
              ))}
            </div>
          </li>
        ))}
      </ol>
      <div className="mt-3 flex items-center gap-2 text-sm">
        <Pill>Selected: {Object.keys(choices).length} / {turingPairs.length}</Pill>
        <button onClick={() => setShow(true)} className="px-3 py-1.5 rounded-xl border bg-white hover:bg-slate-50">Reveal</button>
        {show && <Pill>Score: {score} / {turingPairs.length}</Pill>}
      </div>
      {show && (
        <div className="mt-2 text-xs text-slate-600">
          Reflection: Passing a text-only test doesn’t mean reasoning or perception—this illustrates Topic 1 limitations.
        </div>
      )}
    </Card>
  );
};

// ---------- Quiz (with shuffle on load) ----------
const QUIZ = [
  {
    q: "Which option best defines a rational agent?",
    a: [
      "One that imitates human behavior regardless of outcome",
      "One that maximizes expected utility given its beliefs and actions",
      "One that never makes mistakes"
    ],
    correct: 1
  },
  {
    q: "Pick the AI task NOT assessed by the classic Turing Test:",
    a: ["Dialogue coherence", "Visual perception", "Deception ability"],
    correct: 1
  },
  {
    q: "In the river crossing puzzle, which state is illegal?",
    a: [
      "🧍 with 🐑 on left; 🦁 with 🌿 on right",
      "🐑 with 🌿 alone on one bank",
      "🧍 crosses alone while all are on the same bank"
    ],
    correct: 1
  },
  {
    q: "Good feature for fish identification?",
    a: ["Lives in water", "Square head shape", "Photographed in bright light"],
    correct: 1
  },
  {
    q: "Utility in AI is…",
    a: ["a rule list", "a numeric preference measure for outcomes", "a data structure for sensors"],
    correct: 1
  }
];

// Pure helper: in-place Fisher–Yates clone
function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Create a shuffled quiz with each question's answers shuffled and correct index remapped
function prepareQuizData(base) {
  const withShuffledAnswers = base.map((q) => {
    const indices = q.a.map((_, idx) => idx);
    const perm = shuffleArray(indices);
    const shuffledA = perm.map((k) => q.a[k]);
    const newCorrect = perm.indexOf(q.correct);
    return { ...q, a: shuffledA, correct: newCorrect };
  });
  return shuffleArray(withShuffledAnswers);
}

const Quiz = () => {
  const quizData = useMemo(() => prepareQuizData(QUIZ), []); // shuffle once when Interactive Studio loads
  const [sel, setSel] = useState({});
  const [show, setShow] = useState(false);
  const score = quizData.reduce((s, q, i) => s + (sel[i] === q.correct ? 1 : 0), 0);

  return (
    <Card title="Topic 1 Checkpoint Quiz (click & select)">
      <ol className="space-y-4">
        {quizData.map((q, i) => (
          <li key={i} className="p-3 rounded-xl border bg-white">
            <div className="text-sm font-medium mb-2">Q{i + 1}. {q.q}</div>
            <div className="grid md:grid-cols-3 gap-2">
              {q.a.map((choice, j) => {
                const picked = sel[i] === j;
                const isCorrect = q.correct === j;
                const revealStyle = show ? (isCorrect ? "ring-2 ring-emerald-500" : picked ? "ring-2 ring-rose-500" : "") : "";
                return (
                  <button
                    key={j}
                    onClick={() => setSel({ ...sel, [i]: j })}
                    className={`text-left p-3 rounded-xl border text-sm bg-white hover:bg-slate-50 ${picked ? "border-blue-600" : ""} ${revealStyle}`}
                  >
                    {choice}
                  </button>
                );
              })}
            </div>
          </li>
        ))}
      </ol>
      <div className="mt-3 flex items-center gap-2 text-sm">
        <Pill>
          Answered: {Object.keys(sel).length} / {quizData.length}
        </Pill>
        <button onClick={() => setShow(true)} className="px-3 py-1.5 rounded-xl border bg-white hover:bg-slate-50">Check answers</button>
        {show && <Pill>Score: {score} / {quizData.length}</Pill>}
      </div>
    </Card>
  );
};

// ---------- Lightweight Self‑Tests (runtime) ----------
function runSelfTests() {
  const results = [];
  // 1) violates should flag predator/prey when P absent
  results.push({
    name: "violates flags lion-sheep without person",
    pass: violates({ P: "R", L: "L", S: "L", G: "R" }) === true
  });
  // 2) violates should allow safe config
  results.push({
    name: "violates allows safe state",
    pass: violates({ P: "L", L: "L", S: "R", G: "R" }) === true ? false : true
  });
  // 3) RC_SOLN achieves goal from start
  let sim = { ...RC_START };
  const apply = (st, items = []) => {
    const to = st.boat === "L" ? "R" : "L";
    const nx = { ...st, P: to, boat: to };
    if (items[0]) nx[items[0]] = to;
    return nx;
  };
  RC_SOLN.forEach((m) => (sim = apply(sim, m)));
  const goal = sim.L === "R" && sim.S === "R" && sim.G === "R" && sim.P === "R";
  results.push({ name: "RC_SOLN reaches goal", pass: goal });
  // 4) Feature bank has expected good count
  const goodCount = featureBank.filter((f) => f.good).length;
  results.push({ name: "Feature bank good-count = 3", pass: goodCount === 3 });
  // 5) Quiz length
  results.push({ name: "QUIZ length = 5", pass: Array.isArray(QUIZ) && QUIZ.length === 5 });
  // 6) Quiz shuffle preserves correctness mapping
  const shuffled = prepareQuizData(QUIZ);
  const mappingOk = shuffled.every((q) => {
    const orig = QUIZ.find((o) => o.q === q.q);
    if (!orig) return false;
    const correctLabel = orig.a[orig.correct];
    return q.a[q.correct] === correctLabel;
  });
  results.push({ name: "Quiz shuffle preserves correct answers", pass: mappingOk });
  // 7) Each shuffled question keeps same number of answers
  const countsOk = shuffled.every((q) => {
    const orig = QUIZ.find((o) => o.q === q.q);
    return orig && q.a.length === orig.a.length;
  });
  results.push({ name: "Quiz shuffle preserves answer counts", pass: countsOk });
  return results;
}

// ---------- Main ----------
export default function Topic1Studio() {
  const [tab, setTab] = useState("ideas");
  const [tests, setTests] = useState([]);
  const [showTests, setShowTests] = useState(false);

  useEffect(() => {
    const res = runSelfTests();
    setTests(res);
    // Also log for developers
    try {
      console.table(res.map((r) => ({ test: r.name, pass: r.pass })));
    } catch {}
  }, []);

  const passed = tests.filter((t) => t.pass).length;

  return (
    <div className="min-h-[100vh] w-full bg-slate-50 p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        <header className="mb-5">
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">COE 292 • Topic 1 Interactive Studio</h1>
          <p className="text-slate-600 mt-1 text-sm md:text-base">
            Explore core ideas of AI via animations and mini‑games. Then test yourself with a quick quiz.
          </p>
        </header>

        <nav className="mb-4">
          <TabButton active={tab === "ideas"} onClick={() => setTab("ideas")}>Key Ideas</TabButton>
          <TabButton active={tab === "play"} onClick={() => setTab("play")}>Playgrounds</TabButton>
          <TabButton active={tab === "quiz"} onClick={() => setTab("quiz")}>Quiz</TabButton>
        </nav>

        {tab === "ideas" && (
          <div className="grid gap-4">
            <ConceptCarousel />
            <Card title="What to look for in Topic 1">
              <ul className="list-disc pl-6 text-slate-700 text-sm space-y-1">
                <li>Different lenses on AI (acting/thinking humanly vs. rationally).</li>
                <li>Turing Test origins + its limitations.</li>
                <li>Agents, utility, rational decisions under uncertainty.</li>
                <li>Problem‑solving strategies: Perception‑based modeling; Generate‑and‑Test.</li>
              </ul>
            </Card>
          </div>
        )}

        {tab === "play" && (
          <div className="grid gap-5">
            <RiverCrossing />
            <FeaturePicker />
            <TuringGuesser />
          </div>
        )}

        {tab === "quiz" && <Quiz />}

        <footer className="mt-8 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span>Built for COE 292 • Topic 1.</span>
            <button onClick={() => setShowTests((v) => !v)} className="px-2 py-1 rounded-lg border bg-white hover:bg-slate-50">
              {showTests ? "Hide self‑tests" : `Show self‑tests (${passed}/${tests.length})`}
            </button>
          </div>
          {showTests && (
            <div className="mt-2">
              <ul className="text-[11px] text-slate-600 list-disc pl-5">
                {tests.map((t, i) => (
                  <li key={i} className={t.pass ? "text-emerald-600" : "text-rose-600"}>
                    {t.pass ? "✓" : "✗"} {t.name}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </footer>
      </div>
    </div>
  );
}
