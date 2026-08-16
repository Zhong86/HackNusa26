import React, { useState, useMemo } from "react";

/**
 * Sentinel Loop — analyst console
 *
 * Submits an email to POST /emails (blocking, no SSE) and renders the
 * resulting trace as a vertical pipeline stepper, ending in a stamped
 * verdict badge. Skipped stages (e.g. gather_context/reason when Layer 1
 * was confident) are shown as explicitly skipped, not just omitted, so
 * the branch the graph took is visible at a glance.
 */

const API_BASE = "http://localhost:8000";

const SAMPLE_EMAILS = [
  {
    label: "Safe — GitHub PR notification",
    email: {
      sender: "no-reply@github.com",
      display_name: "GitHub",
      subject: "[sentinel-loop] New pull request",
      body: "A new pull request was opened on sentinel-loop by teammate. Review it when you get a chance.",
      urls: ["https://github.com/org/sentinel-loop/pull/17"],
    },
  },
  {
    label: "Uncertain — Invoice confirmation",
    email: {
      sender: "billing@acme-invoices.net",
      display_name: "Acme Billing",
      subject: "Invoice #4471 - action needed",
      body: "Please confirm your billing details to avoid a delay processing invoice #4471. Let us know if you have questions.",
      urls: ["http://acme-invoices.net/invoice/4471"],
    },
  },
  {
    label: "Malicious — Fake PayPal alert",
    email: {
      sender: "support@paypa1-secure.com",
      display_name: "PayPal Support",
      subject: "Your account has been suspended",
      body: "Dear user, we detected unusual activity. Click here urgently to verify your account or it will be suspended immediately.",
      urls: ["http://paypa1-secure.com/verify-now"],
    },
  },
];

const EMPTY_EMAIL = { sender: "", display_name: "", subject: "", body: "", urls: "" };

const VERDICT_STYLE = {
  allow: { color: "var(--safe)", label: "ALLOW", angle: "-6deg" },
  quarantine: { color: "var(--danger)", label: "QUARANTINE", angle: "4deg" },
  escalate: { color: "var(--warn)", label: "ESCALATE", angle: "-3deg" },
};

function ScoreMeter({ score }) {
  const pct = Math.round(score * 100);
  const zone = score < 0.2 ? "safe" : score > 0.75 ? "danger" : "warn";
  return (
    <div className="meter">
      <div className="meter-track">
        <div className="meter-zone meter-zone-safe" style={{ left: "0%", width: "20%" }} />
        <div className="meter-zone meter-zone-warn" style={{ left: "20%", width: "55%" }} />
        <div className="meter-zone meter-zone-danger" style={{ left: "75%", width: "25%" }} />
        <div className="meter-fill" style={{ left: `calc(${pct}% - 1px)` }} />
      </div>
      <div className="meter-readout">
        <span className={`meter-value zone-${zone}`}>{score.toFixed(3)}</span>
        <span className="meter-caption">layer 1 phishing probability</span>
      </div>
    </div>
  );
}

function Stage({ index, title, status, children }) {
  // status: 'done' | 'skipped' | 'pending'
  return (
    <div className={`stage stage-${status}`}>
      <div className="stage-rail">
        <div className="stage-dot">{status === "skipped" ? "–" : index}</div>
        <div className="stage-line" />
      </div>
      <div className="stage-body">
        <div className="stage-title">
          {title}
          {status === "skipped" && <span className="stage-skip-tag">not reached</span>}
        </div>
        {status !== "skipped" && children}
      </div>
    </div>
  );
}

function KeyVal({ label, value }) {
  return (
    <div className="kv">
      <span className="kv-key">{label}</span>
      <span className="kv-val">{String(value)}</span>
    </div>
  );
}

export default function SentinelLoopApp() {
  const [form, setForm] = useState(SAMPLE_EMAILS[1].email);
  const [urlsText, setUrlsText] = useState(SAMPLE_EMAILS[1].email.urls.join("\n"));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const loadSample = (sample) => {
    setForm(sample.email);
    setUrlsText(sample.email.urls.join("\n"));
    setResult(null);
    setError(null);
  };

  const clearForm = () => {
    setForm(EMPTY_EMAIL);
    setUrlsText("");
    setResult(null);
    setError(null);
  };

  const updateField = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }));

  const submit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        email: {
          sender: form.sender,
          display_name: form.display_name,
          subject: form.subject,
          body: form.body,
          urls: urlsText.split("\n").map((u) => u.trim()).filter(Boolean),
        },
      };
      const res = await fetch(`${API_BASE}/emails`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server responded ${res.status}: ${text.slice(0, 300)}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "Request failed. Is the backend running on :8000?");
    } finally {
      setLoading(false);
    }
  };

  const state = result?.state;
  const trace = result?.trace || [];
  const wasUncertain = trace.includes("gather_context_node");
  const verdictKey = state?.final_verdict;
  const verdictStyle = verdictKey ? VERDICT_STYLE[verdictKey] : null;

  const canSubmit = useMemo(
    () => form.sender.trim() && form.subject.trim() && form.body.trim() && !loading,
    [form, loading]
  );

  return (
    <div className="app">
      <style>{CSS}</style>

      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <span className="brand-name">SENTINEL LOOP</span>
        </div>
        <div className="brand-sub">email detection console — layer 1 / layer 2 trace</div>
      </header>

      <main className="layout">
        {/* LEFT: input */}
        <section className="panel input-panel">
          <div className="panel-header">
            <h2>Email under review</h2>
            <p>Paste a message or load a sample to see how the pipeline routes it.</p>
          </div>

          <div className="samples">
            {SAMPLE_EMAILS.map((s) => (
              <button key={s.label} className="sample-chip" onClick={() => loadSample(s)}>
                {s.label}
              </button>
            ))}
            <button className="sample-chip sample-chip-clear" onClick={clearForm}>
              Clear
            </button>
          </div>

          <label className="field">
            <span>Sender address</span>
            <input value={form.sender} onChange={updateField("sender")} placeholder="support@example.com" />
          </label>

          <label className="field">
            <span>Display name</span>
            <input value={form.display_name} onChange={updateField("display_name")} placeholder="Example Support" />
          </label>

          <label className="field">
            <span>Subject</span>
            <input value={form.subject} onChange={updateField("subject")} placeholder="Your account needs attention" />
          </label>

          <label className="field">
            <span>Body</span>
            <textarea rows={6} value={form.body} onChange={updateField("body")} placeholder="Message content..." />
          </label>

          <label className="field">
            <span>URLs (one per line)</span>
            <textarea
              rows={2}
              value={urlsText}
              onChange={(e) => setUrlsText(e.target.value)}
              placeholder="http://example.com/verify"
            />
          </label>

          <button className="submit-btn" disabled={!canSubmit} onClick={submit}>
            {loading ? "Running pipeline…" : "Run through Sentinel Loop"}
          </button>

          {error && <div className="error-box">{error}</div>}
        </section>

        {/* RIGHT: trace + verdict */}
        <section className="panel trace-panel">
          <div className="panel-header">
            <h2>Pipeline trace</h2>
            <p>{result ? `thread ${result.thread_id?.slice(0, 8)}` : "Awaiting a run."}</p>
          </div>

          {!result && !loading && (
            <div className="empty-state">
              <span className="empty-glyph">○</span>
              <p>Submit an email on the left to see it move through scoring, routing, context, and reasoning.</p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <span className="empty-glyph spin">◐</span>
              <p>Running the graph…</p>
            </div>
          )}

          {result && state && (
            <>
              <div className="stages">
                <Stage index={1} title="Layer 1 — classifier score" status="done">
                  {state.layer1_score && <ScoreMeter score={state.layer1_score.score} />}
                </Stage>

                <Stage
                  index={2}
                  title={wasUncertain ? "Routed to Layer 2 (uncertain zone)" : "Routed to direct decision (confident)"}
                  status="done"
                >
                  <p className="stage-note">
                    {wasUncertain
                      ? "Score fell inside the uncertain band, so context gathering and reasoning ran."
                      : "Score was outside the uncertain band, so Layer 2 was skipped entirely."}
                  </p>
                </Stage>

                <Stage index={3} title="Layer 2 — context gathered" status={wasUncertain ? "done" : "skipped"}>
                  {state.context && (
                    <div className="context-grid">
                      <div className="context-card">
                        <div className="context-card-title">Sender history</div>
                        <KeyVal label="seen before" value={state.context.sender_history?.seen_before} />
                        <KeyVal label="prior flags" value={state.context.sender_history?.prior_flag_count} />
                      </div>
                      <div className="context-card">
                        <div className="context-card-title">Domain age</div>
                        <KeyVal label="age (days)" value={state.context.domain_age?.age_days} />
                        <KeyVal label="newly registered" value={state.context.domain_age?.newly_registered} />
                      </div>
                      <div className="context-card">
                        <div className="context-card-title">Threat intel</div>
                        <KeyVal label="domain flagged" value={state.context.threat_intel?.domain_flagged} />
                        <KeyVal
                          label="malicious URLs"
                          value={state.context.threat_intel?.matched_malicious_urls?.length ?? 0}
                        />
                      </div>
                    </div>
                  )}
                </Stage>

                <Stage index={4} title="Layer 2 — reasoning" status={wasUncertain ? "done" : "skipped"}>
                  {state.reasoning && (
                    <div className="reasoning-block">
                      <p className="reasoning-text">{state.reasoning.justification}</p>
                      <div className="evidence-list">
                        {state.reasoning.evidence_used?.map((e, i) => (
                          <span key={i} className="evidence-tag">
                            {e}
                          </span>
                        ))}
                      </div>
                      <div className="mitre-row">
                        {state.reasoning.mitre_technique_ids?.map((id) => (
                          <span key={id} className="mitre-tag">
                            {id}
                          </span>
                        ))}
                        <span className="confidence-tag">confidence {state.reasoning.confidence?.toFixed(2)}</span>
                      </div>
                    </div>
                  )}
                </Stage>
              </div>

              {verdictStyle && (
                <div className="verdict-wrap">
                  <div
                    className="verdict-stamp"
                    style={{ "--stamp-color": verdictStyle.color, "--stamp-angle": verdictStyle.angle }}
                  >
                    {verdictStyle.label}
                  </div>
                  <p className="verdict-justification">{state.final_justification}</p>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0B0E11;
  --panel: #131820;
  --panel-border: #232B36;
  --text: #E4E7EB;
  --text-dim: #7C8798;
  --safe: #5EEAD4;
  --warn: #F59E0B;
  --danger: #F43F5E;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'Inter', sans-serif;
}

* { box-sizing: border-box; }

.app {
  min-height: 100vh;
  background: var(--bg);
  background-image:
    radial-gradient(circle at 20% 0%, rgba(94,234,212,0.05), transparent 40%),
    radial-gradient(circle at 80% 100%, rgba(244,63,94,0.04), transparent 40%);
  color: var(--text);
  font-family: var(--sans);
}

.topbar {
  padding: 20px 32px;
  border-bottom: 1px solid var(--panel-border);
  display: flex;
  align-items: baseline;
  gap: 16px;
}
.brand { display: flex; align-items: center; gap: 8px; }
.brand-mark { color: var(--safe); font-size: 18px; }
.brand-name {
  font-family: var(--mono);
  font-weight: 700;
  letter-spacing: 0.12em;
  font-size: 14px;
}
.brand-sub { color: var(--text-dim); font-size: 12px; font-family: var(--mono); }

.layout {
  display: grid;
  grid-template-columns: 380px 1fr;
  gap: 20px;
  padding: 24px 32px 48px;
  max-width: 1280px;
  margin: 0 auto;
}
@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; padding: 16px; }
}

.panel {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  padding: 20px;
}

.panel-header h2 {
  margin: 0 0 4px;
  font-size: 15px;
  font-weight: 600;
}
.panel-header p {
  margin: 0 0 18px;
  color: var(--text-dim);
  font-size: 12.5px;
  font-family: var(--mono);
}

.samples {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 18px;
}
.sample-chip {
  background: transparent;
  border: 1px solid var(--panel-border);
  color: var(--text-dim);
  font-size: 11px;
  font-family: var(--mono);
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s;
}
.sample-chip:hover { border-color: var(--safe); color: var(--text); }
.sample-chip-clear { margin-left: auto; opacity: 0.7; }

.field {
  display: block;
  margin-bottom: 14px;
}
.field span {
  display: block;
  font-size: 11px;
  color: var(--text-dim);
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 6px;
}
.field input, .field textarea {
  width: 100%;
  background: #0D1117;
  border: 1px solid var(--panel-border);
  color: var(--text);
  border-radius: 6px;
  padding: 9px 10px;
  font-family: var(--sans);
  font-size: 13px;
  resize: vertical;
}
.field input:focus, .field textarea:focus {
  outline: none;
  border-color: var(--safe);
}

.submit-btn {
  width: 100%;
  background: var(--safe);
  color: #06201C;
  border: none;
  border-radius: 6px;
  padding: 12px;
  font-weight: 700;
  font-size: 13px;
  font-family: var(--mono);
  letter-spacing: 0.03em;
  cursor: pointer;
  margin-top: 4px;
  transition: opacity 0.15s;
}
.submit-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.submit-btn:not(:disabled):hover { opacity: 0.88; }

.error-box {
  margin-top: 12px;
  border: 1px solid var(--danger);
  background: rgba(244,63,94,0.08);
  color: var(--danger);
  font-size: 12px;
  font-family: var(--mono);
  padding: 10px;
  border-radius: 6px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 60px 20px;
  color: var(--text-dim);
}
.empty-glyph { font-size: 28px; margin-bottom: 12px; color: var(--panel-border); }
.empty-glyph.spin { animation: spin 1.1s linear infinite; color: var(--safe); }
@keyframes spin { to { transform: rotate(360deg); } }
.empty-state p { font-size: 12.5px; max-width: 280px; font-family: var(--mono); }

.stages { display: flex; flex-direction: column; }
.stage { display: flex; gap: 14px; }
.stage-rail { display: flex; flex-direction: column; align-items: center; width: 24px; }
.stage-dot {
  width: 24px; height: 24px;
  border-radius: 50%;
  border: 1.5px solid var(--panel-border);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  flex-shrink: 0;
}
.stage-done .stage-dot { border-color: var(--safe); color: var(--safe); }
.stage-skipped .stage-dot { border-style: dashed; opacity: 0.5; }
.stage-line { flex: 1; width: 1px; background: var(--panel-border); margin: 2px 0; }
.stage:last-child .stage-line { display: none; }

.stage-body { padding-bottom: 22px; flex: 1; min-width: 0; }
.stage-title {
  font-family: var(--mono);
  font-size: 12.5px;
  font-weight: 600;
  margin-bottom: 10px;
  display: flex; align-items: center; gap: 8px;
}
.stage-skipped .stage-title { color: var(--text-dim); }
.stage-skip-tag {
  font-size: 9.5px;
  border: 1px solid var(--panel-border);
  color: var(--text-dim);
  padding: 2px 6px;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.stage-note { font-size: 12.5px; color: var(--text-dim); margin: 0; line-height: 1.5; }

.meter { }
.meter-track {
  position: relative;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  background: #0D1117;
  margin-bottom: 8px;
}
.meter-zone { position: absolute; top: 0; bottom: 0; opacity: 0.35; }
.meter-zone-safe { background: var(--safe); }
.meter-zone-warn { background: var(--warn); }
.meter-zone-danger { background: var(--danger); }
.meter-fill {
  position: absolute; top: -2px; bottom: -2px;
  width: 2px; background: var(--text);
  box-shadow: 0 0 6px rgba(228,231,235,0.8);
}
.meter-readout { display: flex; align-items: baseline; gap: 10px; }
.meter-value { font-family: var(--mono); font-weight: 700; font-size: 18px; }
.meter-value.zone-safe { color: var(--safe); }
.meter-value.zone-warn { color: var(--warn); }
.meter-value.zone-danger { color: var(--danger); }
.meter-caption { font-size: 11px; color: var(--text-dim); font-family: var(--mono); }

.context-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
}
.context-card {
  background: #0D1117;
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  padding: 10px 12px;
}
.context-card-title {
  font-family: var(--mono);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  margin-bottom: 8px;
}
.kv { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
.kv-key { color: var(--text-dim); }
.kv-val { font-family: var(--mono); }

.reasoning-block { }
.reasoning-text { font-size: 13px; line-height: 1.55; margin: 0 0 12px; }
.evidence-list { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.evidence-tag {
  font-size: 11px;
  background: rgba(245,158,11,0.1);
  border: 1px solid rgba(245,158,11,0.3);
  color: var(--warn);
  padding: 4px 8px;
  border-radius: 5px;
  font-family: var(--mono);
}
.mitre-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.mitre-tag {
  font-size: 11px;
  font-family: var(--mono);
  background: rgba(94,234,212,0.08);
  border: 1px solid rgba(94,234,212,0.25);
  color: var(--safe);
  padding: 4px 8px;
  border-radius: 5px;
}
.confidence-tag {
  font-size: 11px;
  font-family: var(--mono);
  color: var(--text-dim);
  margin-left: auto;
}

.verdict-wrap {
  margin-top: 8px;
  padding-top: 24px;
  border-top: 1px dashed var(--panel-border);
  text-align: center;
}
.verdict-stamp {
  display: inline-block;
  font-family: var(--mono);
  font-weight: 700;
  font-size: 26px;
  letter-spacing: 0.08em;
  color: var(--stamp-color);
  border: 3px solid var(--stamp-color);
  padding: 10px 28px;
  border-radius: 8px;
  transform: rotate(var(--stamp-angle));
  text-shadow: 0 0 12px color-mix(in srgb, var(--stamp-color) 50%, transparent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--stamp-color) 12%, transparent);
}
.verdict-justification {
  margin: 18px auto 0;
  max-width: 480px;
  font-size: 12.5px;
  color: var(--text-dim);
  line-height: 1.6;
}
`;
