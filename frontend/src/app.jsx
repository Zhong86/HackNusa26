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

// --- Layer 1 feature breakdown --------------------------------------
// layer1_score.features is a flat dict of every structural + embedding
// feature extract.py computed for this email. We surface it as readable
// indicators so an analyst can see *why* the score landed where it did,
// not just the number itself — this is the actual "why" behind an
// allow/quarantine decision made without ever touching Layer 2.

const FLAG_FEATURES = [
  { key: "display_name_mismatch", label: "Display name impersonates a known brand" },
  { key: "org_identity_mismatch", label: "Display name claims an org identity the domain doesn't match" },
  { key: "has_digit_in_domain", label: "Digit substitution in sending domain (lookalike)" },
  { key: "has_ip_url", label: "Link points to a raw IP address" },
  { key: "has_shortener_url", label: "Link uses a URL shortener" },
  { key: "suspicious_tld_flag", label: "Link uses a high-risk TLD" },
];

const SCORE_FEATURES = [
  { key: "urgency_score", label: "Urgency language hits" },
  { key: "financial_request_score", label: "Financial-request language hits" },
  { key: "credential_request_score", label: "Credential-request language hits" },
];

function featureSummaryLine(features) {
  if (!features) return null;
  const firedFlags = FLAG_FEATURES.filter((f) => features[f.key] === 1).map((f) => f.label);
  const scoreHits = SCORE_FEATURES.filter((f) => (features[f.key] ?? 0) > 0);
  if (firedFlags.length === 0 && scoreHits.length === 0) {
    return "No structural red flags or suspicious-language hits fired. The score reflects semantic similarity to known phishing/benign examples and general email shape.";
  }
  const parts = [];
  if (firedFlags.length) parts.push(firedFlags.join("; "));
  if (scoreHits.length) {
    parts.push(scoreHits.map((f) => `${f.label} (${features[f.key]})`).join("; "));
  }
  return parts.join(". ") + ".";
}

function FeatureBreakdown({ features }) {
  if (!features) return null;

  const firedFlags = FLAG_FEATURES.filter((f) => features[f.key] === 1);
  const clearFlags = FLAG_FEATURES.filter((f) => features[f.key] === 0);
  const hasEmbedding = "embed_similarity_margin" in features;

  return (
    <div className="feature-breakdown">
      <p className="feature-summary">{featureSummaryLine(features)}</p>

      {firedFlags.length > 0 && (
        <div className="feature-group">
          <div className="feature-group-title">Flags triggered</div>
          <div className="feature-flags">
            {firedFlags.map((f) => (
              <span key={f.key} className="feature-flag feature-flag-hit">
                ⚠ {f.label}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="feature-group">
        <div className="feature-group-title">Language signals</div>
        <div className="feature-score-row">
          {SCORE_FEATURES.map((f) => (
            <div
              key={f.key}
              className={`feature-score-chip ${features[f.key] > 0 ? "feature-score-chip-hit" : ""}`}
            >
              <span className="feature-score-chip-val">{features[f.key] ?? 0}</span>
              <span className="feature-score-chip-label">{f.label}</span>
            </div>
          ))}
        </div>
      </div>

      {hasEmbedding && (
        <div className="feature-group">
          <div className="feature-group-title">Semantic similarity</div>
          <div className="feature-score-row">
            <div className="feature-score-chip">
              <span className="feature-score-chip-val">
                {features.embed_phishing_similarity?.toFixed(3)}
              </span>
              <span className="feature-score-chip-label">to known phishing</span>
            </div>
            <div className="feature-score-chip">
              <span className="feature-score-chip-val">
                {features.embed_benign_similarity?.toFixed(3)}
              </span>
              <span className="feature-score-chip-label">to known benign</span>
            </div>
            <div
              className={`feature-score-chip ${
                features.embed_similarity_margin > 0 ? "feature-score-chip-hit" : ""
              }`}
            >
              <span className="feature-score-chip-val">
                {features.embed_similarity_margin?.toFixed(3)}
              </span>
              <span className="feature-score-chip-label">margin (+ = phishing-leaning)</span>
            </div>
          </div>
        </div>
      )}

      {clearFlags.length > 0 && (
        <details className="feature-clear-details">
          <summary>
            {clearFlags.length} other check{clearFlags.length === 1 ? "" : "s"} passed clean
          </summary>
          <div className="feature-flags">
            {clearFlags.map((f) => (
              <span key={f.key} className="feature-flag feature-flag-clear">
                ✓ {f.label}
              </span>
            ))}
          </div>
        </details>
      )}
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
  // gather_context_node currently short-circuits with `return {}` before it
  // appends itself to trace (see backend/graph/nodes.py), so it never shows
  // up here even when the graph does route through Layer 2. reason_node and
  // auto_decide_node don't have that bug, so use those as the reliable
  // signal instead.
  const wasUncertain = trace.includes("reason_node") || trace.includes("auto_decide_node");
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
                <Stage index={1} title="Classifier score" status="done">
                  {state.layer1_score && (
                    <>
                      <ScoreMeter score={state.layer1_score.score} />
                      <FeatureBreakdown features={state.layer1_score.features} />
                    </>
                  )}
                </Stage>

                <Stage
                  index={2}
                  title={wasUncertain ? "Escalated for deeper reasoning" : "Decided directly"}
                  status="done"
                >
                  <p className="stage-note">
                    {wasUncertain
                      ? "The score fell in an uncertain range, so the system gathered more context and reasoned through it before deciding."
                      : "The score was confident enough to decide on its own — no further reasoning was needed."}
                  </p>

                  {wasUncertain && state.context && (
                    <div className="context-grid">
                      <div className="context-card">
                        <div className="context-card-title">Sender history</div>
                        <KeyVal label="seen before" value={state.context.sender_history?.seen_before} />
                        <KeyVal label="prior flags" value={state.context.sender_history?.prior_flag_count} />
                      </div>
                    </div>
                  )}

                  {wasUncertain && state.reasoning && (
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

.meter { margin-bottom: 14px; }
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

.feature-breakdown {
  margin-top: 4px;
  border-top: 1px dashed var(--panel-border);
  padding-top: 12px;
}
.feature-summary {
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.55;
  margin: 0 0 12px;
}
.feature-group { margin-bottom: 12px; }
.feature-group:last-child { margin-bottom: 0; }
.feature-group-title {
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  margin-bottom: 6px;
}
.feature-flags { display: flex; flex-wrap: wrap; gap: 6px; }
.feature-flag {
  font-size: 11px;
  font-family: var(--mono);
  padding: 4px 8px;
  border-radius: 5px;
  border: 1px solid var(--panel-border);
}
.feature-flag-hit {
  color: var(--danger);
  border-color: rgba(244,63,94,0.35);
  background: rgba(244,63,94,0.08);
}
.feature-flag-clear {
  color: var(--text-dim);
  opacity: 0.85;
}
.feature-score-row { display: flex; flex-wrap: wrap; gap: 8px; }
.feature-score-chip {
  background: #0D1117;
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  padding: 6px 10px;
  min-width: 84px;
}
.feature-score-chip-hit {
  border-color: rgba(245,158,11,0.4);
  background: rgba(245,158,11,0.06);
}
.feature-score-chip-val {
  display: block;
  font-family: var(--mono);
  font-weight: 700;
  font-size: 14px;
}
.feature-score-chip-hit .feature-score-chip-val { color: var(--warn); }
.feature-score-chip-label {
  display: block;
  font-size: 10px;
  color: var(--text-dim);
  margin-top: 2px;
  line-height: 1.3;
}
.feature-clear-details {
  margin-top: 4px;
}
.feature-clear-details summary {
  font-size: 11px;
  font-family: var(--mono);
  color: var(--text-dim);
  cursor: pointer;
  margin-bottom: 8px;
}
.feature-clear-details summary:hover { color: var(--text); }

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
.verdict-source {
  margin: 14px 0 0;
  font-size: 11px;
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
}
.verdict-justification {
  margin: 8px auto 0;
  max-width: 480px;
  font-size: 12.5px;
  color: var(--text-dim);
  line-height: 1.6;
}
`;
