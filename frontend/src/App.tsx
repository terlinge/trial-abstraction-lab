// App.tsx  — full file
import React, { useEffect, useMemo, useState } from "react";
import "./app.css";

type Study = {
  title?: string | null;
  authors?: string[]; // array of strings
  nct_id?: string | null;
  pmid?: string | null;
  doi?: string | null;
  year?: number | null;
  design?: string | null;
  country?: string | null;
  condition?: string | null;
  notes?: string | null;
};

type Arm = { arm_id: string; label: string; n_randomized?: number | null };
type Outcome = {
  name: string;
  type: string;
  timepoints: { label: string; measures: any[] }[];
  subgroups: any[];
};

type Draft = {
  study: Study;
  arms: Arm[];
  outcomes: Outcome[];
  _flags?: { grobid?: boolean; llm?: boolean; ocr?: boolean; camelot_tables?: boolean };
  _tables?: any[];
};

type ReviewField = {
  accepted?: boolean;
  value?: any;
  evidence?: string;
  // new UI-only field to make intent clear
  mode?: "draft" | "custom" | "exclude";
};

type ReviewerReview = {
  study?: Partial<Record<keyof Study, ReviewField>>;
  arms?: Record<string, ReviewField>; // by arm_id
};

type ServerDoc = {
  doc_id: string;
  filename?: string;
  draft?: Draft;
  reviewA?: ReviewerReview | null;
  reviewB?: ReviewerReview | null;
  final?: Draft | null;
};

type Health = {
  mock_mode: boolean;
  grobid_url: string | null;
  grobid_alive: boolean;
  use_llm: boolean;
  llm_configured: boolean;
  openai_model: string | null;
  api_port: number;
};

// Support both VITE_API (older) and VITE_API_URL (this repo's .env.local) with a safe fallback
const API =
  (import.meta as any).env?.VITE_API || (import.meta as any).env?.VITE_API_URL || "http://127.0.0.1:8001";

// ---------- small helpers ----------
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const copy = async (text: string) => {
  await navigator.clipboard.writeText(text);
};

function Toast({ msg }: { msg: string }) {
  if (!msg) return null;
  return <div className="toast">{msg}</div>;
}

function Section({ title, children }: React.PropsWithChildren<{ title: string }>) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <div style={{ height: 8 }} />
      {children}
    </div>
  );
}

// Build a review object that accepts *all* draft values, used by the “Accept All” button
function buildAcceptAll(draft?: Draft): ReviewerReview {
  const r: ReviewerReview = { study: {}, arms: {} };
  if (draft?.study) {
    (Object.keys(draft.study) as (keyof Study)[]).forEach((k) => {
      const v = (draft.study as any)[k];
      (r.study as any)[k] = { accepted: true, value: v, mode: "draft" };
    });
  }
  if (draft?.arms) {
    draft.arms.forEach((a) => {
      r.arms![a.arm_id] = { accepted: true, value: a, mode: "draft" };
    });
  }
  return r;
}

function diffStudy(a?: ReviewerReview, b?: ReviewerReview) {
  const conflicts: string[] = [];
  const keys = new Set<string>([
    ...(a?.study ? Object.keys(a.study) : []),
    ...(b?.study ? Object.keys(b.study) : []),
  ]);
  keys.forEach((k) => {
    const av = a?.study?.[k as keyof Study]?.value;
    const bv = b?.study?.[k as keyof Study]?.value;
    const aa = a?.study?.[k as keyof Study]?.accepted;
    const bb = b?.study?.[k as keyof Study]?.accepted;
    if ((aa || bb) && JSON.stringify(av) !== JSON.stringify(bv)) {
      conflicts.push(k);
    }
  });
  return conflicts;
}

function StatusPill({ on, label }: { on: boolean; label: string }) {
  return (
    <span className={`badge ${on ? "" : "ghost"}`} style={{ marginRight: 8 }}>
      {label}: {on ? "ON" : "OFF"}
    </span>
  );
}

// ---------- Reviewer Panel -------------
function ReviewerPanel({
  name,
  draft,
  initial,
  onSave,
}: {
  name: "A" | "B";
  draft?: Draft;
  initial?: ReviewerReview | null;
  onSave: (r: ReviewerReview) => Promise<void>;
}) {
  const [review, setReview] = useState<ReviewerReview>(initial || { study: {}, arms: {} });
  const [toast, setToast] = useState("");
  const [forceOcr, setForceOcr] = useState(false);

  useEffect(() => {
    if (initial) setReview(initial);
  }, [initial]);

  // Ensure sensible defaults for each field mode:
  const ensureStudyFieldDefault = (k: keyof Study) => {
    const rf = review.study?.[k];
    const draftVal = (draft?.study as any)?.[k];
    if (!rf) {
      setReview((prev) => ({
        ...prev,
        study: {
          ...(prev.study || {}),
          [k]: {
            accepted: draftVal != null && draftVal !== "" ? true : false,
            mode: draftVal != null && draftVal !== "" ? "draft" : "custom",
            value: draftVal != null ? draftVal : undefined,
            evidence: "",
          },
        },
      }));
    }
  };

  const setStudyField = (k: keyof Study, upd: Partial<ReviewField>) => {
    setReview((prev) => ({
      ...prev,
      study: {
        ...(prev.study || {}),
        [k]: { ...(prev.study?.[k] || {}), ...upd },
      },
    }));
  };

  const setArm = (arm_id: string, upd: Partial<ReviewField>) => {
    setReview((prev) => ({
      ...prev,
      arms: {
        ...(prev.arms || {}),
        [arm_id]: { ...(prev.arms?.[arm_id] || {}), ...upd },
      },
    }));
  };

  const acceptAll = () => {
    setReview(buildAcceptAll(draft));
    setToast(`Reviewer ${name}: accepted all from draft`);
  };

  const save = async () => {
    await onSave(review);
    setToast(`Reviewer ${name}: saved`);
    await sleep(1200);
    setToast("");
  };

  const studyKeys: (keyof Study)[] = [
    "title",
    "authors",
    "doi",
    "year",
    "design",
    "condition",
    "country",
    "nct_id",
    "pmid",
    "notes",
  ];

  const renderCustomStudyInput = (k: keyof Study, rf: ReviewField, draftVal: any) => {
    if (k === "authors") {
      const val =
        Array.isArray(rf.value) ? (rf.value as string[]).join("; ") : (rf.value ?? "");
      return (
        <input
          className="text"
          placeholder="Custom authors (semicolon-separated)"
          value={val}
          onChange={(e) =>
            setStudyField(k, {
              value: e.target.value
                .split(";")
                .map((s) => s.trim())
                .filter(Boolean),
            })
          }
        />
      );
    }
    if (k === "year") {
      return (
        <input
          type="number"
          className="text"
          placeholder="Custom year"
          value={typeof rf.value === "number" ? String(rf.value) : ""}
          onChange={(e) =>
            setStudyField(k, {
              value: e.target.value === "" ? null : Number(e.target.value),
            })
          }
          style={{ width: 160 }}
        />
      );
    }
    return (
      <input
        className="text"
        placeholder={`Custom ${String(k)}`}
        value={rf.value ?? ""}
        onChange={(e) => setStudyField(k, { value: e.target.value })}
      />
    );
  };

  return (
    <Section title={`Reviewer ${name}`}>
      <div className="row">
        <button className="btn" onClick={acceptAll}>Accept All from Draft</button>
        <button className="btn secondary" onClick={() => copy(JSON.stringify(review, null, 2))}>
          Copy Accepted ({name})
        </button>
        <button className="btn ghost" onClick={save}>Save Review</button>
      </div>

      {!draft && <div className="badge">Upload and generate a draft first.</div>}

      {draft && (
        <>
          <div className="divider" />
          <h3>Study</h3>
          <div className="kv">
            {studyKeys.map((k) => {
              ensureStudyFieldDefault(k);
              const draftVal = (draft.study as any)[k];
              const rf = review.study?.[k] || { mode: "draft", accepted: true, value: draftVal };

              const valStr =
                Array.isArray(draftVal) ? draftVal.join("; ") : draftVal == null ? "" : String(draftVal);

              const onModeChange = (mode: "draft" | "custom" | "exclude") => {
                if (mode === "draft") {
                  setStudyField(k, { mode, accepted: true, value: draftVal });
                } else if (mode === "custom") {
                  // seed custom from existing value or blank
                  const seed =
                    rf.value !== undefined ? rf.value :
                    k === "authors" ? [] :
                    k === "year" ? null : "";
                  setStudyField(k, { mode, accepted: true, value: seed });
                } else {
                  setStudyField(k, { mode, accepted: false, value: undefined });
                }
              };

              return (
                <React.Fragment key={String(k)}>
                  <div className="label">{k}</div>
                  <div>
                    <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                      {/* radios */}
                      <label><input
                        type="radio"
                        name={`study-${name}-${String(k)}`}
                        checked={rf.mode === "draft"}
                        onChange={() => onModeChange("draft")}
                      />{" "}Use Draft</label>
                      <label><input
                        type="radio"
                        name={`study-${name}-${String(k)}`}
                        checked={rf.mode === "custom"}
                        onChange={() => onModeChange("custom")}
                      />{" "}Use Custom</label>
                      <label><input
                        type="radio"
                        name={`study-${name}-${String(k)}`}
                        checked={rf.mode === "exclude"}
                        onChange={() => onModeChange("exclude")}
                      />{" "}Exclude</label>

                      {/* evidence */}
                      <input
                        className="text"
                        placeholder="Evidence pointer (page/figure/etc.)"
                        value={rf.evidence || ""}
                        onChange={(e) => setStudyField(k, { evidence: e.target.value })}
                        style={{ minWidth: 220 }}
                      />
                    </div>

                    {/* custom editor */}
                    {rf.mode === "custom" && (
                      <div style={{ marginTop: 6 }}>
                        {renderCustomStudyInput(k, rf, draftVal)}
                      </div>
                    )}

                    {/* draft display */}
                    <div style={{ fontFamily: "ui-monospace, monospace", fontSize: ".9rem", marginTop: 6 }}>
                      <span className="label">Draft value</span>
                      <div>{valStr || <span className="label">(empty)</span>}</div>
                    </div>
                  </div>
                </React.Fragment>
              );
            })}
          </div>

          <div className="divider" />
          <h3>Arms</h3>
          {draft.arms?.length ? (
            draft.arms.map((a) => {
              const rf = review.arms?.[a.arm_id] || { mode: "draft", accepted: true, value: a };
              const onArmMode = (mode: "draft" | "custom" | "exclude") => {
                if (mode === "draft") {
                  setArm(a.arm_id, { mode, accepted: true, value: a });
                } else if (mode === "custom") {
                  const seed = rf.value || { ...a };
                  setArm(a.arm_id, { mode, accepted: true, value: seed });
                } else {
                  setArm(a.arm_id, { mode, accepted: false, value: undefined });
                }
              };

              return (
                <div key={a.arm_id} className="card" style={{ padding: 12, marginBottom: 8 }}>
                  <div className="row" style={{ gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <div className="badge">{a.arm_id}</div>
                    <div>
                      {a.label}
                      {typeof a.n_randomized === "number" && (
                        <span className="label" style={{ marginLeft: 8 }}>n={a.n_randomized}</span>
                      )}
                    </div>

                    <label><input
                      type="radio"
                      name={`arm-${name}-${a.arm_id}`}
                      checked={rf.mode === "draft"}
                      onChange={() => onArmMode("draft")}
                    />{" "}Use Draft</label>

                    <label><input
                      type="radio"
                      name={`arm-${name}-${a.arm_id}`}
                      checked={rf.mode === "custom"}
                      onChange={() => onArmMode("custom")}
                    />{" "}Use Custom</label>

                    <label><input
                      type="radio"
                      name={`arm-${name}-${a.arm_id}`}
                      checked={rf.mode === "exclude"}
                      onChange={() => onArmMode("exclude")}
                    />{" "}Exclude</label>

                    <input
                      className="text"
                      placeholder="Evidence pointer"
                      value={rf.evidence || ""}
                      onChange={(e) => setArm(a.arm_id, { evidence: e.target.value })}
                      style={{ minWidth: 220 }}
                    />
                  </div>

                  {rf.mode === "custom" && (
                    <div className="row" style={{ marginTop: 8, gap: 10, flexWrap: "wrap" }}>
                      <input
                        className="text"
                        placeholder="Custom arm label"
                        value={(rf.value?.label ?? a.label) as string}
                        onChange={(e) => setArm(a.arm_id, {
                          value: { ...(rf.value || a), label: e.target.value }
                        })}
                        style={{ minWidth: 260 }}
                      />
                      <input
                        type="number"
                        min={0}
                        className="text"
                        placeholder="Custom n randomized"
                        value={
                          typeof (rf.value as any)?.n_randomized === "number"
                            ? String((rf.value as any).n_randomized)
                            : ""
                        }
                        onChange={(e) => {
                          const n = e.target.value === "" ? null : Number(e.target.value);
                          setArm(a.arm_id, {
                            value: { ...(rf.value || a), n_randomized: n },
                          });
                        }}
                        style={{ width: 180 }}
                      />
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <div className="label">(no arms)</div>
          )}
        </>
      )}

      <Toast msg={toast} />
    </Section>
  );
}

// --------------------- Main App --------------------------
export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [docId, setDocId] = useState<string>("");
  const [serverDoc, setServerDoc] = useState<ServerDoc | null>(null);
  const [toast, setToast] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [forceOcr, setForceOcr] = useState(false);

  // Status bar — use backend health + draft flags as fallback
  const grobidOn = useMemo(() => {
    const flag = serverDoc?.draft?._flags?.grobid;
    const byNotes = /GROBID=on|grobid tei/i.test(serverDoc?.draft?.study?.notes || "");
    return Boolean(flag ?? byNotes ?? health?.grobid_alive);
  }, [serverDoc, health]);

  const llmOn = useMemo(() => {
    const flag = serverDoc?.draft?._flags?.llm;
    const byNotes = /LLM|GROBID \+ LLM/i.test(serverDoc?.draft?.study?.notes || "");
    const byHealth = Boolean(health?.use_llm && health?.llm_configured);
    return Boolean(flag ?? byNotes ?? byHealth);
  }, [serverDoc, health]);

  const ocrOn = useMemo(() => {
    const flag = serverDoc?.draft?._flags?.ocr;
    return Boolean(flag);
  }, [serverDoc]);

  useEffect(() => {
    // fetch backend health once on load
    (async () => {
      try {
        const r = await fetch(`${API}/api/health`);
        if (r.ok) {
          const j = await r.json();
          setHealth(j);
        }
      } catch {}
    })();
  }, []);

  async function upload() {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
    const j = await r.json();
    setDocId(j.doc_id);
    setServerDoc(j);
    setToast("Uploaded PDF");
    await sleep(1000);
    setToast("");
  }

  async function refresh() {
    if (!docId) return;
    const r = await fetch(`${API}/api/doc/${docId}`);
    const j = await r.json();
    setServerDoc(j);
  }

  async function genDraft() {
    if (!docId) {
      setToast("Upload a PDF first"); await sleep(1200); setToast(""); return;
    }
    const r = await fetch(`${API}/api/extract/${docId}?force_ocr=${forceOcr ? "true" : "false"}`, { method: "POST" });
    const j = await r.json();
    setServerDoc(j);
    setToast("Draft generated");
    await sleep(1000);
    setToast("");
  }

  // Save review to backend if available; otherwise keep client-side only
  async function saveReview(which: "A" | "B", review: ReviewerReview) {
    if (!docId) return;
    try {
      const r = await fetch(`${API}/api/review/${docId}/${which}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(review),
      });
      if (r.ok) {
        await refresh();
        return;
      }
    } catch {
      /* backend route might not exist; fall back below */
    }
    // Fallback: save locally so you can still merge/copy
    setServerDoc((prev) => ({
      ...(prev || ({} as any)),
      doc_id: docId,
      reviewA: which === "A" ? review : prev?.reviewA || null,
      reviewB: which === "B" ? review : prev?.reviewB || null,
    }) as ServerDoc);
  }

  function computeConflicts(): string[] {
    return diffStudy(serverDoc?.reviewA || undefined, serverDoc?.reviewB || undefined);
  }

  function mergeFinal(): Draft | null {
    const draft = serverDoc?.draft;
    if (!draft) return null;

    const a = serverDoc?.reviewA || undefined;
    const b = serverDoc?.reviewB || undefined;

    // start from draft then overlay A then B for accepted fields
    const mergedStudy: Study = { ...(draft.study || {}) };

    const put = (rev?: ReviewerReview) => {
      if (!rev?.study) return;
      for (const k of Object.keys(rev.study) as (keyof Study)[]) {
        const rf = rev.study[k];
        if (rf?.accepted) {
          (mergedStudy as any)[k] = rf.value ?? (draft.study as any)[k];
        }
      }
    };

    put(a);
    put(b); // B wins ties by default

    // Arms — union accepted arms, prefer reviewer custom value if provided
    const byId: Record<string, Arm> = {};
    draft.arms.forEach((arm) => (byId[arm.arm_id] = arm));

    const outArms: Arm[] = [];
    const seen = new Set<string>();

    const appendFrom = (rev?: ReviewerReview) => {
      if (!rev?.arms) return;
      for (const [arm_id, rf] of Object.entries(rev.arms)) {
        if (!rf?.accepted) continue;
        if (seen.has(arm_id)) continue;
        // prefer reviewer custom value if present
        const base = byId[arm_id] || { arm_id, label: arm_id, n_randomized: null };
        const v = (rf.value as Arm) || base;
        outArms.push({
          arm_id,
          label: v.label ?? base.label,
          n_randomized:
            typeof v.n_randomized === "number" ? v.n_randomized : base.n_randomized ?? null,
        });
        seen.add(arm_id);
      }
    };

    appendFrom(a);
    appendFrom(b);

    return { study: mergedStudy, arms: outArms, outcomes: draft.outcomes || [] };
  }

  const conflicts = computeConflicts();
  const finalCandidate = mergeFinal();

  return (
    <div className="container">
      <h1>Trial Abstraction Prototype</h1>

      {/* Status bar */}
      <div className="row" style={{ marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
        <StatusPill on={grobidOn} label="GROBID" />
        <StatusPill on={llmOn} label="LLM" />
        {serverDoc?.doc_id && <span className="badge">Doc: {serverDoc.doc_id}</span>}
        {serverDoc?.filename && (
          <span className="badge ghost">File: {serverDoc.filename}</span>
        )}
      </div>

      <Section title="1) Upload PDF">
        <div className="row">
          <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button className="btn" onClick={upload}>Upload</button>
        </div>
        {serverDoc?.filename && (
          <div style={{ marginTop: 8 }}>
            <span className="badge">Doc ID: {serverDoc.doc_id}</span>&nbsp; | &nbsp;
            <span className="label">File:</span> {serverDoc.filename}
          </div>
        )}
      </Section>

      <Section title="2) Draft Extraction">
        <div className="row">
          <button className="btn" onClick={genDraft}>Generate Draft</button>
          <button className="btn secondary" onClick={refresh}>Refresh</button>
          <button
            className="btn ghost"
            onClick={() => serverDoc?.draft && copy(JSON.stringify(serverDoc.draft, null, 2))}
          >
            Copy Draft JSON
          </button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 8 }}>
            <input type="checkbox" checked={forceOcr} onChange={(e) => setForceOcr(e.target.checked)} /> Force OCR
          </label>
        </div>

        {serverDoc?.draft ? (
          <>
            <div style={{ marginTop: 8 }}>
              <span className="badge">{grobidOn ? "GROBID: ON" : "GROBID: OFF (fallback parser)"}</span>
              <span className="badge" style={{ marginLeft: 8 }}>
                {llmOn ? "LLM: ON" : "LLM: OFF"}
              </span>
              <span className="badge" style={{ marginLeft: 8 }}>
                {Boolean(serverDoc?.draft?._flags && (serverDoc.draft._flags as any).ocr) ? "OCR: ON" : "OCR: OFF"}
              </span>
              {serverDoc?.draft?._tables && (
                <button
                  className="btn ghost"
                  style={{ marginLeft: 8 }}
                  onClick={() => copy(JSON.stringify(serverDoc.draft!._tables, null, 2))}
                >
                  Copy Tables JSON
                </button>
              )}
            </div>
            <div className="code" style={{ marginTop: 10 }}>
              <pre>{JSON.stringify(serverDoc.draft, null, 2)}</pre>
            </div>
          </>
        ) : (
          <div className="label">No draft yet.</div>
        )}
      </Section>

      <ReviewerPanel
        name="A"
        draft={serverDoc?.draft || undefined}
        initial={serverDoc?.reviewA || undefined}
        onSave={(r) => saveReview("A", r)}
      />

      <ReviewerPanel
        name="B"
        draft={serverDoc?.draft || undefined}
        initial={serverDoc?.reviewB || undefined}
        onSave={(r) => saveReview("B", r)}
      />

      <Section title="5) Conflicts & Merge">
        <div className="row">
          <button className="btn secondary" onClick={() => copy(JSON.stringify(conflicts, null, 2))}>
            Copy Conflicts
          </button>
          <button
            className="btn"
            onClick={() => finalCandidate && copy(JSON.stringify(finalCandidate, null, 2))}
          >
            Copy Final JSON
          </button>
        </div>

        <div style={{ marginTop: 8 }}>
          <div><strong>Conflicts (study fields):</strong> {conflicts.length ? conflicts.join(", ") : "none"}</div>
        </div>

        {finalCandidate ? (
          <div className="code" style={{ marginTop: 10 }}>
            <pre>{JSON.stringify(finalCandidate, null, 2)}</pre>
          </div>
        ) : (
          <div className="label">Generate a draft and save Reviewer A/B to build a final.</div>
        )}

        <div style={{ marginTop: 10 }} className="label">API base: {API}</div>
      </Section>

      <Toast msg={toast} />
    </div>
  );
}
