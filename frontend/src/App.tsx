// App.tsx  — Arms: tri-state radios + custom edit, merge respects custom
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

type Flags = { grobid?: boolean; llm?: boolean };
type Draft = { study: Study; arms: Arm[]; outcomes: Outcome[]; _flags?: Flags };

type ReviewField = {
  accepted?: boolean;      // true => include in final
  value?: any;             // if accepted & set, becomes the field value
  evidence?: string;
  // UI-only helper (not required by backend)
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

const API = (import.meta as any).env?.VITE_API || "http://127.0.0.1:8001";

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
      (r.study as any)[k] = {
        accepted: v != null && v !== "",
        value: v,
        mode: v != null && v !== "" ? "draft" : "exclude",
      };
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

// ----------------- Study FieldEditor -----------------
function stringifyValueForEditor(v: any, key: keyof Study): string {
  if (key === "authors") {
    const arr = Array.isArray(v) ? v : (typeof v === "string" && v.trim() ? v.split(/\s*;\s*|\s*,\s*/g) : []);
    return arr.join("; ");
  }
  if (typeof v === "number") return String(v);
  if (v == null) return "";
  return String(v);
}

function parseEditorToValue(text: string, key: keyof Study, draftVal: any): any {
  if (!text.trim()) return null;

  if (key === "authors") {
    return text.split(/;|,/g).map((s) => s.trim()).filter(Boolean);
  }
  if (typeof draftVal === "number") {
    const n = Number(text);
    return Number.isFinite(n) ? n : null;
  }
  if (key === "year") {
    const n = Number(text);
    return Number.isFinite(n) ? n : null;
  }
  return text;
}

function FieldEditor({
  fieldKey,
  draftVal,
  rf,
  radioName,
  onChange,
}: {
  fieldKey: keyof Study;
  draftVal: any;
  rf: ReviewField;
  radioName: string;
  onChange: (upd: Partial<ReviewField>) => void;
}) {
  // derive mode if not set
  let mode: ReviewField["mode"] =
    rf.mode ||
    (rf.accepted
      ? (JSON.stringify(rf.value) === JSON.stringify(draftVal) ? "draft" : "custom")
      : "exclude");

  const [customText, setCustomText] = useState(
    mode === "custom" ? stringifyValueForEditor(rf.value, fieldKey) : ""
  );

  useEffect(() => {
    if (mode === "custom") {
      setCustomText(stringifyValueForEditor(rf.value, fieldKey));
    } else {
      setCustomText("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  function setMode(next: "draft" | "custom" | "exclude") {
    if (next === "draft") {
      onChange({ mode: "draft", accepted: true, value: draftVal });
    } else if (next === "custom") {
      const parsed = parseEditorToValue(customText, fieldKey, draftVal);
      onChange({ mode: "custom", accepted: true, value: parsed });
    } else {
      onChange({ mode: "exclude", accepted: false, value: undefined });
    }
  }

  function saveCustom(text: string) {
    setCustomText(text);
    const parsed = parseEditorToValue(text, fieldKey, draftVal);
    onChange({ mode: "custom", accepted: true, value: parsed });
  }

  const draftStr = stringifyValueForEditor(draftVal, fieldKey);
  const isAuthors = fieldKey === "authors";
  const isNumber = typeof draftVal === "number" || fieldKey === "year";

  return (
    <div className="field-editor">
      <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
        <label className="radio">
          <input
            type="radio"
            name={radioName}
            checked={mode === "draft"}
            onChange={() => setMode("draft")}
          />
          Use Draft
        </label>
        <label className="radio">
          <input
            type="radio"
            name={radioName}
            checked={mode === "custom"}
            onChange={() => setMode("custom")}
          />
          Use Custom
        </label>
        <label className="radio">
          <input
            type="radio"
            name={radioName}
            checked={mode === "exclude"}
            onChange={() => setMode("exclude")}
          />
          Exclude
        </label>
        <input
          className="text"
          placeholder="Evidence pointer (page/figure/etc.)"
          value={rf.evidence || ""}
          onChange={(e) => onChange({ evidence: e.target.value })}
          style={{ minWidth: 240 }}
        />
      </div>

      <div style={{ marginTop: 6 }}>
        <div style={{ fontSize: ".85rem", color: "var(--muted)" }}>Draft value</div>
        <div className="code" style={{ padding: "6px 8px", fontSize: ".9rem" }}>
          {draftStr || <span className="label">(empty)</span>}
        </div>
      </div>

      {mode === "custom" && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: ".85rem", color: "var(--muted)" }}>Custom value</div>
          {isAuthors ? (
            <textarea
              className="text"
              rows={3}
              placeholder="Author A; Author B; Author C"
              value={customText}
              onChange={(e) => saveCustom(e.target.value)}
            />
          ) : (
            <input
              className="text"
              placeholder={isNumber ? "e.g., 2006" : "Type your corrected value"}
              value={customText}
              onChange={(e) => saveCustom(e.target.value)}
            />
          )}
        </div>
      )}
    </div>
  );
}

// ----------------- Arm Editor -----------------
function ArmEditor({
  draftArm,
  rf,
  radioName,
  onChange,
}: {
  draftArm: Arm;
  rf: ReviewField;
  radioName: string;
  onChange: (upd: Partial<ReviewField>) => void;
}) {
  // derive mode if not set
  const equalToDraft = rf.value && typeof rf.value === "object"
    ? rf.value.arm_id === draftArm.arm_id &&
      rf.value.label === draftArm.label &&
      (rf.value.n_randomized ?? null) === (draftArm.n_randomized ?? null)
    : false;

  let mode: ReviewField["mode"] =
    rf.mode || (rf.accepted ? (equalToDraft ? "draft" : "custom") : "exclude");

  // local edit state for custom
  const [labelText, setLabelText] = useState(
    mode === "custom" ? (rf.value?.label ?? draftArm.label) : draftArm.label
  );
  const [nRandText, setNRandText] = useState(
    mode === "custom"
      ? (rf.value?.n_randomized ?? draftArm.n_randomized ?? "")
      : (draftArm.n_randomized ?? "")
  );

  useEffect(() => {
    if (mode === "custom") {
      setLabelText(rf.value?.label ?? draftArm.label);
      setNRandText(rf.value?.n_randomized ?? draftArm.n_randomized ?? "");
    } else {
      setLabelText(draftArm.label);
      setNRandText(draftArm.n_randomized ?? "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, draftArm.arm_id]);

  function toArmValue(lbl: string, nText: any): Arm {
    const n = nText === "" ? null : Number(nText);
    return {
      arm_id: draftArm.arm_id, // arm_id stays stable
      label: lbl,
      n_randomized: Number.isFinite(n) ? n : null,
    };
  }

  function setMode(next: "draft" | "custom" | "exclude") {
    if (next === "draft") {
      onChange({ mode: "draft", accepted: true, value: { ...draftArm } });
    } else if (next === "custom") {
      onChange({ mode: "custom", accepted: true, value: toArmValue(labelText, nRandText) });
    } else {
      onChange({ mode: "exclude", accepted: false, value: undefined });
    }
  }

  function saveCustom(lbl: string, nText: any) {
    setLabelText(lbl);
    setNRandText(nText);
    onChange({ mode: "custom", accepted: true, value: toArmValue(lbl, nText) });
  }

  return (
    <div className="field-editor">
      <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
        <label className="radio">
          <input
            type="radio"
            name={radioName}
            checked={mode === "draft"}
            onChange={() => setMode("draft")}
          />
          Use Draft
        </label>
        <label className="radio">
          <input
            type="radio"
            name={radioName}
            checked={mode === "custom"}
            onChange={() => setMode("custom")}
          />
          Use Custom
        </label>
        <label className="radio">
          <input
            type="radio"
            name={radioName}
            checked={mode === "exclude"}
            onChange={() => setMode("exclude")}
          />
          Exclude
        </label>
        <input
          className="text"
          placeholder="Evidence pointer"
          value={rf.evidence || ""}
          onChange={(e) => onChange({ evidence: e.target.value })}
          style={{ minWidth: 240 }}
        />
      </div>

      <div style={{ marginTop: 6 }}>
        <div style={{ fontSize: ".85rem", color: "var(--muted)" }}>Draft arm</div>
        <div className="code" style={{ padding: "6px 8px", fontSize: ".9rem" }}>
          {draftArm.arm_id} — {draftArm.label}
          {draftArm.n_randomized != null ? ` (n=${draftArm.n_randomized})` : ""}
        </div>
      </div>

      {mode === "custom" && (
        <div style={{ marginTop: 8, display: "grid", gap: 8, gridTemplateColumns: "minmax(220px, 360px) 160px" }}>
          <div>
            <div style={{ fontSize: ".85rem", color: "var(--muted)" }}>Custom label</div>
            <input
              className="text"
              placeholder="e.g., budesonide 9 mg"
              value={labelText}
              onChange={(e) => saveCustom(e.target.value, nRandText)}
            />
          </div>
          <div>
            <div style={{ fontSize: ".85rem", color: "var(--muted)" }}>N randomized</div>
            <input
              className="text"
              type="number"
              min={0}
              placeholder="e.g., 42"
              value={String(nRandText)}
              onChange={(e) => saveCustom(labelText, e.target.value)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------- Reviewer Panel -------------
function ReviewerPanel({
  name,
  docId,
  draft,
  initial,
  onSave,
}: {
  name: "A" | "B";
  docId?: string;
  draft?: Draft;
  initial?: ReviewerReview | null;
  onSave: (r: ReviewerReview) => Promise<void>;
}) {
  const [review, setReview] = useState<ReviewerReview>(initial || { study: {}, arms: {} });
  const [toast, setToast] = useState("");

  // If a saved review arrives, load it
  useEffect(() => {
    if (initial) setReview(initial);
  }, [initial]);

  // Seed study defaults
  useEffect(() => {
    if (!draft) return;
    const hasAny = review.study && Object.keys(review.study).length > 0;
    if (hasAny) return;
    const seeded: Partial<Record<keyof Study, ReviewField>> = {};
    ([
      "title","authors","doi","year","design","condition","country","nct_id","pmid","notes"
    ] as (keyof Study)[]).forEach((k) => {
      const v = (draft.study as any)[k];
      if (v != null && !(Array.isArray(v) && v.length === 0) && v !== "") {
        seeded[k] = { accepted: true, value: v, mode: "draft" };
      } else {
        seeded[k] = { accepted: false, value: undefined, mode: "exclude" };
      }
    });
    setReview((prev) => ({ ...prev, study: seeded, arms: prev.arms || {} }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft?.study]);

  // Seed arm defaults (Use Draft by default)
  useEffect(() => {
    if (!draft?.arms?.length) return;
    setReview((prev) => {
      const next = { ...(prev || {}), arms: { ...(prev.arms || {}) } };
      let changed = false;
      draft.arms.forEach((a) => {
        if (!next.arms![a.arm_id]) {
          next.arms![a.arm_id] = { accepted: true, value: a, mode: "draft" };
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [draft?.arms]);

  function setStudyField(k: keyof Study, upd: Partial<ReviewField>) {
    setReview((prev) => ({
      ...prev,
      study: {
        ...(prev.study || {}),
        [k]: { ...(prev.study?.[k] || {}), ...upd },
      },
    }));
  }

  function setArm(arm_id: string, upd: Partial<ReviewField>) {
    setReview((prev) => ({
      ...prev,
      arms: {
        ...(prev.arms || {}),
        [arm_id]: { ...(prev.arms?.[arm_id] || {}), ...upd },
      },
    }));
  }

  const acceptAll = () => {
    setReview(buildAcceptAll(draft));
    setToast(`Reviewer ${name}: accepted all from draft`);
  };

  const save = async () => {
    await onSave(review);
    setToast(`Reviewer ${name}: saved`);
    await sleep(1500);
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

  // distinct radio group names per field + reviewer (+ doc) to prevent cross-field interference
  const radioPrefix = `mode-${name}-${docId || "doc"}`;

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
              const draftVal = (draft.study as any)[k];
              const rf: ReviewField =
                review.study?.[k] ??
                (draftVal != null && draftVal !== "" && (!Array.isArray(draftVal) || draftVal.length > 0)
                  ? { accepted: true, value: draftVal, mode: "draft" }
                  : { accepted: false, value: undefined, mode: "exclude" });

              return (
                <React.Fragment key={String(k)}>
                  <div className="label">{k}</div>
                  <div>
                    <FieldEditor
                      fieldKey={k}
                      draftVal={draftVal}
                      rf={rf}
                      radioName={`${radioPrefix}-${String(k)}`}
                      onChange={(upd) => setStudyField(k, upd)}
                    />
                  </div>
                </React.Fragment>
              );
            })}
          </div>

          <div className="divider" />
          <h3>Arms</h3>
          {draft.arms?.length ? (
            <div className="kv">
              {draft.arms.map((a) => {
                const rf = review.arms?.[a.arm_id] || { accepted: true, value: a, mode: "draft" };
                return (
                  <React.Fragment key={a.arm_id}>
                    <div className="label">{a.arm_id}</div>
                    <div>
                      <div style={{ marginBottom: 4, fontWeight: 600 }}>{a.label}</div>
                      <ArmEditor
                        draftArm={a}
                        rf={rf}
                        radioName={`${radioPrefix}-arm-${a.arm_id}`}
                        onChange={(upd) => setArm(a.arm_id, upd)}
                      />
                    </div>
                  </React.Fragment>
                );
              })}
            </div>
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

  // Old heuristic from notes:
  const grobidFromNotes = useMemo(() => {
    const note = serverDoc?.draft?.study?.notes || "";
    return /GROBID=on/i.test(note);
  }, [serverDoc]);

  // Preferred explicit flags if present:
  const flags = serverDoc?.draft?._flags || {};
  const grobidOn = typeof flags.grobid === "boolean" ? flags.grobid : grobidFromNotes;
  const llmOn = !!flags.llm;

  async function upload() {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
    const j = await r.json();
    setDocId(j.doc_id);
    setServerDoc(j);
    setToast("Uploaded PDF");
    await sleep(1200);
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
    const r = await fetch(`${API}/api/extract/${docId}`, { method: "POST" });
    const j = await r.json();
    setServerDoc(j);
    setToast("Draft generated");
    await sleep(1200);
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
    // Fallback: save into our local serverDoc state so you can still merge/copy
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

    const a = serverDoc?.reviewA;
    const b = serverDoc?.reviewB;

    // start from draft then overlay A then B for accepted fields
    const mergedStudy: Study = { ...(draft.study || {}) };

    const putStudy = (rev?: ReviewerReview) => {
      if (!rev?.study) return;
      for (const k of Object.keys(rev.study) as (keyof Study)[]) {
        const rf = rev.study[k];
        if (rf?.accepted) {
          (mergedStudy as any)[k] = rf.value ?? (draft.study as any)[k];
        }
      }
    };

    putStudy(a);
    putStudy(b); // B wins ties by default; change order if you prefer A to win

    // Arms: respect custom values if provided, include only accepted
    const byId = new Map<string, Arm>();
    const takeArms = (rev?: ReviewerReview) => {
      if (!rev?.arms) return;
      for (const [id, rf] of Object.entries(rev.arms)) {
        if (!rf?.accepted) continue;
        const fromDraft = draft.arms.find((x) => x.arm_id === id);
        const val: Arm | undefined = rf.value
          ? { arm_id: id, label: rf.value.label ?? fromDraft?.label ?? id, n_randomized: rf.value.n_randomized ?? fromDraft?.n_randomized ?? null }
          : (fromDraft ? { ...fromDraft } : undefined);
        if (val) byId.set(id, val); // later reviewer wins
      }
    };
    takeArms(a);
    takeArms(b);

    const mergedArms = Array.from(byId.values());

    return { study: mergedStudy, arms: mergedArms, outcomes: draft.outcomes || [] };
  }

  const conflicts = computeConflicts();
  const finalCandidate = mergeFinal();

  return (
    <div className="container">
      <h1>Trial Abstraction Prototype</h1>

      {/* --- Status / indicator bar --- */}
      <div className="row" style={{ gap: 8, marginBottom: 12, alignItems: "center", flexWrap: "wrap" }}>
        <span className="badge">GROBID: {grobidOn ? "ON" : "OFF"}</span>
        <span className="badge">LLM: {llmOn ? "ON" : "OFF"}</span>
        {serverDoc?.doc_id && <span className="badge">Doc: {serverDoc.doc_id}</span>}
        {serverDoc?.filename && <span className="badge">File: {serverDoc.filename}</span>}
      </div>

      <Section title="1) Upload PDF">
        <div className="row">
          <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button className="btn" onClick={upload}>Upload</button>
        </div>
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
        </div>

        {serverDoc?.draft ? (
          <>
            <div style={{ marginTop: 8 }}>
              <span className="badge">{grobidOn ? "GROBID: ON" : "GROBID: OFF (fallback parser)"}</span>
              <span className="badge" style={{ marginLeft: 6 }}>{llmOn ? "LLM: ON" : "LLM: OFF"}</span>
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
        docId={docId}
        draft={serverDoc?.draft || undefined}
        initial={serverDoc?.reviewA || undefined}
        onSave={(r) => saveReview("A", r)}
      />

      <ReviewerPanel
        name="B"
        docId={docId}
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
