// App.tsx  (full file)
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

type Draft = { study: Study; arms: Arm[]; outcomes: Outcome[] };

type ReviewField = {
  accepted?: boolean;
  value?: any;
  evidence?: string;
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
      (r.study as any)[k] = { accepted: true, value: (draft.study as any)[k] };
    });
  }
  if (draft?.arms) {
    draft.arms.forEach((a) => {
      r.arms![a.arm_id] = { accepted: true, value: a };
    });
  }
  return r;
}

function getAcceptedStudyValues(draft: Draft | undefined, review: ReviewerReview | undefined) {
  const out: Partial<Study> = {};
  if (!draft || !review?.study) return out;
  for (const k of Object.keys(review.study) as (keyof Study)[]) {
    const rf = review.study[k];
    if (rf?.accepted) {
      out[k] = rf.value ?? (draft.study as any)[k];
    }
  }
  return out;
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

  useEffect(() => {
    if (initial) setReview(initial);
  }, [initial]);

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
              const rf = review.study?.[k] || {};
              const valStr =
                Array.isArray(draftVal) ? draftVal.join("; ") : draftVal == null ? "" : String(draftVal);
              return (
                <React.Fragment key={String(k)}>
                  <div className="label">{k}</div>
                  <div>
                    <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={!!rf.accepted}
                        onChange={(e) => setStudyField(k, { accepted: e.target.checked, value: draftVal })}
                        title="Accept this field from draft"
                      />
                      <input
                        className="text"
                        placeholder="Evidence pointer (page/figure/etc.)"
                        value={rf.evidence || ""}
                        onChange={(e) => setStudyField(k, { evidence: e.target.value })}
                      />
                    </div>
                    <div style={{ fontFamily: "ui-monospace, monospace", fontSize: ".9rem", marginTop: 4 }}>
                      {valStr || <span className="label">(empty)</span>}
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
              const rf = review.arms?.[a.arm_id] || {};
              return (
                <div key={a.arm_id} className="row" style={{ alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={!!rf.accepted}
                    onChange={(e) => setArm(a.arm_id, { accepted: e.target.checked, value: a })}
                  />
                  <div className="badge">{a.arm_id}</div>
                  <div>{a.label}</div>
                  <input
                    className="text"
                    placeholder="Evidence pointer"
                    value={rf.evidence || ""}
                    onChange={(e) => setArm(a.arm_id, { evidence: e.target.value })}
                  />
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

  const grobidOn = useMemo(() => {
    const note = serverDoc?.draft?.study?.notes || "";
    return /GROBID=on/i.test(note);
  }, [serverDoc]);

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
    } catch (_) {
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
    put(b); // B wins ties by default; change order if you prefer A to win

    const mergedArms: Arm[] = [];
    const acceptedArmIds = new Set<string>([
      ...(a?.arms ? Object.keys(a.arms).filter((id) => a.arms?.[id]?.accepted) : []),
      ...(b?.arms ? Object.keys(b.arms).filter((id) => b.arms?.[id]?.accepted) : []),
    ]);
    draft.arms.forEach((arm) => {
      if (acceptedArmIds.has(arm.arm_id)) mergedArms.push(arm);
    });

    return { study: mergedStudy, arms: mergedArms, outcomes: draft.outcomes || [] };
  }

  const conflicts = computeConflicts();
  const finalCandidate = mergeFinal();

  return (
    <div className="container">
      <h1>Trial Abstraction Prototype</h1>

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
        </div>

        {serverDoc?.draft ? (
          <>
            <div style={{ marginTop: 8 }}>
              <span className="badge">{grobidOn ? "GROBID: ON" : "GROBID: OFF (fallback parser)"}</span>
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
