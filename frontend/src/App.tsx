import React, { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { flatten } from './util'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

type Conflict = { key: string; A: any; B: any; type: 'numeric'|'text' }

export default function App() {
  const [docId, setDocId] = useState<string>('')
  const [docState, setDocState] = useState<any>(null)
  const [draftFlat, setDraftFlat] = useState<Record<string,any>>({})
  const [reviewA, setReviewA] = useState<Record<string,any>>({})
  const [reviewB, setReviewB] = useState<Record<string,any>>({})
  const [verifyA, setVerifyA] = useState<Record<string,boolean>>({})
  const [verifyB, setVerifyB] = useState<Record<string,boolean>>({})
  const [evidenceA, setEvidenceA] = useState<Record<string,string>>({})
  const [evidenceB, setEvidenceB] = useState<Record<string,string>>({})
  const [conflicts, setConflicts] = useState<Conflict[]>([])
  const [resolution, setResolution] = useState<Record<string,any>>({})

  async function refresh() {
    if (!docId) return
    const r = await axios.get(`${API}/api/doc/${docId}`)
    setDocState(r.data)
    if (r.data.draft) setDraftFlat(flatten(r.data.draft))
    if (r.data.reviewA) setReviewA(r.data.reviewA.data || {})
    if (r.data.reviewB) setReviewB(r.data.reviewB.data || {})
  }

  useEffect(() => { if (docId) refresh() }, [docId])

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    axios.post(`${API}/api/upload`, form).then(r => {
      setDocId(r.data.doc_id)
    })
  }

  function startExtract() {
    if (!docId) return
    axios.post(`${API}/api/extract/${docId}`).then(() => refresh())
  }

  async function submitReview(who: 'A'|'B', data: Record<string,any>, verified: Record<string,boolean>, evidence: Record<string,string>) {
    if (!docId) return
    const form = new FormData()
    form.append('reviewer', who)
    form.append('data', JSON.stringify(data))
    form.append('verified', JSON.stringify(verified))
    form.append('evidence', JSON.stringify(evidence))
    await axios.post(`${API}/api/review/${docId}`, form)
    await refresh()
  }

  async function loadConflicts() {
    if (!docId) return
    const r = await axios.get(`${API}/api/conflicts/${docId}`)
    setConflicts(r.data.conflicts || [])
  }

  async function finalize() {
    if (!docId) return
    const form = new FormData()
    form.append('adjudicator', 'ReviewerC')
    form.append('resolution', JSON.stringify(resolution))
    form.append('notes', 'Resolved via UI selection')
    await axios.post(`${API}/api/adjudicate/${docId}`, form)
    await refresh()
  }

  async function getProvenance() {
    if (!docId) return
    const r = await axios.get(`${API}/api/provenance/${docId}`)
    alert(JSON.stringify(r.data, null, 2))
  }

  async function exportContrasts() {
    if (!docId) return
    const r = await axios.get(`${API}/api/export/contrasts/${docId}`, { responseType: 'text' })
    const blob = new Blob([r.data], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `contrasts_${docId}.csv`
    a.click()
  }

  const fields = useMemo(() => Object.keys(draftFlat), [draftFlat])

  return (
    <div className="container">
      <h1>Trial Abstraction Prototype</h1>
      <p className="muted">Upload a PDF, generate a mock AI draft, two reviewers validate with per-field checkboxes and evidence pointers, adjudicate, then export contrasts.</p>

      <div className="card">
        <h3>1) Upload PDF</h3>
        <input type="file" accept="application/pdf" onChange={onFileChange} />
        {docId && <div>Doc ID: <code>{docId}</code></div>}
      </div>

      <div className="card">
        <h3>2) Draft Extraction</h3>
        <div className="row">
          <button className="btn" onClick={startExtract} disabled={!docId}>Generate Draft (Mock)</button>
          <button className="btn secondary" onClick={refresh} disabled={!docId}>Refresh</button>
          <button className="btn secondary" onClick={getProvenance} disabled={!docId}>View Provenance</button>
        </div>
        {docState?.draft && (
          <div style={{marginTop:12}}>
            <span className="badge">Draft available</span>
            <pre className="small">{JSON.stringify(docState.draft, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="card">
        <h3>3) Reviewer A</h3>
        <FormEditor
          disabled={!docState?.draft}
          baseline={draftFlat}
          value={reviewA}
          verified={verifyA}
          evidence={evidenceA}
          onChange={setReviewA}
          onToggleVerified={(k, v) => setVerifyA(prev => ({...prev, [k]: v}))}
          onChangeEvidence={(k, v) => setEvidenceA(prev => ({...prev, [k]: v}))}
          onSave={() => submitReview('A', reviewA, verifyA, evidenceA)}
        />
      </div>

      <div className="card">
        <h3>4) Reviewer B</h3>
        <FormEditor
          disabled={!docState?.draft}
          baseline={draftFlat}
          value={reviewB}
          verified={verifyB}
          evidence={evidenceB}
          onChange={setReviewB}
          onToggleVerified={(k, v) => setVerifyB(prev => ({...prev, [k]: v}))}
          onChangeEvidence={(k, v) => setEvidenceB(prev => ({...prev, [k]: v}))}
          onSave={() => submitReview('B', reviewB, verifyB, evidenceB)}
        />
      </div>

      <div className="card">
        <h3>5) Conflicts</h3>
        <div className="row">
          <button className="btn" onClick={loadConflicts} disabled={!docId}>Load Conflicts</button>
        </div>
        {conflicts.length > 0 ? (
          <div style={{marginTop:12}}>
            {conflicts.map(c => (
              <div key={c.key} className="conflict" style={{marginBottom:8}}>
                <div><strong>{c.key}</strong></div>
                <div className="row" style={{marginTop:6}}>
                  <button className="btn secondary" onClick={() => setResolution(prev => ({...prev, [c.key]: c.A}))}>Choose A</button>
                  <button className="btn secondary" onClick={() => setResolution(prev => ({...prev, [c.key]: c.B}))}>Choose B</button>
                </div>
                <div style={{marginTop:8}}>
                  <table className="table">
                    <thead><tr><th>Reviewer</th><th>Value</th></tr></thead>
                    <tbody>
                      <tr><td>A</td><td>{String(c.A)}</td></tr>
                      <tr><td>B</td><td>{String(c.B)}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        ) : <p className="muted">No conflicts loaded yet.</p>}
      </div>

      <div className="card">
        <h3>6) Adjudication & Export</h3>
        <div className="row">
          <button className="btn" onClick={finalize} disabled={!docId || conflicts.length===0}>Finalize</button>
          <button className="btn secondary" onClick={exportContrasts} disabled={!docState?.final}>Export Contrasts (CSV)</button>
        </div>
        {docState?.final && (
          <div style={{marginTop:12}}>
            <span className="badge">Finalized</span>
            <pre className="small">{JSON.stringify(docState.final, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

function FormEditor({
  baseline, value, verified, evidence, onChange, onToggleVerified, onChangeEvidence, onSave, disabled
}: {
  baseline: Record<string,any>,
  value: Record<string,any>,
  verified: Record<string,boolean>,
  evidence: Record<string,string>,
  onChange: (v: Record<string,any>) => void,
  onToggleVerified: (k: string, v: boolean) => void,
  onChangeEvidence: (k: string, v: string) => void,
  onSave: () => void,
  disabled?: boolean
}) {
  const fields = Object.keys(baseline || {})
  if (disabled) return <p className="muted">Upload and generate a draft first.</p>
  return (
    <div>
      <div className="row" style={{marginBottom:12}}>
        <button className="btn" onClick={() => onSave()}>Save Review</button>
      </div>
      <div className="grid">
        {fields.map(k => (
          <div key={k}>
            <label style={{fontSize:12}}>{k}</label>
            <input
              type="text"
              value={value[k] ?? baseline[k] ?? ''}
              onChange={e => onChange({...value, [k]: e.target.value})}
            />
            <div className="row" style={{alignItems:'center', marginTop:6}}>
              <label className="cb">
                <input type="checkbox" checked={!!verified[k]} onChange={e => onToggleVerified(k, e.target.checked)} />
                Verified
              </label>
              <input
                type="text"
                placeholder="Evidence pointer (e.g., p7 Table 2)"
                value={evidence[k] ?? ''}
                onChange={e => onChangeEvidence(k, e.target.value)}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}