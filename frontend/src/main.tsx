import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Activity, Ban, Eye, Image, Lock, Play, Plus, RefreshCw, TestTube2, Unlock } from 'lucide-react'
import './styles.css'

type Aggressiveness = 'conservative' | 'balanced' | 'aggressive'

type LockedElement = { type: string; value?: string; bbox?: number[] }
type Constraints = {
  brand: { colors: string[]; fonts: string[]; tone: string; logo_present: boolean }
  locked_elements: LockedElement[]
  aggressiveness: Aggressiveness
}
type StreamEvent = { seq: number; event: string; ts: number; [key: string]: any }
type Variant = {
  id: string
  target_blocker: string
  description: string
  rewritten_copy: string
  cta_instruction: string
  visual_instruction: string
  before_score: number
  after_score: number
  delta: number
  accepted: boolean
  image_url?: string
  explanation?: string
}
type BuyerReaction = { dimension: string; severity: string; blocker: string; explanation: string }
type Result = {
  job_id: string
  image_url: string
  heatmap_url: string
  best_image_url?: string
  baseline: { fixate_score: number; blockers: string[]; regions: { zone: string }[] }
  final: { fixate_score: number; blockers: string[]; regions: { zone: string }[] }
  buyer_reactions: BuyerReaction[]
  diagnosis: { working: string[]; ignored: string[]; hurting_conversion: string[]; summary: string }
  best_variant: Variant | null
  variants: Variant[]
  blocked_edits: { blocker: string; reason: string; estimated_gain: number; variant: any }[]
  experiment_plan: {
    hypothesis: string
    recommended_channel: string
    target_audience: string
    success_metric: string
    ab_test_setup: string
    next_step: string
  }
}

const labels: Record<string, string> = {
  capture_started: 'Capture started',
  capture_done: 'Capture done',
  heatmap_ready: 'Heatmap ready',
  scored: 'Baseline scored',
  buyer_panel: 'Buyer panel',
  diagnosis_ready: 'Diagnosis',
  blocker_found: 'Blocker found',
  variant_proposed: 'Variant proposed',
  variant_applied: 'Variant applied',
  variant_scored: 'Variant scored',
  edit_blocked: 'Edit blocked',
  iteration_done: 'Iteration done',
  job_complete: 'Job complete',
  job_error: 'Job error',
}

const defaultConstraints: Constraints = {
  brand: { colors: ['#0D7D59'], fonts: ['Inter'], tone: 'clear, confident, never hypey', logo_present: false },
  locked_elements: [],
  aggressiveness: 'balanced',
}

function App() {
  const [mode, setMode] = useState<'url' | 'html'>('url')
  const [url, setUrl] = useState('https://example.com')
  const [html, setHtml] = useState('<main><h1>Grow faster with Fixate</h1><p>See what buyers notice before you launch.</p><button>Start now</button></main>')
  const [targetCustomer, setTargetCustomer] = useState('startup founder')
  const [goal, setGoal] = useState('increase signups')
  const [iterations, setIterations] = useState(2)
  const [constraints, setConstraints] = useState<Constraints>(defaultConstraints)
  const [newColor, setNewColor] = useState('#E91E63')
  const [fontInput, setFontInput] = useState('Inter, Poppins')
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [result, setResult] = useState<Result | null>(null)
  const [jobId, setJobId] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [imageUrl, setImageUrl] = useState('')
  const [heatmapUrl, setHeatmapUrl] = useState('')
  const [preview, setPreview] = useState<'heatmap' | 'screenshot' | 'best'>('heatmap')
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => () => esRef.current?.close(), [])

  const scorePoints = useMemo(() => {
    const points: { label: string; score: number }[] = []
    events.forEach(event => {
      if (event.event === 'scored') points.push({ label: 'Base', score: Number(event.fixate_score) })
      if (event.event === 'variant_scored') points.push({ label: `V${event.iteration}`, score: Number(event.fixate_score) })
      if (event.event === 'job_complete') points.push({ label: 'Final', score: Number(event.final_score) })
    })
    return points
  }, [events])

  const zoneCounts = useMemo(() => {
    const regions = result?.final?.regions || result?.baseline?.regions || []
    return regions.reduce<Record<string, number>>((acc, region) => {
      acc[region.zone] = (acc[region.zone] || 0) + 1
      return acc
    }, {})
  }, [result])

  function sourceBody() {
    return mode === 'url' ? { url } : { html }
  }

  function updateLock(type: string, checked: boolean, value?: string) {
    setConstraints(prev => {
      const rest = prev.locked_elements.filter(item => item.type !== type)
      return {
        ...prev,
        locked_elements: checked ? [...rest, { type, value }] : rest,
      }
    })
  }

  function addColor() {
    if (!/^#[0-9a-fA-F]{6}$/.test(newColor)) return
    setConstraints(prev => ({
      ...prev,
      brand: { ...prev.brand, colors: Array.from(new Set([...prev.brand.colors, newColor.toUpperCase()])) },
    }))
  }

  function syncFonts(value: string) {
    setFontInput(value)
    setConstraints(prev => ({
      ...prev,
      brand: { ...prev.brand, fonts: value.split(',').map(font => font.trim()).filter(Boolean) },
    }))
  }

  async function startJob(nextConstraints = constraints) {
    esRef.current?.close()
    setBusy(true)
    setError('')
    setEvents([])
    setResult(null)
    setJobId('')
    setImageUrl('')
    setHeatmapUrl('')
    try {
      const response = await fetch('/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...sourceBody(),
          target_customer: targetCustomer,
          goal,
          iterations,
          constraints: nextConstraints,
        }),
      })
      if (!response.ok) throw new Error(await response.text())
      const data = await response.json()
      setJobId(data.job_id)
      const es = new EventSource(`/job/${data.job_id}/stream`)
      esRef.current = es
      Object.keys(labels).forEach(name => {
        es.addEventListener(name, message => {
          const event = JSON.parse((message as MessageEvent).data) as StreamEvent
          setEvents(prev => [...prev, event])
          if (event.event === 'capture_done' && event.image_url) setImageUrl(event.image_url)
          if (event.event === 'heatmap_ready' && event.heatmap_url) {
            setHeatmapUrl(event.heatmap_url)
            setPreview('heatmap')
          }
          if (event.event === 'variant_applied' && event.image_url) setPreview('best')
          if (event.event === 'job_complete') {
            es.close()
            fetch(`/job/${data.job_id}/result`)
              .then(r => r.json())
              .then(payload => {
                setResult(payload)
                if (payload.best_image_url) setPreview('best')
              })
              .finally(() => setBusy(false))
          }
          if (event.event === 'job_error') {
            es.close()
            setError(event.message || 'Job failed.')
            setBusy(false)
          }
        })
      })
      es.onerror = () => {
        setError('SSE stream disconnected.')
        setBusy(false)
        es.close()
      }
    } catch (err) {
      setBusy(false)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  function unlockAndRerun(blockedType?: string) {
    const type = blockedType || 'layout'
    const next = {
      ...constraints,
      locked_elements: constraints.locked_elements.filter(item => item.type !== type),
    }
    setConstraints(next)
    startJob(next)
  }

  const previewSrc =
    preview === 'best' && result?.best_image_url
      ? result.best_image_url
      : preview === 'screenshot'
        ? imageUrl || result?.image_url
        : heatmapUrl || result?.heatmap_url

  return (
    <main className="shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Fixate</h1>
            <p>AI growth simulator for pre-launch marketing experiments.</p>
          </div>
          <div className="scoreBadge">
            <Activity size={18} />
            <span>{result?.final.fixate_score ?? scorePoints.at(-1)?.score ?? '--'}</span>
          </div>
        </header>

        <div className="appGrid">
          <section className="panel controls">
            <div className="segmented">
              <button className={mode === 'url' ? 'active' : ''} onClick={() => setMode('url')}>URL</button>
              <button className={mode === 'html' ? 'active' : ''} onClick={() => setMode('html')}>HTML</button>
            </div>

            {mode === 'url' ? (
              <label>URL<input value={url} onChange={event => setUrl(event.target.value)} /></label>
            ) : (
              <label>HTML<textarea rows={7} value={html} onChange={event => setHtml(event.target.value)} /></label>
            )}

            <label>Target customer<input value={targetCustomer} onChange={event => setTargetCustomer(event.target.value)} /></label>
            <label>Goal<input value={goal} onChange={event => setGoal(event.target.value)} /></label>
            <label>Iterations<input type="range" min={1} max={10} value={iterations} onChange={event => setIterations(Number(event.target.value))} /><span>{iterations}</span></label>

            <div className="subhead"><Lock size={16} /> Brand & constraints</div>
            <div className="swatches">
              {constraints.brand.colors.map(color => <button key={color} className="swatch" style={{ background: color }} title={color} onClick={() => setNewColor(color)} />)}
              <input value={newColor} onChange={event => setNewColor(event.target.value)} />
              <button onClick={addColor} title="Add color"><Plus size={16} /></button>
            </div>
            <label>Allowed fonts<input value={fontInput} onChange={event => syncFonts(event.target.value)} /></label>
            <label>Tone<input value={constraints.brand.tone} onChange={event => setConstraints(prev => ({ ...prev, brand: { ...prev.brand, tone: event.target.value } }))} /></label>

            <div className="segmented three">
              {(['conservative', 'balanced', 'aggressive'] as Aggressiveness[]).map(item => (
                <button key={item} className={constraints.aggressiveness === item ? 'active' : ''} onClick={() => setConstraints(prev => ({ ...prev, aggressiveness: item }))}>{item}</button>
              ))}
            </div>

            <label className="check"><input type="checkbox" checked={constraints.brand.logo_present} onChange={event => setConstraints(prev => ({ ...prev, brand: { ...prev.brand, logo_present: event.target.checked } }))} /> Logo is present</label>
            <label className="check"><input type="checkbox" checked={constraints.locked_elements.some(item => item.type === 'logo')} onChange={event => updateLock('logo', event.target.checked)} /> Lock logo</label>
            <label className="check"><input type="checkbox" checked={constraints.locked_elements.some(item => item.type === 'legal_text')} onChange={event => updateLock('legal_text', event.target.checked)} /> Lock legal text</label>
            <label className="check"><input type="checkbox" checked={constraints.locked_elements.some(item => item.type === 'layout')} onChange={event => updateLock('layout', event.target.checked, 'fixed')} /> Do not move layout</label>

            <button className="primary" onClick={() => startJob()} disabled={busy}>
              {busy ? <RefreshCw size={17} className="spin" /> : <Play size={17} />}
              Run Fixate
            </button>
            {error && <div className="error">{error}</div>}
          </section>

          <section className="mainStack">
            <section className="panel previewPanel">
              <div className="panelHeader">
                <h2><Eye size={18} /> Attention Preview</h2>
                <div className="segmented compact">
                  <button className={preview === 'heatmap' ? 'active' : ''} onClick={() => setPreview('heatmap')}>Heatmap</button>
                  <button className={preview === 'screenshot' ? 'active' : ''} onClick={() => setPreview('screenshot')}>Original</button>
                  <button className={preview === 'best' ? 'active' : ''} onClick={() => setPreview('best')}>Best</button>
                </div>
              </div>
              {previewSrc ? <img src={previewSrc} alt="Fixate visual preview" /> : <div className="empty">Run a job to generate the capture, heatmap, and edited image.</div>}
            </section>

            <section className="panel scorePanel">
              <h2>Score Trend</h2>
              <div className="scoreChart">
                {scorePoints.map((point, index) => (
                  <div key={`${point.label}-${index}`} className="barWrap">
                    <div className="bar" style={{ height: `${Math.max(8, point.score)}%` }} />
                    <span>{point.label}</span>
                    <strong>{point.score}</strong>
                  </div>
                ))}
                {!scorePoints.length && <div className="empty">Scores stream in during the run.</div>}
              </div>
            </section>

            <section className="lowerGrid">
              <section className="panel">
                <h2>Buyer Reactions</h2>
                <div className="reactionList">
                  {(result?.buyer_reactions || []).map(reaction => (
                    <article key={reaction.dimension} className={`reaction ${reaction.severity}`}>
                      <strong>{reaction.dimension.replace('_', ' ')}</strong>
                      <span>{reaction.severity}</span>
                      <p>{reaction.explanation}</p>
                    </article>
                  ))}
                  {!result?.buyer_reactions?.length && <div className="empty">Buyer-panel flags appear in results.</div>}
                </div>
              </section>
              <section className="panel">
                <h2>Zone Analysis</h2>
                <div className="zoneGrid">
                  {['power_zone', 'attention_trap', 'hidden_value', 'dead_zone'].map(zone => (
                    <div key={zone} className={`zone ${zone}`}>
                      <span>{zone.replace('_', ' ')}</span>
                      <strong>{zoneCounts[zone] || 0}</strong>
                    </div>
                  ))}
                </div>
                {result?.diagnosis && <p className="summary">{result.diagnosis.summary}</p>}
              </section>
            </section>

            <section className="panel">
              <div className="panelHeader">
                <h2>Live Events</h2>
                <span>{jobId || 'No active job'}</span>
              </div>
              <div className="timeline">
                {events.map(event => (
                  <article key={event.seq} className={`event ${event.event === 'edit_blocked' ? 'blocked' : ''}`}>
                    {event.event === 'edit_blocked' ? <Ban size={15} /> : <Activity size={15} />}
                    <div>
                      <strong>{labels[event.event] || event.event}</strong>
                      <pre>{JSON.stringify(event, null, 2)}</pre>
                    </div>
                  </article>
                ))}
                {!events.length && <div className="empty">SSE progress events will stream here.</div>}
              </div>
            </section>
          </section>
        </div>

        {result && (
          <section className="results">
            <section className="resultBand">
              <div><span>Baseline</span><strong>{result.baseline.fixate_score}</strong></div>
              <div><span>Final</span><strong>{result.final.fixate_score}</strong></div>
              <div><span>Delta</span><strong>{(result.final.fixate_score - result.baseline.fixate_score).toFixed(1)}</strong></div>
              <div><span>Winner</span><strong>{result.best_variant?.target_blocker || 'None'}</strong></div>
            </section>

            <section className="panel">
              <h2><Image size={18} /> Ranked Variants</h2>
              <div className="variantGrid">
                {result.variants.map(variant => (
                  <article key={variant.id} className="variantCard">
                    {variant.image_url && <img src={variant.image_url} alt={variant.id} />}
                    <div className="variantBody">
                      <div className="variantTop">
                        <strong>{variant.target_blocker}</strong>
                        <span className={variant.accepted ? 'accepted' : 'rejected'}>{variant.accepted ? 'accepted' : 'rejected'}</span>
                      </div>
                      <p>{variant.explanation || variant.description}</p>
                      <code>{variant.rewritten_copy}</code>
                      <div className="delta">{variant.before_score} → {variant.after_score} ({variant.delta > 0 ? '+' : ''}{variant.delta})</div>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="panel">
              <h2><Ban size={18} /> Blocked By Constraints</h2>
              <div className="blockedGrid">
                {result.blocked_edits.map((blocked, index) => (
                  <article key={`${blocked.blocker}-${index}`} className="blockedCard">
                    <strong>{blocked.blocker}</strong>
                    <p>{blocked.reason}</p>
                    <span>Estimated gain: +{blocked.estimated_gain}</span>
                    <button onClick={() => unlockAndRerun(blocked.reason.includes('Layout') ? 'layout' : undefined)}>
                      <Unlock size={16} />
                      Unlock & re-run
                    </button>
                  </article>
                ))}
                {!result.blocked_edits.length && <div className="empty">No edits were blocked in this run.</div>}
              </div>
            </section>

            <section className="panel abPlan">
              <h2><TestTube2 size={18} /> A/B Test Plan</h2>
              <dl>
                <dt>Hypothesis</dt><dd>{result.experiment_plan.hypothesis}</dd>
                <dt>Channel</dt><dd>{result.experiment_plan.recommended_channel}</dd>
                <dt>Audience</dt><dd>{result.experiment_plan.target_audience}</dd>
                <dt>Success metric</dt><dd>{result.experiment_plan.success_metric}</dd>
                <dt>Setup</dt><dd>{result.experiment_plan.ab_test_setup}</dd>
                <dt>Next step</dt><dd>{result.experiment_plan.next_step}</dd>
              </dl>
            </section>
          </section>
        )}
      </section>
    </main>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
