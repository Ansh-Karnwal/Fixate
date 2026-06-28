import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity,
  Ban,
  Brain,
  Crosshair,
  Eye,
  FileCode2,
  Image as ImageIcon,
  Lock,
  MousePointerClick,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Sparkles,
  Target,
  TestTube2,
  Unlock,
  Upload,
  Users,
} from 'lucide-react'
import './styles.css'

type SourceMode = 'url' | 'html' | 'image'
type Aggressiveness = 'conservative' | 'balanced' | 'aggressive'

type LockedElement = { type: string; value?: string; bbox?: number[] }
type Constraints = {
  brand: { colors: string[]; fonts: string[]; tone: string; logo_present: boolean }
  locked_elements: LockedElement[]
  aggressiveness: Aggressiveness
}
type StreamEvent = { seq: number; event: string; agent?: string; ts: number; [key: string]: any }
type FixationRegion = { rank: number; bbox: number[]; saliency_score: number; peak_coords: number[]; reason?: string }
type DemographicSegment = {
  id: string
  name: string
  summary: string
  messaging_angle: string
  visual_direction: string
  recommended_channel: string
  why_it_fits: string
}
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
  demographic_focus?: string
  explanation?: string
}
type BuyerReaction = { dimension: string; severity: string; blocker: string; explanation: string }
type ApiStatus = {
  health?: {
    openai_configured: boolean
    openai_required: boolean
    openai_model: string
    provider_label?: string
    runtime_label?: string
    meta_tribe_demo?: boolean
  }
  debug?: { ok: boolean; model: string; error_type?: string; error?: string; response?: string }
}
type Result = {
  job_id: string
  source_type: SourceMode
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
  demographics: DemographicSegment[]
  selected_demographic?: DemographicSegment | null
}

const eventMeta: Record<string, { label: string; agent: string; icon: React.ElementType }> = {
  capture_started: { label: 'Capture started', agent: 'Capture Agent', icon: Upload },
  capture_done: { label: 'Asset captured', agent: 'Capture Agent', icon: Upload },
  demographics_started: { label: 'Audience scan started', agent: 'Demographics Agent', icon: Users },
  demographics_ready: { label: 'Demographics ready', agent: 'Demographics Agent', icon: Users },
  heatmap_ready: { label: 'Heatmap ready', agent: 'Attention Agent', icon: Eye },
  scored: { label: 'Buyer response scored', agent: 'Buyer-Response Scorer', icon: Brain },
  buyer_panel: { label: 'Buyer panel simulated', agent: 'Buyer Panel Agents', icon: Users },
  diagnosis_ready: { label: 'Diagnosis ready', agent: 'Growth Strategist Agent', icon: Target },
  blocker_found: { label: 'Blocker found', agent: 'Growth Strategist Agent', icon: Target },
  variant_proposed: { label: 'Variant proposed', agent: 'Creative Agent', icon: Pencil },
  variant_applied: { label: 'Image variant edited', agent: 'Image Editing Agent', icon: Sparkles },
  variant_image_failed: { label: 'Image edit failed', agent: 'Image Editing Agent', icon: Ban },
  variant_scored: { label: 'Variant re-scored', agent: 'Buyer-Response Scorer', icon: Brain },
  edit_blocked: { label: 'Edit blocked', agent: 'Constraint Guard', icon: Ban },
  iteration_done: { label: 'Loop iteration done', agent: 'Experiment Loop', icon: RefreshCw },
  job_complete: { label: 'A/B plan complete', agent: 'Experiment Agent', icon: TestTube2 },
  job_error: { label: 'Job error', agent: 'System', icon: Ban },
}

const agentBoard = [
  { agent: 'Capture Agent', task: 'Ingest URL, HTML, screenshots, ads, flyers, social posts, and general images.', icon: Upload },
  { agent: 'Demographics Agent', task: 'Find outreach segments and select the audience lens for the run.', icon: Users },
  { agent: 'Attention Agent', task: 'Predict scan path, fixation points, heatmap zones, and ignored areas.', icon: Eye },
  { agent: 'Buyer-Response Scorer', task: 'Score trust, desire, memory, relevance, cognitive load, and CTA strength.', icon: Brain },
  { agent: 'Buyer Panel Agents', task: 'Simulate confusion, trust, desire, urgency, and CTA reactions.', icon: MousePointerClick },
  { agent: 'Growth Strategist Agent', task: 'Connect attention patterns to conversion blockers.', icon: Target },
  { agent: 'Creative Agent', task: 'Write demographic-tuned copy, CTA, layout, and visual instructions.', icon: Pencil },
  { agent: 'Image Editing Agent', task: 'Generate a tuned creative variant from the asset.', icon: Sparkles },
  { agent: 'Constraint Guard', task: 'Protect locked brand, logo, legal, and layout rules.', icon: Lock },
  { agent: 'Experiment Agent', task: 'Package the winning variant into an A/B launch plan.', icon: TestTube2 },
]

const defaultConstraints: Constraints = {
  brand: { colors: ['#0D7D59'], fonts: ['Inter'], tone: 'clear, confident, never hypey', logo_present: false },
  locked_elements: [],
  aggressiveness: 'balanced',
}

function compactEvent(event: StreamEvent) {
  const keep = Object.fromEntries(
    Object.entries(event).filter(([key]) => !['seq', 'ts', 'event', 'agent', 'regions'].includes(key)),
  )
  return JSON.stringify(keep, null, 2)
}

function App() {
  const [mode, setMode] = useState<SourceMode>('url')
  const [url, setUrl] = useState('https://example.com')
  const [html, setHtml] = useState('<main><h1>Grow faster with Fixate</h1><p>See what buyers notice before you launch.</p><button>Start now</button></main>')
  const [imageBase64, setImageBase64] = useState('')
  const [imageName, setImageName] = useState('')
  const [targetCustomer, setTargetCustomer] = useState('startup founder')
  const [demographicTarget, setDemographicTarget] = useState('')
  const [autoFindDemographics, setAutoFindDemographics] = useState(true)
  const [goal, setGoal] = useState('increase signups')
  const [iterations, setIterations] = useState(2)
  const [constraints, setConstraints] = useState<Constraints>(defaultConstraints)
  const [newColor, setNewColor] = useState('#E91E63')
  const [fontInput, setFontInput] = useState('Inter, Poppins')
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [result, setResult] = useState<Result | null>(null)
  const [discoveredSegments, setDiscoveredSegments] = useState<DemographicSegment[]>([])
  const [selectedSegment, setSelectedSegment] = useState<DemographicSegment | null>(null)
  const [apiStatus, setApiStatus] = useState<ApiStatus>({})
  const [jobId, setJobId] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [imageUrl, setImageUrl] = useState('')
  const [heatmapUrl, setHeatmapUrl] = useState('')
  const [latestVariantUrl, setLatestVariantUrl] = useState('')
  const [preview, setPreview] = useState<'heatmap' | 'screenshot' | 'best'>('heatmap')
  const [regions, setRegions] = useState<FixationRegion[]>([])
  const [scanPathCount, setScanPathCount] = useState(0)
  const [imageDims, setImageDims] = useState<{ w: number; h: number } | null>(null)
  const [activeRank, setActiveRank] = useState<number | null>(null)
  const [explanations, setExplanations] = useState<Record<number, string>>({})
  const [explainLoading, setExplainLoading] = useState<number | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => () => esRef.current?.close(), [])

  useEffect(() => {
    async function checkOpenAI() {
      try {
        const health = await fetch('/health').then(response => response.json())
        setApiStatus(prev => ({ ...prev, health }))
        if (health.openai_configured) {
          const debug = await fetch('/debug/openai').then(response => response.json())
          setApiStatus({ health, debug })
        }
      } catch {
        setApiStatus({})
      }
    }
    checkOpenAI()
  }, [])

  const scorePoints = useMemo(() => {
    const points: { label: string; score: number }[] = []
    events.forEach(event => {
      if (event.event === 'scored') points.push({ label: 'Base', score: Number(event.fixate_score) })
      if (event.event === 'variant_scored') points.push({ label: `V${event.iteration}`, score: Number(event.fixate_score) })
      if (event.event === 'job_complete') points.push({ label: 'Final', score: Number(event.final_score) })
    })
    return points
  }, [events])

  const activeAgents = useMemo(() => new Set(events.map(event => event.agent || eventMeta[event.event]?.agent)), [events])

  const zoneCounts = useMemo(() => {
    const currentRegions = result?.final?.regions || result?.baseline?.regions || []
    return currentRegions.reduce<Record<string, number>>((acc, region) => {
      acc[region.zone] = (acc[region.zone] || 0) + 1
      return acc
    }, {})
  }, [result])

  function sourceBody() {
    if (mode === 'url') return { url }
    if (mode === 'image') return { image_base64: imageBase64, image_name: imageName }
    return { html }
  }

  function updateLock(type: string, checked: boolean, value?: string) {
    setConstraints(prev => {
      const rest = prev.locked_elements.filter(item => item.type !== type)
      return { ...prev, locked_elements: checked ? [...rest, { type, value }] : rest }
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

  function readImage(file?: File) {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      setImageBase64(String(reader.result || ''))
      setImageName(file.name)
      setMode('image')
    }
    reader.readAsDataURL(file)
  }

  async function startJob(nextConstraints = constraints, nextDemographic = demographicTarget) {
    if (mode === 'image' && !imageBase64) {
      setError('Choose an image before running Fixate.')
      return
    }
    esRef.current?.close()
    setBusy(true)
    setError('')
    setEvents([])
    setResult(null)
    setJobId('')
    setImageUrl('')
    setHeatmapUrl('')
    setLatestVariantUrl('')
    setRegions([])
    setActiveRank(null)
    setExplanations({})
    try {
      const response = await fetch('/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...sourceBody(),
          target_customer: targetCustomer,
          demographic_target: nextDemographic || null,
          auto_find_demographics: autoFindDemographics,
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
      Object.keys(eventMeta).forEach(name => {
        es.addEventListener(name, message => {
          const event = JSON.parse((message as MessageEvent).data) as StreamEvent
          setEvents(prev => [...prev, event])
          if (event.event === 'capture_done' && event.image_url) setImageUrl(event.image_url)
          if (event.event === 'demographics_ready') {
            const segments = Array.isArray(event.segments) ? event.segments as DemographicSegment[] : []
            setDiscoveredSegments(segments)
            setSelectedSegment(event.selected || null)
          }
          if (event.event === 'heatmap_ready' && event.heatmap_url) {
            setHeatmapUrl(event.heatmap_url)
            setPreview('heatmap')
            if (Array.isArray(event.regions)) setRegions(event.regions as FixationRegion[])
            if (typeof event.scan_path_count === 'number') setScanPathCount(event.scan_path_count)
            if (event.image_width && event.image_height) setImageDims({ w: event.image_width, h: event.image_height })
            setActiveRank(null)
            setExplanations({})
          }
          if (event.event === 'variant_applied' && event.image_url) {
            setLatestVariantUrl(event.image_url)
            setPreview('best')
          }
          if (event.event === 'job_complete') {
            es.close()
            fetch(`/job/${data.job_id}/result`)
              .then(r => r.json())
              .then(payload => {
                setResult(payload)
                setDiscoveredSegments(payload.demographics || [])
                setSelectedSegment(payload.selected_demographic || null)
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

  function chooseSegment(segment: DemographicSegment) {
    setSelectedSegment(segment)
    setDemographicTarget(segment.name)
  }

  function rerunForSegment(segment: DemographicSegment) {
    chooseSegment(segment)
    startJob(constraints, segment.name)
  }

  function unlockAndRerun(blockedType?: string) {
    const type = blockedType || 'layout'
    const next = { ...constraints, locked_elements: constraints.locked_elements.filter(item => item.type !== type) }
    setConstraints(next)
    startJob(next)
  }

  async function explainRegion(rank: number) {
    if (activeRank === rank) {
      setActiveRank(null)
      return
    }
    setActiveRank(rank)
    if (explanations[rank] || !jobId) return
    setExplainLoading(rank)
    try {
      const response = await fetch(`/job/${jobId}/region/${rank}/explain`)
      if (!response.ok) throw new Error(await response.text())
      const data = await response.json()
      setExplanations(prev => ({ ...prev, [rank]: data.explanation }))
    } catch {
      setExplanations(prev => ({ ...prev, [rank]: 'Could not load explanation.' }))
    } finally {
      setExplainLoading(null)
    }
  }

  const previewSrc =
    preview === 'best' && (result?.best_image_url || latestVariantUrl)
      ? result?.best_image_url || latestVariantUrl
      : preview === 'screenshot'
        ? imageUrl || result?.image_url
        : heatmapUrl || result?.heatmap_url

  return (
    <main className="shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Fixate</h1>
            <p>Simulate, segment, tune, and package pre-launch creative tests.</p>
          </div>
          <div className="scoreBadge">
            <Activity size={18} />
            <span>{result?.final.fixate_score ?? scorePoints.at(-1)?.score ?? '--'}</span>
          </div>
        </header>

        <section className={`apiBanner ${apiStatus.debug?.ok ? 'ok' : apiStatus.debug ? 'bad' : ''}`}>
          <strong>{apiStatus.health?.provider_label || 'OpenAI API'}</strong>
          <span>
            {apiStatus.health?.meta_tribe_demo
              ? `Demo adapter enabled; runtime: ${apiStatus.health.runtime_label || 'existing pipeline'}`
              : apiStatus.debug
              ? apiStatus.debug.ok
                ? `Live on ${apiStatus.debug.model}`
                : `${apiStatus.debug.error_type || 'Error'} on ${apiStatus.debug.model}: ${apiStatus.debug.error || 'OpenAI call failed'}`
              : apiStatus.health
                ? `Configured for ${apiStatus.health.openai_model}`
                : 'Checking backend status...'}
          </span>
        </section>

        <section className="agentBoard">
          {agentBoard.map(({ agent, task, icon: Icon }) => (
            <article key={agent} className={activeAgents.has(agent) ? 'active' : ''}>
              <Icon size={17} />
              <strong>{agent}</strong>
              <span>{task}</span>
            </article>
          ))}
        </section>

        <div className="appGrid">
          <section className="panel controls">
            <div className="segmented three">
              <button className={mode === 'url' ? 'active' : ''} onClick={() => setMode('url')}><Crosshair size={15} />URL</button>
              <button className={mode === 'html' ? 'active' : ''} onClick={() => setMode('html')}><FileCode2 size={15} />HTML</button>
              <button className={mode === 'image' ? 'active' : ''} onClick={() => setMode('image')}><ImageIcon size={15} />Image</button>
            </div>

            {mode === 'url' && (
              <label>URL<input value={url} onChange={event => setUrl(event.target.value)} /></label>
            )}
            {mode === 'html' && (
              <label>HTML<textarea rows={7} value={html} onChange={event => setHtml(event.target.value)} /></label>
            )}
            {mode === 'image' && (
              <div className="uploadBox">
                <label className="filePick">
                  <Upload size={18} />
                  <span>{imageName || 'Choose campaign image'}</span>
                  <input type="file" accept="image/*" onChange={event => readImage(event.target.files?.[0])} />
                </label>
                {imageBase64 && <img src={imageBase64} alt="Uploaded campaign asset" />}
              </div>
            )}

            <label>Target customer hint<input value={targetCustomer} onChange={event => setTargetCustomer(event.target.value)} /></label>
            <label>Goal<input value={goal} onChange={event => setGoal(event.target.value)} /></label>
            <label>Demographic focus<input value={demographicTarget} onChange={event => setDemographicTarget(event.target.value)} placeholder="Leave blank to let Fixate choose" /></label>
            <label className="check"><input type="checkbox" checked={autoFindDemographics} onChange={event => setAutoFindDemographics(event.target.checked)} /> Find product demographics</label>
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
              {previewSrc ? (
                <div className="previewScroll">
                  <div className="heatmapStage">
                    <img src={previewSrc} alt="Fixate visual preview" onClick={() => window.open(previewSrc, '_blank')} title="Open full size" />
                    {preview === 'heatmap' && imageDims && regions
                      .filter(region => region.rank <= scanPathCount)
                      .map(region => (
                        <div
                          key={region.rank}
                          className="fixMarkerWrap"
                          style={{
                            left: `${(region.peak_coords[0] / imageDims.w) * 100}%`,
                            top: `${(region.peak_coords[1] / imageDims.h) * 100}%`,
                          }}
                        >
                          <button className={`fixHotspot ${activeRank === region.rank ? 'active' : ''}`} onClick={() => explainRegion(region.rank)} title={`Explain fixation #${region.rank}`} />
                          {activeRank === region.rank && (
                            <div className="fixPopover">
                              <strong>Fixation point #{region.rank}</strong>
                              <p>{explainLoading === region.rank ? 'Analyzing this point...' : (explanations[region.rank] || '')}</p>
                            </div>
                          )}
                        </div>
                      ))}
                  </div>
                </div>
              ) : <div className="empty">Run a job to generate the capture, heatmap, and tuned image.</div>}
            </section>

            <section className="splitGrid">
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

              <section className="panel">
                <h2><Users size={18} /> Demographic Focus</h2>
                {selectedSegment ? (
                  <article className="selectedSegment">
                    <strong>{selectedSegment.name}</strong>
                    <p>{selectedSegment.summary}</p>
                    <span>{selectedSegment.recommended_channel}</span>
                  </article>
                ) : <div className="empty">Audience segments appear after the Demographics Agent runs.</div>}
              </section>
            </section>

            <section className="panel">
              <h2><Users size={18} /> Outreach Segments</h2>
              <div className="segmentGrid">
                {(discoveredSegments.length ? discoveredSegments : result?.demographics || []).map(segment => (
                  <article key={segment.id} className={selectedSegment?.id === segment.id ? 'selected' : ''}>
                    <div>
                      <strong>{segment.name}</strong>
                      <p>{segment.summary}</p>
                    </div>
                    <dl>
                      <dt>Message</dt><dd>{segment.messaging_angle}</dd>
                      <dt>Visual</dt><dd>{segment.visual_direction}</dd>
                      <dt>Channel</dt><dd>{segment.recommended_channel}</dd>
                    </dl>
                    <button onClick={() => rerunForSegment(segment)} disabled={busy}>
                      <Target size={16} />
                      Tune to this
                    </button>
                  </article>
                ))}
                {!discoveredSegments.length && !result?.demographics?.length && <div className="empty">Demographics Agent output will land here.</div>}
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
                <h2>Live Agent Work</h2>
                <span>{jobId || 'No active job'}</span>
              </div>
              <div className="timeline">
                {events.map(event => {
                  const meta = eventMeta[event.event] || { label: event.event, agent: event.agent || 'Fixate', icon: Activity }
                  const Icon = meta.icon
                  return (
                    <article key={event.seq} className={`event ${event.event === 'edit_blocked' ? 'blocked' : ''}`}>
                      <Icon size={16} />
                      <div>
                        <strong>
                          <span>{event.agent || meta.agent}</span>
                          <em>{meta.label}</em>
                          {typeof event.live === 'boolean' && (
                            <span className={`liveBadge ${event.live ? 'live' : 'fallback'}`}>
                              {event.live ? 'OpenAI' : 'fallback'}
                            </span>
                          )}
                        </strong>
                        <pre>{compactEvent(event)}</pre>
                      </div>
                    </article>
                  )
                })}
                {!events.length && <div className="empty">SSE progress events will stream here by agent.</div>}
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
              <div><span>Audience</span><strong>{result.selected_demographic?.name || result.experiment_plan.target_audience || 'General'}</strong></div>
            </section>

            <section className="panel">
              <h2><ImageIcon size={18} /> Ranked Variants</h2>
              <div className="variantGrid">
                {result.variants.map(variant => (
                  <article key={variant.id} className="variantCard">
                    {variant.image_url && <img src={variant.image_url} alt={variant.id} />}
                    <div className="variantBody">
                      <div className="variantTop">
                        <strong>{variant.target_blocker}</strong>
                        <span className={variant.accepted ? 'accepted' : 'rejected'}>{variant.accepted ? 'accepted' : 'rejected'}</span>
                      </div>
                      {variant.demographic_focus && <small>{variant.demographic_focus}</small>}
                      <p>{variant.explanation || variant.description}</p>
                      <code>{variant.rewritten_copy}</code>
                      <div className="delta">{variant.before_score} to {variant.after_score} ({variant.delta > 0 ? '+' : ''}{variant.delta})</div>
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
                      Unlock and re-run
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
