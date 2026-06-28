import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ConvexProvider, ConvexReactClient, useMutation, useQuery } from 'convex/react'
import {
  Activity,
  ArrowRight,
  Ban,
  Brain,
  Check,
  Crosshair,
  Eye,
  FileCode2,
  Gauge,
  Image as ImageIcon,
  KeyRound,
  LineChart,
  Lock,
  Mail,
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
  Zap,
} from 'lucide-react'
import { api } from '../convex/_generated/api'
import heroImage from './assets/fixate-hero.png'
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
type SavedAnalysisJob = {
  _id: string
  externalJobId: string
  sourceType: string
  targetCustomer: string
  goal: string
  status: 'running' | 'complete' | 'failed'
  finalScore?: number
  heatmapUrl?: string
  bestImageUrl?: string
  selectedAudience?: string
  updatedAt: number
}
type ConvexSync = {
  enabled: boolean
  savedJobs: SavedAnalysisJob[]
  createJob: (args: {
    externalJobId: string
    sourceType: SourceMode
    targetCustomer: string
    demographicTarget?: string
    goal: string
  }) => Promise<unknown>
  addEvent: (args: {
    externalJobId: string
    event: string
    agent?: string
    label?: string
    payload: Record<string, unknown>
  }) => Promise<unknown>
  completeJob: (args: {
    externalJobId: string
    baselineScore?: number
    finalScore?: number
    heatmapUrl?: string
    originalImageUrl?: string
    bestImageUrl?: string
    selectedAudience?: string
  }) => Promise<unknown>
  failJob: (args: { externalJobId: string; errorMessage: string }) => Promise<unknown>
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

// Order matches the backend's actual SSE emission sequence (pipeline/loop.py):
// capture -> attention(heatmap) -> scoring -> audience(demographics) -> panel -> diagnosis -> creative -> image -> guard -> experiment
const pipelineStages = [
  { key: 'capture', label: 'Capture', icon: Upload },
  { key: 'attention', label: 'Attention', icon: Eye },
  { key: 'scoring', label: 'Scoring', icon: Brain },
  { key: 'demographics', label: 'Audience', icon: Users },
  { key: 'panel', label: 'Buyer Panel', icon: MousePointerClick },
  { key: 'diagnosis', label: 'Diagnosis', icon: Target },
  { key: 'creative', label: 'Creative', icon: Pencil },
  { key: 'image', label: 'Image Edit', icon: Sparkles },
  { key: 'guard', label: 'Constraints', icon: Lock },
  { key: 'experiment', label: 'A/B Plan', icon: TestTube2 },
]

// Stages tied to generating a new ad — hidden from the pipeline when image generation is off.
const imageStageKeys = new Set(['creative', 'image', 'guard'])

const eventToStageKey: Record<string, string> = {
  capture_started: 'capture',
  capture_done: 'capture',
  heatmap_ready: 'attention',
  scored: 'scoring',
  demographics_started: 'demographics',
  demographics_ready: 'demographics',
  buyer_panel: 'panel',
  diagnosis_ready: 'diagnosis',
  blocker_found: 'diagnosis',
  variant_proposed: 'creative',
  variant_applied: 'image',
  variant_image_failed: 'image',
  variant_scored: 'image',
  edit_blocked: 'guard',
  iteration_done: 'image',
  job_complete: 'experiment',
}

const stageIndexByKey: Record<string, number> = Object.fromEntries(
  pipelineStages.map((stage, index) => [stage.key, index]),
)

const defaultConstraints: Constraints = {
  brand: { colors: ['#0D7D59'], fonts: ['Inter'], tone: 'clear, confident, never hypey', logo_present: false },
  locked_elements: [],
  aggressiveness: 'balanced',
}

const disabledConvexSync: ConvexSync = {
  enabled: false,
  savedJobs: [],
  createJob: async () => undefined,
  addEvent: async () => undefined,
  completeJob: async () => undefined,
  failJob: async () => undefined,
}

function remember(promise: Promise<unknown>) {
  promise.catch(error => {
    console.warn('Convex sync failed', error)
  })
}

function compactEvent(event: StreamEvent) {
  const keep = Object.fromEntries(
    Object.entries(event).filter(([key]) => !['seq', 'ts', 'event', 'agent', 'regions'].includes(key)),
  )
  return JSON.stringify(keep, null, 2)
}

function TopNav({ onStart, onLogin }: { onStart: () => void; onLogin: () => void }) {
  return (
    <nav className="topNav">
      <button className="brand" onClick={onStart}>
        <span className="brandMark"><Target size={17} /></span>
        <span>Fixate</span>
      </button>
      <div className="navLinks">
        <span>Attention</span>
        <span>Audience</span>
        <span>Variants</span>
        <span>Experiments</span>
      </div>
      <div className="navActions">
        <button className="ghost" onClick={onLogin}>Log in</button>
        <button className="cta" onClick={onStart}>
          Launch lab
          <ArrowRight size={16} />
        </button>
      </div>
    </nav>
  )
}

function LandingPage({ onStart, onLogin }: { onStart: () => void; onLogin: () => void }) {
  return (
    <main className="marketingShell">
      <TopNav onStart={onStart} onLogin={onLogin} />

      <section className="hero">
        <div className="heroCopy">
          <span className="eyebrow">Pre-launch creative intelligence</span>
          <h1>Test the ad <span className="grad">before</span> you pay for the click.</h1>
          <p className="lede">
            Fixate simulates how your target buyer experiences a campaign — what they notice,
            what they ignore, and what blocks the sale — then generates stronger variants
            within your brand rules and re-scores them.
          </p>
          <div className="heroActions">
            <button className="cta" onClick={onStart}>
              Start a free analysis
              <ArrowRight size={18} />
            </button>
            <button className="ghost" onClick={onLogin}>Sign in</button>
          </div>
          <div className="heroProof">
            <div><strong>84</strong><span>Avg Fixate Score</span></div>
            <div><strong>+18</strong><span>Best variant lift</span></div>
            <div><strong>10</strong><span>AI agents per run</span></div>
          </div>
        </div>

        <div className="heroVisual">
          <div className="heroFrame">
            <img src={heroImage} alt="Fixate creative analysis dashboard" />
          </div>
          <div className="heroChip tl"><Eye size={15} /> Attention mapped</div>
          <div className="heroChip br"><Sparkles size={15} /> Variant +18</div>
        </div>
      </section>

      <div className="logoStrip">
        <span>URL</span>
        <span>HTML</span>
        <span>Images</span>
        <span>Ads & flyers</span>
        <span>Landing pages</span>
        <span>Social posts</span>
        <span>Emails</span>
      </div>

      <section className="features">
        <article className="featureCard">
          <span className="ic"><Eye size={20} /></span>
          <strong>Attention heatmaps</strong>
          <p>See the predicted scan path, fixation points, ignored value, and attention traps — then click any hotspot to ask why.</p>
        </article>
        <article className="featureCard">
          <span className="ic"><Users size={20} /></span>
          <strong>Audience fit</strong>
          <p>Discover outreach-ready demographic segments and re-tune the creative for whichever buyer you want to win.</p>
        </article>
        <article className="featureCard">
          <span className="ic"><LineChart size={20} /></span>
          <strong>Variants + A/B plan</strong>
          <p>Generate improved creatives within your locked brand rules, watch the score climb, and ship the winner with a launch plan.</p>
        </article>
      </section>

      <section className="bandCta">
        <span className="eyebrow">No traffic required</span>
        <h2>Know what works before launch day.</h2>
        <p>Upload an asset, pick a buyer, set your goal. Fixate does the rest in one live run.</p>
        <button className="cta" onClick={onStart}>
          Launch the lab
          <ArrowRight size={18} />
        </button>
      </section>
    </main>
  )
}

function LoginPage({ onEnter, onBack }: { onEnter: () => void; onBack: () => void }) {
  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    onEnter()
  }

  return (
    <main className="loginShell">
      <section className="loginVisual">
        <button className="brand" onClick={onBack}>
          <span className="brandMark"><Target size={17} /></span>
          <span>Fixate</span>
        </button>
        <div className="pitch">
          <span className="eyebrow">Campaign lab access</span>
          <h1>Enter the workspace</h1>
          <p>Use any path for the demo — each route opens the same Fixate analysis platform.</p>
          <div className="loginQuotes">
            <div><Check size={16} /> Predict buyer attention before launch</div>
            <div><Check size={16} /> Diagnose what blocks the conversion</div>
            <div><Check size={16} /> Generate stronger variants within brand rules</div>
          </div>
        </div>
      </section>

      <section className="loginPanel">
        <div>
          <h2>Welcome back</h2>
          <p className="sub">Continue to creative testing and buyer-response scoring.</p>
        </div>
        <form onSubmit={submit}>
          <label>
            Email
            <span className="inputIcon">
              <Mail size={16} />
              <input type="email" placeholder="team@company.com" />
            </span>
          </label>
          <label>
            Password
            <span className="inputIcon">
              <KeyRound size={16} />
              <input type="password" placeholder="Password" />
            </span>
          </label>
          <button className="cta loginPrimary" type="submit">
            Continue
            <ArrowRight size={17} />
          </button>
        </form>
        <div className="divider">or</div>
        <div className="loginOptions">
          <button onClick={onEnter}>Continue with Google</button>
          <button onClick={onEnter}>Continue with GitHub</button>
          <button className="cta" onClick={onEnter}>Use demo account</button>
        </div>
      </section>
    </main>
  )
}

function App({ convexSync = disabledConvexSync }: { convexSync?: ConvexSync }) {
  const [view, setView] = useState<'landing' | 'login' | 'app'>('landing')
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
  const [generateImages, setGenerateImages] = useState(true)
  const [imagePrompt, setImagePrompt] = useState('')
  const [constraints, setConstraints] = useState<Constraints>(defaultConstraints)
  const [newColor, setNewColor] = useState('#E91E63')
  const [fontInput, setFontInput] = useState('Inter, Poppins')
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [result, setResult] = useState<Result | null>(null)
  const [discoveredSegments, setDiscoveredSegments] = useState<DemographicSegment[]>([])
  const [selectedSegment, setSelectedSegment] = useState<DemographicSegment | null>(null)
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

  const scorePoints = useMemo(() => {
    const points: { label: string; score: number }[] = []
    events.forEach(event => {
      if (event.event === 'scored') points.push({ label: 'Base', score: Number(event.fixate_score) })
      if (event.event === 'variant_scored') points.push({ label: `V${event.iteration}`, score: Number(event.fixate_score) })
      if (event.event === 'job_complete') points.push({ label: 'Final', score: Number(event.final_score) })
    })
    return points
  }, [events])

  const isComplete = useMemo(() => !!result || events.some(event => event.event === 'job_complete'), [result, events])
  const lastEvent = events[events.length - 1]
  // Hide the image-generation stages from the pipeline when the user turns off new ad creation.
  const visibleStages = useMemo(
    () => (generateImages ? pipelineStages : pipelineStages.filter(stage => !imageStageKeys.has(stage.key))),
    [generateImages],
  )
  // Track the furthest canonical stage reached so the active marker never jumps backwards
  // (the backend re-emits earlier events like heatmap_ready during later iterations).
  const currentStage = useMemo(() => {
    let max = -1
    events.forEach(event => {
      const key = eventToStageKey[event.event]
      if (key && stageIndexByKey[key] > max) max = stageIndexByKey[key]
    })
    return max
  }, [events])
  const reachedVisible = visibleStages.filter(stage => stageIndexByKey[stage.key] <= currentStage).length
  const progress = isComplete ? 100 : busy ? Math.min(96, Math.round((reachedVisible / visibleStages.length) * 100)) : 0
  const generatingImage = busy && lastEvent?.event === 'variant_proposed'
  const capturing = busy && !heatmapUrl && !imageUrl && !result

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
          generate_images: generateImages,
          image_prompt: generateImages && imagePrompt.trim() ? imagePrompt.trim() : null,
          constraints: nextConstraints,
        }),
      })
      if (!response.ok) throw new Error(await response.text())
      const data = await response.json()
      setJobId(data.job_id)
      remember(convexSync.createJob({
        externalJobId: data.job_id,
        sourceType: mode,
        targetCustomer,
        demographicTarget: nextDemographic || undefined,
        goal,
      }))
      const es = new EventSource(`/job/${data.job_id}/stream`)
      esRef.current = es
      Object.keys(eventMeta).forEach(name => {
        es.addEventListener(name, message => {
          const event = JSON.parse((message as MessageEvent).data) as StreamEvent
          const meta = eventMeta[event.event]
          remember(convexSync.addEvent({
            externalJobId: data.job_id,
            event: event.event,
            agent: event.agent || meta?.agent,
            label: meta?.label,
            payload: event,
          }))
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
                remember(convexSync.completeJob({
                  externalJobId: data.job_id,
                  baselineScore: payload.baseline?.fixate_score,
                  finalScore: payload.final?.fixate_score,
                  heatmapUrl: payload.heatmap_url,
                  originalImageUrl: payload.image_url,
                  bestImageUrl: payload.best_image_url,
                  selectedAudience: payload.selected_demographic?.name || payload.experiment_plan?.target_audience,
                }))
                if (payload.best_image_url) setPreview('best')
              })
              .finally(() => setBusy(false))
          }
          if (event.event === 'job_error') {
            es.close()
            setError(event.message || 'Job failed.')
            remember(convexSync.failJob({
              externalJobId: data.job_id,
              errorMessage: event.message || 'Job failed.',
            }))
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
  const latestScore = scorePoints[scorePoints.length - 1]?.score
  const showGenOverlay = busy && (capturing || generatingImage)
  const displayScore = result?.final.fixate_score ?? latestScore

  if (view === 'landing') {
    return <LandingPage onStart={() => setView('login')} onLogin={() => setView('login')} />
  }

  if (view === 'login') {
    return <LoginPage onEnter={() => setView('app')} onBack={() => setView('landing')} />
  }

  return (
    <main className="shell">
      <section className="workspace">
        <header className="topbar">
          <div className="lhs">
            <button className="brand" onClick={() => setView('landing')}>
              <span className="brandMark"><Target size={17} /></span>
              <span>Fixate</span>
            </button>
            <span className="sub">Pre-launch creative intelligence lab</span>
          </div>
          <div className="rhs">
            <div className={`scoreBadge ${busy ? 'live' : ''}`}>
              <Gauge size={16} />
              <span className="lbl">Score</span>
              <span className="val">{displayScore ?? '--'}</span>
            </div>
          </div>
        </header>

        <section className={`pipeline ${busy ? 'running' : ''}`}>
          <div className="pipelineTop">
            <div className="pipelineStatus">
              <span className={`statusDot ${isComplete ? 'done' : busy ? 'on' : ''}`} />
              <div>
                <div className="ttl">
                  {isComplete ? 'Analysis complete' : busy ? (lastEvent ? eventMeta[lastEvent.event]?.label ?? 'Working…' : 'Starting run…') : 'Ready to run'}
                </div>
                <div className="meta">{jobId ? `job ${jobId.slice(0, 12)}` : `${visibleStages.length} specialist agents · live SSE stream`}</div>
              </div>
            </div>
            <div className="progressRing">
              <svg width="54" height="54" viewBox="0 0 54 54">
                <defs>
                  <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#2fe6a8" />
                    <stop offset="100%" stopColor="#7c93ff" />
                  </linearGradient>
                </defs>
                <circle className="track" cx="27" cy="27" r="22" fill="none" strokeWidth="5" />
                <circle
                  className="fill"
                  cx="27" cy="27" r="22" fill="none" strokeWidth="5"
                  strokeDasharray={2 * Math.PI * 22}
                  strokeDashoffset={2 * Math.PI * 22 * (1 - progress / 100)}
                />
              </svg>
              <span className="pct">{progress}%</span>
            </div>
          </div>

          <div className="flow">
            {visibleStages.map(stage => {
              const Icon = stage.icon
              const pos = stageIndexByKey[stage.key]
              const done = isComplete || pos < currentStage
              const active = busy && !isComplete && pos === currentStage
              return (
                <div key={stage.key} className={`node ${active ? 'active' : ''} ${done ? 'done' : ''}`}>
                  <span className="connector" />
                  <span className="dot"><Icon size={20} /></span>
                  {done && <span className="check"><Check size={12} /></span>}
                  <span className="nm">{stage.label}</span>
                </div>
              )
            })}
          </div>
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

            <div className={`genToggle ${generateImages ? 'on' : ''}`}>
              <div className="genToggleText">
                <strong><Sparkles size={15} /> Generate a new ad</strong>
                <span>{generateImages ? 'Fixate will design & re-score improved image variants.' : 'Diagnosis only — analyze attention without creating new images.'}</span>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={generateImages}
                className="switch"
                onClick={() => setGenerateImages(value => !value)}
              >
                <span className="knob" />
              </button>
            </div>

            {generateImages && (
              <>
                <label>Iterations
                  <div className="rangeRow">
                    <input type="range" min={1} max={10} value={iterations} onChange={event => setIterations(Number(event.target.value))} />
                    <span className="rangeVal">{iterations}</span>
                  </div>
                </label>

                <label><span className="labelRow">Custom image prompt <span className="optional">optional</span></span>
                  <textarea
                    rows={3}
                    value={imagePrompt}
                    onChange={event => setImagePrompt(event.target.value)}
                    placeholder="Direct the image generator, e.g. “make the CTA bright orange and add a product close-up top-right”."
                  />
                </label>

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
              </>
            )}

            <button className="cta runBtn" onClick={() => startJob()} disabled={busy}>
              {busy ? <RefreshCw size={17} className="spin" /> : <Play size={17} />}
              {busy ? 'Running Fixate…' : 'Run Fixate'}
            </button>
            {error && <div className="error">{error}</div>}

            <section className="convexPanel">
              <div className="convexHead">
                <strong><Activity size={14} /> Convex Sync</strong>
                <span className={`pill ${convexSync.enabled ? 'online' : 'offline'}`}>
                  {convexSync.enabled ? 'active' : 'local only'}
                </span>
              </div>
              <div className="historyList">
                {convexSync.savedJobs.slice(0, 5).map(saved => (
                  <article key={saved._id}>
                    <div>
                      <strong>{saved.goal}</strong>
                      <span>{saved.targetCustomer}</span>
                    </div>
                    <em>{saved.finalScore ?? '--'}</em>
                  </article>
                ))}
                {convexSync.enabled && !convexSync.savedJobs.length && <p>No saved analyses yet.</p>}
                {!convexSync.enabled && <p>Local-only mode.</p>}
              </div>
            </section>
          </section>

          <section className="mainStack">
            <section className="panel previewPanel">
              <div className="panelHeader">
                <h2><Eye size={18} /> Attention Preview</h2>
                <div className="segmented compact three">
                  <button className={preview === 'heatmap' ? 'active' : ''} onClick={() => setPreview('heatmap')}>Heatmap</button>
                  <button className={preview === 'screenshot' ? 'active' : ''} onClick={() => setPreview('screenshot')}>Original</button>
                  <button className={preview === 'best' ? 'active' : ''} onClick={() => setPreview('best')}>Best</button>
                </div>
              </div>
              <div className="previewStage">
                {showGenOverlay && (
                  <div className="genOverlay">
                    <div className="scan" />
                    <div className="genOrb"><Sparkles size={28} /></div>
                    <div className="gTitle">{generatingImage ? 'Generating creative variant' : 'Capturing your asset'}</div>
                    <div className="gSub">
                      {generatingImage ? 'Image Editing Agent is rendering a tuned creative' : 'Reading layout, copy, and visual hierarchy'}
                      <span className="dots"><span /><span /><span /></span>
                    </div>
                  </div>
                )}
                {previewSrc ? (
                  <div className="previewScroll">
                    <div className="heatmapStage">
                      <img key={previewSrc} className="reveal" src={previewSrc} alt="Fixate visual preview" onClick={() => window.open(previewSrc, '_blank')} title="Open full size" />
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
                ) : busy ? (
                  <div style={{ padding: 16 }}><div className="shimmer" /></div>
                ) : (
                  <div className="skeleton">
                    <Eye className="ic" size={40} />
                    <strong>No preview yet</strong>
                    <p>Run an analysis to generate the capture, attention heatmap, and tuned image variant — they’ll stream in here live.</p>
                  </div>
                )}
              </div>
            </section>

            <section className="splitGrid">
              <section className="panel scorePanel">
                <h2><LineChart size={18} /> Score Trend</h2>
                <div className="scoreChart">
                  {scorePoints.map((point, index) => (
                    <div key={`${point.label}-${index}`} className="barWrap">
                      <div className="barTrack">
                        <div className="bar" style={{ height: `${Math.max(6, point.score)}%` }} />
                      </div>
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
                    <span className="chan">{selectedSegment.recommended_channel}</span>
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
                <h2><Brain size={18} /> Buyer Reactions</h2>
                <div className="reactionList">
                  {(result?.buyer_reactions || []).map(reaction => (
                    <article key={reaction.dimension} className={`reaction ${reaction.severity}`}>
                      <strong>{reaction.dimension.replace('_', ' ')}</strong>
                      <span className="sev">{reaction.severity}</span>
                      <p>{reaction.explanation}</p>
                    </article>
                  ))}
                  {!result?.buyer_reactions?.length && <div className="empty">Buyer-panel flags appear in results.</div>}
                </div>
              </section>
              <section className="panel">
                <h2><Crosshair size={18} /> Zone Analysis</h2>
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
                <h2><Zap size={18} /> Live Agent Work</h2>
                <span className="mono">{jobId ? jobId.slice(0, 16) : 'no active job'}</span>
              </div>
              <div className="timeline">
                {events.map(event => {
                  const meta = eventMeta[event.event] || { label: event.event, agent: event.agent || 'Fixate', icon: Activity }
                  const Icon = meta.icon
                  return (
                    <article key={event.seq} className={`event ${event.event === 'edit_blocked' ? 'blocked' : ''}`}>
                      <span className="evIc"><Icon size={15} /></span>
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
              <div><span>Final</span><strong className="up">{result.final.fixate_score}</strong></div>
              <div><span>Delta</span><strong className={result.final.fixate_score >= result.baseline.fixate_score ? 'up' : ''}>{result.final.fixate_score - result.baseline.fixate_score >= 0 ? '+' : ''}{(result.final.fixate_score - result.baseline.fixate_score).toFixed(1)}</strong></div>
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
                      {variant.demographic_focus && <span className="focus">{variant.demographic_focus}</span>}
                      <p>{variant.explanation || variant.description}</p>
                      <code>{variant.rewritten_copy}</code>
                      <div className={`delta ${variant.delta >= 0 ? 'pos' : 'neg'}`}>{variant.before_score} → {variant.after_score} ({variant.delta > 0 ? '+' : ''}{variant.delta})</div>
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
                    <strong><Lock size={15} /> {blocked.blocker}</strong>
                    <p>{blocked.reason}</p>
                    <span className="gain">Estimated gain: +{blocked.estimated_gain}</span>
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

const convexUrl = import.meta.env.VITE_CONVEX_URL
const convexClient = convexUrl ? new ConvexReactClient(convexUrl) : null

function ConvexBackedApp() {
  const createJob = useMutation(api.analyses.createJob)
  const addEvent = useMutation(api.analyses.addEvent)
  const completeJob = useMutation(api.analyses.completeJob)
  const failJob = useMutation(api.analyses.failJob)
  const savedJobs = useQuery(api.analyses.listJobs, {}) as SavedAnalysisJob[] | undefined

  const convexSync = useMemo<ConvexSync>(() => ({
    enabled: true,
    savedJobs: savedJobs || [],
    createJob,
    addEvent,
    completeJob,
    failJob,
  }), [addEvent, completeJob, createJob, failJob, savedJobs])

  return <App convexSync={convexSync} />
}

function Root() {
  if (!convexClient) return <App />
  return (
    <ConvexProvider client={convexClient}>
      <ConvexBackedApp />
    </ConvexProvider>
  )
}

createRoot(document.getElementById('root')!).render(<Root />)
