'use client'

import { Eraser, FileText, RefreshCw, Search } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import {
  defaultModel,
  modelNames,
  modelOptions,
  type ModelName,
  type PipelineModel,
} from '@/components/preferences/models'
import { ModelOptions } from '@/components/preferences/PipelinePreferences'
import {
  NumberField,
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { getServer, unloadServer, updateServer, warmupServer } from '@/lib/backend'
import type {
  DetectionModel,
  InpaintingModel,
  OcrModel,
  ServerConfig,
  ServerStatus,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Switch } from '@koharu/ui/components/switch'

const defaultServerConfig: ServerConfig = {
  enabled: false,
  require_api_token: true,
  cpu: false,
  features: {
    detection: true,
    ocr: true,
    inpainting: true,
  },
  detection: defaultModel('koharu-layout-rfdetr-seg-2xl') as DetectionModel,
  ocr: defaultModel('paddleocr-vl-1.6') as OcrModel,
  inpainting: defaultModel('lama') as InpaintingModel,
  max_request_mib: 64,
  max_image_pixels: 32000000,
  max_queue: 4,
}

function toMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) return error.message
  if (typeof error === 'string' && error.trim().length > 0) return error
  return fallback
}

export function ServerPreferences() {
  const { t } = useTranslation()
  const [status, setStatus] = useState<ServerStatus | null>(null)
  const [draft, setDraft] = useState<ServerConfig>(defaultServerConfig)
  const [loading, setLoading] = useState(true)
  const [loadedOnce, setLoadedOnce] = useState(false)
  const [saving, setSaving] = useState(false)
  const [actionBusy, setActionBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function loadStatus() {
    setLoading(true)
    setError(null)
    try {
      const current = await getServer()
      setStatus(current)
    } catch (cause) {
      setError(toMessage(cause, t('settings.server.loadFailed')))
      throw cause
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let mounted = true
    void getServer()
      .then((current) => {
        if (!mounted) return
        setStatus(current)
        setDraft(current.config)
        setLoadedOnce(true)
      })
      .catch((cause) => {
        if (!mounted) return
        setError(toMessage(cause, t('settings.server.loadFailed')))
      })
      .finally(() => {
        if (mounted) setLoading(false)
      })
    return () => {
      mounted = false
    }
  }, [t])

  async function handleApply() {
    setSaving(true)
    setError(null)
    try {
      const updated = await updateServer(draft)
      setStatus(updated)
      setDraft(updated.config)
    } catch (cause) {
      setError(toMessage(cause, t('settings.server.applyFailed')))
      throw cause
    } finally {
      setSaving(false)
    }
  }

  async function handleUnload() {
    setActionBusy('unload')
    setError(null)
    try {
      const updated = await unloadServer()
      setStatus(updated)
    } catch (cause) {
      setError(toMessage(cause, t('settings.server.unloadFailed')))
      throw cause
    } finally {
      setActionBusy(null)
    }
  }

  async function handleWarmup(stage: 'detection' | 'ocr' | 'inpainting') {
    setActionBusy(`warmup-${stage}`)
    setError(null)
    try {
      const updated = await warmupServer(stage)
      setStatus(updated)
    } catch (cause) {
      setError(toMessage(cause, t('settings.server.warmupFailed')))
      throw cause
    } finally {
      setActionBusy(null)
    }
  }

  if (loading && !status) {
    return (
      <PreferencePage
        title={t('settings.server.title')}
        description={t('settings.server.description')}
      >
        {error && (
          <div className='grid gap-3'>
            <p
              className='rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-[12px] text-destructive'
              role='alert'
            >
              {error}
            </p>
            <div>
              <Button
                type='button'
                size='sm'
                variant='outline'
                onClick={() => void loadStatus().catch(() => undefined)}
              >
                <RefreshCw className='mr-1 size-3' />
                {t('settings.server.refresh')}
              </Button>
            </div>
          </div>
        )}
        <p className='py-8 text-[12px] text-muted-foreground'>{t('settings.loading')}</p>
      </PreferencePage>
    )
  }

  return (
    <PreferencePage
      title={t('settings.server.title')}
      description={t('settings.server.description')}
    >
      {error && (
        <p
          className='rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-[12px] text-destructive'
          role='alert'
        >
          {error}
        </p>
      )}
      <PreferenceSection
        title={t('settings.server.status')}
        description={t('settings.server.statusDescription')}
      >
        <PreferenceRow
          title={t('settings.server.currentStatus')}
          description={
            status?.last_error
              ? t('settings.server.lastError', { error: status.last_error })
              : t('settings.server.statusCounters', {
                  queued: status?.queued ?? 0,
                  completed: status?.completed ?? 0,
                  failed: status?.failed ?? 0,
                })
          }
        >
          <div className='flex flex-wrap items-center gap-2'>
            <span className='rounded bg-muted px-2 py-0.5 text-[11px] font-medium'>
              {status?.active
                ? t('settings.server.activeStage', { stage: status.active })
                : t('settings.server.idle')}
            </span>
            <span className='rounded border px-2 py-0.5 text-[11px] text-muted-foreground'>
              {t('settings.server.loadedModel', {
                loaded: status?.loaded ?? t('settings.server.none'),
              })}
            </span>
            <Button
              type='button'
              size='sm'
              variant='outline'
              disabled={loading}
              onClick={() => void loadStatus().catch(() => undefined)}
            >
              <RefreshCw className='mr-1 size-3' />
              {t('settings.server.refresh')}
            </Button>
            <Button
              type='button'
              size='sm'
              variant='outline'
              disabled={actionBusy !== null}
              onClick={() => void handleUnload().catch(() => undefined)}
            >
              {t('settings.server.unload')}
            </Button>
          </div>
        </PreferenceRow>

        <PreferenceRow
          title={t('settings.server.warmup')}
          description={t('settings.server.warmupDescription')}
        >
          <div className='flex gap-2'>
            <Button
              type='button'
              size='sm'
              variant='outline'
              disabled={actionBusy !== null}
              onClick={() => void handleWarmup('detection').catch(() => undefined)}
            >
              {t('settings.pipeline.stages.detection.title')}
            </Button>
            <Button
              type='button'
              size='sm'
              variant='outline'
              disabled={actionBusy !== null}
              onClick={() => void handleWarmup('ocr').catch(() => undefined)}
            >
              {t('settings.pipeline.stages.ocr.title')}
            </Button>
            <Button
              type='button'
              size='sm'
              variant='outline'
              disabled={actionBusy !== null}
              onClick={() => void handleWarmup('inpainting').catch(() => undefined)}
            >
              {t('settings.pipeline.stages.inpainting.title')}
            </Button>
          </div>
        </PreferenceRow>
      </PreferenceSection>

      <PreferenceSection
        title={t('settings.server.configuration')}
        description={t('settings.server.configurationDescription')}
      >
        <PreferenceRow
          title={t('settings.server.enabled')}
          description={t('settings.server.enabledDescription')}
        >
          <Switch
            checked={draft.enabled}
            aria-label={t('settings.server.enabled')}
            onCheckedChange={(enabled) => setDraft({ ...draft, enabled })}
          />
        </PreferenceRow>

        <PreferenceRow
          title={t('settings.server.requireToken')}
          description={t('settings.server.requireTokenDescription')}
        >
          <Switch
            checked={draft.require_api_token}
            aria-label={t('settings.server.requireToken')}
            onCheckedChange={(require_api_token) => setDraft({ ...draft, require_api_token })}
          />
        </PreferenceRow>

        <PreferenceRow
          title={t('settings.server.cpu')}
          description={t('settings.server.cpuDescription')}
        >
          <Switch
            checked={draft.cpu}
            aria-label={t('settings.server.cpu')}
            onCheckedChange={(cpu) => setDraft({ ...draft, cpu })}
          />
        </PreferenceRow>

        <PreferenceRow
          title={t('settings.server.features')}
          description={t('settings.server.featuresDescription')}
          align='start'
        >
          <div className='grid gap-2'>
            <label className='flex items-center justify-between gap-2 text-[12px]'>
              <span>{t('settings.pipeline.stages.detection.title')}</span>
              <Switch
                checked={draft.features.detection}
                aria-label={t('settings.pipeline.stages.detection.title')}
                onCheckedChange={(detection) =>
                  setDraft({
                    ...draft,
                    features: { ...draft.features, detection },
                  })
                }
              />
            </label>
            <label className='flex items-center justify-between gap-2 text-[12px]'>
              <span>{t('settings.pipeline.stages.ocr.title')}</span>
              <Switch
                checked={draft.features.ocr}
                aria-label={t('settings.pipeline.stages.ocr.title')}
                onCheckedChange={(ocr) =>
                  setDraft({ ...draft, features: { ...draft.features, ocr } })
                }
              />
            </label>
            <label className='flex items-center justify-between gap-2 text-[12px]'>
              <span>{t('settings.pipeline.stages.inpainting.title')}</span>
              <Switch
                checked={draft.features.inpainting}
                aria-label={t('settings.pipeline.stages.inpainting.title')}
                onCheckedChange={(inpainting) =>
                  setDraft({
                    ...draft,
                    features: { ...draft.features, inpainting },
                  })
                }
              />
            </label>
          </div>
        </PreferenceRow>

        <PreferenceRow
          title={t('settings.server.limits')}
          description={t('settings.server.limitsDescription')}
          align='start'
        >
          <div className='grid grid-cols-3 gap-2'>
            <NumberField
              label={t('settings.server.maxRequestMib')}
              value={draft.max_request_mib}
              min={1}
              max={256}
              step={1}
              onChange={(max_request_mib) =>
                max_request_mib !== null && setDraft({ ...draft, max_request_mib })
              }
            />
            <NumberField
              label={t('settings.server.maxImagePixels')}
              value={draft.max_image_pixels}
              min={1}
              max={64000000}
              step={1000000}
              onChange={(max_image_pixels) =>
                max_image_pixels !== null && setDraft({ ...draft, max_image_pixels })
              }
            />
            <NumberField
              label={t('settings.server.maxQueue')}
              value={draft.max_queue}
              min={0}
              max={32}
              step={1}
              onChange={(max_queue) => max_queue !== null && setDraft({ ...draft, max_queue })}
            />
          </div>
        </PreferenceRow>
      </PreferenceSection>

      <PreferenceSection
        title={t('settings.server.models')}
        description={t('settings.server.modelsDescription')}
      >
        <ServerModelRow
          stage='detection'
          icon={Search}
          model={draft.detection}
          onChange={(model) => setDraft({ ...draft, detection: model as DetectionModel })}
        />
        <ServerModelRow
          stage='ocr'
          icon={FileText}
          model={draft.ocr}
          onChange={(model) => setDraft({ ...draft, ocr: model as OcrModel })}
        />
        <ServerModelRow
          stage='inpainting'
          icon={Eraser}
          model={draft.inpainting}
          onChange={(model) => setDraft({ ...draft, inpainting: model as InpaintingModel })}
        />
      </PreferenceSection>

      <div className='flex justify-end pt-4'>
        <Button
          type='button'
          disabled={saving || !loadedOnce || !status}
          onClick={() => void handleApply().catch(() => undefined)}
        >
          {t(saving ? 'settings.server.applying' : 'settings.server.apply')}
        </Button>
      </div>
    </PreferencePage>
  )
}

function ServerModelRow({
  stage,
  icon: Icon,
  model,
  onChange,
}: {
  stage: 'detection' | 'ocr' | 'inpainting'
  icon: typeof Search
  model: PipelineModel
  onChange: (model: PipelineModel) => void
}) {
  const { t } = useTranslation()
  const title = t(`settings.pipeline.stages.${stage}.title`)

  return (
    <PreferenceRow
      title={title}
      description={t(`settings.pipeline.stages.${stage}.description`)}
      align='start'
    >
      <div className='grid gap-3'>
        <div className='flex items-center gap-2'>
          <Icon className='size-3.5 shrink-0 text-muted-foreground' />
          <Select
            value={model.model}
            items={Object.fromEntries(modelOptions[stage].map((name) => [name, modelNames[name]]))}
            onValueChange={(name) => {
              if (name) {
                onChange(defaultModel(name as ModelName))
              }
            }}
          >
            <SelectTrigger
              aria-label={t('settings.pipeline.modelLabel', { stage: title })}
              className='h-8 min-w-0 flex-1 text-[11px]'
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {modelOptions[stage].map((name) => (
                <SelectItem key={name} value={name}>
                  {modelNames[name]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <ModelOptions model={model} onChange={onChange} />
      </div>
    </PreferenceRow>
  )
}
