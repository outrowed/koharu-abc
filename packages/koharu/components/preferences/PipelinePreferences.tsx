'use client'

import { Eraser, FileText, Search } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { Flux2KleinOptions } from '@/components/preferences/Flux2KleinOptions'
import { Flux3KleinOptions } from '@/components/preferences/Flux3KleinOptions'
import { LaMaOptions } from '@/components/preferences/LaMaOptions'
import { MangaInpaintorOptions } from '@/components/preferences/MangaInpaintorOptions'
import { MiGanOptions } from '@/components/preferences/MiGanOptions'
import {
  defaultModel,
  modelNames,
  modelOptions,
  replaceStage,
  stageModel,
  type ModelName,
  type ModelStage,
  type PipelineModel,
} from '@/components/preferences/models'
import { PowerPaintOptions } from '@/components/preferences/PowerPaintOptions'
import {
  NumberField,
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
  TextField,
} from '@/components/preferences/PreferenceFields'
import { CredentialField } from '@/components/preferences/ProviderPreferences'
import { testEndpoint } from '@/lib/backend'
import { remoteEndpointIssue } from '@/lib/remote-endpoint'
import type {
  CredentialInput,
  PipelineConfig,
  RemoteCredentials,
  RemoteEndpointConfig,
  StageExecution,
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

const stages = [
  ['detection', Search],
  ['ocr', FileText],
  ['inpainting', Eraser],
] as const satisfies ReadonlyArray<readonly [ModelStage, typeof Search]>

export function PipelinePreferences({
  value,
  onChange,
  credentials,
  onCredentialsChange,
}: {
  value: PipelineConfig
  onChange: (value: PipelineConfig) => void
  credentials: RemoteCredentials
  onCredentialsChange: (value: RemoteCredentials) => void
}) {
  const { t } = useTranslation()
  return (
    <PreferencePage
      title={t('settings.pipeline.title')}
      description={t('settings.pipeline.description')}
    >
      <PreferenceSection title={t('settings.pipeline.processing')}>
        {stages.map(([stage, Icon]) => {
          const model = stageModel(value, stage)
          const title = t(`settings.pipeline.stages.${stage}.title`)
          return (
            <PreferenceRow
              key={stage}
              title={title}
              description={t(`settings.pipeline.stages.${stage}.description`)}
              align='start'
            >
              <div className='grid gap-3'>
                <RemoteStageSettings
                  stage={stage}
                  execution={value.remote?.[stage] ?? { mode: 'local' }}
                  credential={credentials[stage]}
                  onExecutionChange={(execution) =>
                    onChange({
                      ...value,
                      remote: {
                        detection: value.remote?.detection ?? { mode: 'local' },
                        ocr: value.remote?.ocr ?? { mode: 'local' },
                        inpainting: value.remote?.inpainting ?? { mode: 'local' },
                        [stage]: execution,
                      },
                    })
                  }
                  onCredentialChange={(credential) =>
                    onCredentialsChange({ ...credentials, [stage]: credential })
                  }
                />
                <div className='flex items-center gap-2'>
                  <Icon className='size-3.5 shrink-0 text-muted-foreground' />
                  <Select
                    value={model.model}
                    items={Object.fromEntries(
                      modelOptions[stage].map((name) => [name, modelNames[name]]),
                    )}
                    onValueChange={(name) => {
                      if (name) {
                        onChange(replaceStage(value, stage, defaultModel(name as ModelName)))
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
                <ModelOptions
                  model={model}
                  onChange={(next) => onChange(replaceStage(value, stage, next))}
                />
              </div>
            </PreferenceRow>
          )
        })}
      </PreferenceSection>
    </PreferencePage>
  )
}

function RemoteStageSettings({
  stage,
  execution,
  credential,
  onExecutionChange,
  onCredentialChange,
}: {
  stage: ModelStage
  execution: StageExecution
  credential: CredentialInput
  onExecutionChange: (execution: StageExecution) => void
  onCredentialChange: (credential: CredentialInput) => void
}) {
  const { t } = useTranslation()
  const [test, setTest] = useState<{
    busy: boolean
    message: string | null
    reachable: boolean | null
  }>({
    busy: false,
    message: null,
    reachable: null,
  })
  const remote = execution.mode === 'remote'
  const endpoint: { mode: 'remote' } & RemoteEndpointConfig = remote
    ? execution
    : {
        mode: 'remote',
        base_url: null,
        model: null,
        timeout_seconds: 300,
        max_response_mib: 64,
      }
  const endpointIssue = remote ? remoteEndpointIssue(execution) : null

  async function runTest() {
    setTest({ busy: true, message: null, reachable: null })
    try {
      const result = await testEndpoint(stage, endpoint)
      setTest({ busy: false, message: result.message, reachable: result.reachable })
    } catch (error) {
      setTest({
        busy: false,
        message:
          error instanceof Error ? error.message : t('settings.pipeline.endpoint.testFailed'),
        reachable: false,
      })
    }
  }

  return (
    <div className='grid gap-3 rounded-lg border border-border/70 bg-background/40 p-3'>
      <div className='flex items-center justify-between gap-3'>
        <div>
          <p className='text-[11px] font-medium'>{t('settings.pipeline.endpoint.mode')}</p>
          <p className='text-[10px] text-muted-foreground'>
            {t(
              remote
                ? 'settings.pipeline.endpoint.remoteDescription'
                : 'settings.pipeline.endpoint.localDescription',
            )}
          </p>
        </div>
        <div className='flex items-center gap-2'>
          <span className='text-[10px] text-muted-foreground'>
            {t(remote ? 'settings.pipeline.endpoint.remote' : 'settings.pipeline.endpoint.local')}
          </span>
          <Switch
            checked={remote}
            aria-label={t('settings.pipeline.endpoint.remoteModeLabel', {
              stage: t(`settings.pipeline.stages.${stage}.title`),
            })}
            onCheckedChange={(checked) =>
              onExecutionChange(
                checked
                  ? { mode: 'remote', timeout_seconds: 300, max_response_mib: 64 }
                  : { mode: 'local' },
              )
            }
          />
        </div>
      </div>
      {remote && (
        <>
          <TextField
            label={t('settings.pipeline.endpoint.baseUrl')}
            type='url'
            value={endpoint.base_url ?? ''}
            onChange={(base_url) =>
              onExecutionChange({ ...endpoint, mode: 'remote', base_url: base_url || null })
            }
          />
          {endpointIssue && (
            <p className='text-[10px] text-destructive' role='status'>
              {t(`settings.pipeline.endpoint.validation.${endpointIssue}`)}
            </p>
          )}
          <TextField
            label={t('settings.pipeline.endpoint.modelOverride')}
            value={endpoint.model ?? ''}
            onChange={(model) =>
              onExecutionChange({ ...endpoint, mode: 'remote', model: model || null })
            }
          />
          <div className='grid grid-cols-2 gap-2'>
            <NumberField
              label={t('settings.pipeline.endpoint.timeout')}
              value={endpoint.timeout_seconds}
              min={1}
              max={3600}
              step={1}
              onChange={(timeout_seconds) =>
                timeout_seconds !== null &&
                onExecutionChange({ ...endpoint, mode: 'remote', timeout_seconds })
              }
            />
            <NumberField
              label={t('settings.pipeline.endpoint.responseLimit')}
              value={endpoint.max_response_mib}
              min={1}
              max={256}
              step={1}
              onChange={(max_response_mib) =>
                max_response_mib !== null &&
                onExecutionChange({ ...endpoint, mode: 'remote', max_response_mib })
              }
            />
          </div>
          <CredentialField
            label={t(`settings.pipeline.stages.${stage}.title`)}
            value={credential}
            onChange={onCredentialChange}
          />
          <div className='flex flex-wrap items-center gap-2'>
            <Button
              type='button'
              size='sm'
              variant='outline'
              disabled={test.busy || endpointIssue !== null}
              onClick={() => void runTest()}
            >
              {t(
                test.busy
                  ? 'settings.pipeline.endpoint.testing'
                  : 'settings.pipeline.endpoint.test',
              )}
            </Button>
            <span className='text-[10px] text-muted-foreground'>
              {t('settings.pipeline.endpoint.savedCredentialHint')}
            </span>
            {test.message && (
              <span
                className={
                  test.reachable ? 'text-[10px] text-green-600' : 'text-[10px] text-destructive'
                }
                role='status'
              >
                {test.message}
              </span>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export function ModelOptions({
  model,
  onChange,
}: {
  model: PipelineModel
  onChange: (model: PipelineModel) => void
}) {
  const { t } = useTranslation()
  switch (model.model) {
    case 'koharu-layout-rfdetr-seg-2xl':
      return (
        <div className='grid grid-cols-3 gap-2'>
          <NumberField
            label={t('settings.pipeline.options.textThreshold')}
            value={model.text_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(text_threshold) => onChange({ ...model, text_threshold })}
          />
          <NumberField
            label={t('settings.pipeline.options.bubbleThreshold')}
            value={model.bubble_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(bubble_threshold) => onChange({ ...model, bubble_threshold })}
          />
          <NumberField
            label={t('settings.pipeline.options.panelThreshold')}
            value={model.panel_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(panel_threshold) => onChange({ ...model, panel_threshold })}
          />
        </div>
      )
    case 'lama':
      return (
        <LaMaOptions value={model} onChange={(changes) => onChange({ ...model, ...changes })} />
      )
    case 'mi-gan':
      return (
        <MiGanOptions value={model} onChange={(changes) => onChange({ ...model, ...changes })} />
      )
    case 'manga-inpaintor':
      return (
        <MangaInpaintorOptions
          value={model}
          onChange={(changes) => onChange({ ...model, ...changes })}
        />
      )
    case 'powerpaint':
      return (
        <PowerPaintOptions
          value={model}
          onChange={(changes) => onChange({ ...model, ...changes })}
        />
      )
    case 'flux2-klein':
      return (
        <Flux2KleinOptions
          value={model}
          onChange={(changes) => onChange({ ...model, ...changes })}
        />
      )
    case 'flux3-klein':
      return (
        <Flux3KleinOptions
          value={model}
          onChange={(changes) => onChange({ ...model, ...changes })}
        />
      )
    case 'rorem-mixed':
      return (
        <div className='grid grid-cols-2 gap-2'>
          <TextField
            label={t('settings.pipeline.options.prompt')}
            value={model.prompt ?? ''}
            onChange={(prompt) => onChange({ ...model, prompt })}
          />
          <TextField
            label={t('settings.pipeline.options.negativePrompt')}
            value={model.negative_prompt ?? ''}
            onChange={(negative_prompt) => onChange({ ...model, negative_prompt })}
          />
        </div>
      )
    case 'paddleocr-vl-1.6':
    case 'paddleocr-vl-1.6-manga-lora':
    case 'manga-ocr':
    case 'manga-ocr-cpt-flex':
    case 'baberu-ocr':
    case 'hayai-ocr':
    case 'aot-inpainting':
      return null
  }
}
