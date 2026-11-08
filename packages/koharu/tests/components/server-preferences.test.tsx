import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { defaultModel } from '@/components/preferences/models'
import { PipelinePreferences } from '@/components/preferences/PipelinePreferences'
import { ServerPreferences } from '@/components/preferences/ServerPreferences'
import * as backend from '@/lib/backend'
import type {
  CredentialInput,
  DetectionModel,
  InpaintingModel,
  OcrModel,
  PipelineConfig,
  RemoteCredentials,
  ServerStatus,
} from '@koharu/bridge/protocol'

vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: vi.fn(async () => undefined) }))

// This repository's installed React bundle omits act, which this installed
// Testing Library version expects. Keep the shim local to this focused suite.
;(React as unknown as { act: (callback: () => unknown) => unknown }).act = (callback) => callback()

const emptyCredential = (): CredentialInput => ({
  configured: false,
  editable: true,
  environment_variable: null,
  value: null,
  clear: false,
})

const defaultPipeline: PipelineConfig = {
  detection: defaultModel('koharu-layout-rfdetr-seg-2xl') as DetectionModel,
  ocr: defaultModel('paddleocr-vl-1.6') as OcrModel,
  inpainting: defaultModel('lama') as InpaintingModel,
  translation: {
    model: {
      provider: 'local',
      model: 'gemma4-e2b-it',
      quantization: null,
      vision: true,
      reasoning: true,
    },
    generation: { vision: true, reasoning: false },
    target_language: 'en-US',
    instructions: null,
  },
  processor: {},
  remote: {
    detection: { mode: 'local' },
    ocr: { mode: 'local' },
    inpainting: { mode: 'local' },
  },
}

const defaultCredentials: RemoteCredentials = {
  detection: emptyCredential(),
  ocr: emptyCredential(),
  inpainting: emptyCredential(),
}

const mockStatus: ServerStatus = {
  config: {
    enabled: true,
    require_api_token: true,
    cpu: false,
    features: { detection: true, ocr: true, inpainting: true },
    detection: defaultModel('koharu-layout-rfdetr-seg-2xl') as DetectionModel,
    ocr: defaultModel('paddleocr-vl-1.6') as OcrModel,
    inpainting: defaultModel('lama') as InpaintingModel,
    max_request_mib: 64,
    max_image_pixels: 32000000,
    max_queue: 4,
  },
  active: null,
  loaded: 'detection',
  queued: 0,
  completed: 12,
  failed: 1,
  last_error: null,
}

describe('Server and Remote Endpoint Preferences', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(backend, 'getServer').mockResolvedValue(mockStatus)
    vi.spyOn(backend, 'updateServer').mockImplementation(async (cfg) => ({
      ...mockStatus,
      config: cfg,
    }))
    vi.spyOn(backend, 'unloadServer').mockResolvedValue({
      ...mockStatus,
      loaded: null,
    })
    vi.spyOn(backend, 'warmupServer').mockImplementation(async (stage) => ({
      ...mockStatus,
      loaded: stage,
    }))
    vi.spyOn(backend, 'testEndpoint').mockResolvedValue({
      reachable: true,
      models: ['lama'],
      message: 'Connection successful',
    })
  })

  it('renders server status and controls correctly', async () => {
    render(<ServerPreferences />)

    await waitFor(() => {
      expect(screen.getByText('Server status')).toBeDefined()
    })

    // Active stage status and loaded model
    expect(screen.getByText('Idle')).toBeDefined()
    expect(screen.getByText('Loaded model: detection')).toBeDefined()
    expect(screen.getByText('Queued: 0 · Completed: 12 · Failed: 1')).toBeDefined()

    // Test unload action
    const unloadBtn = screen.getByRole('button', { name: 'Unload models' })
    fireEvent.click(unloadBtn)
    await waitFor(() => {
      expect(backend.unloadServer).toHaveBeenCalledTimes(1)
    })

    // Test apply settings button
    const applyBtn = screen.getByRole('button', { name: 'Apply settings' })
    fireEvent.click(applyBtn)
    await waitFor(() => {
      expect(backend.updateServer).toHaveBeenCalledTimes(1)
    })
  })

  it('shows initial load errors and keeps Apply disabled until status loads', async () => {
    vi.spyOn(backend, 'getServer').mockRejectedValueOnce(new Error('status unavailable'))
    render(<ServerPreferences />)

    expect(await screen.findByRole('alert')).toHaveTextContent('status unavailable')
    expect(screen.getByRole('button', { name: 'Apply settings' })).toBeDisabled()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Apply settings' })).toBeEnabled()
    })
  })

  it('shows Apply errors and retains draft during status refresh', async () => {
    vi.spyOn(backend, 'updateServer').mockRejectedValueOnce(new Error('apply rejected'))
    const getServerMock = vi.spyOn(backend, 'getServer')
    render(<ServerPreferences />)
    await screen.findByText('Server status')

    fireEvent.click(screen.getByRole('switch', { name: 'Accept remote inference' }))
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => {
      expect(getServerMock.mock.calls.length).toBeGreaterThanOrEqual(2)
    })
    expect(screen.getByRole('switch', { name: 'Accept remote inference' })).toBeChecked()

    fireEvent.click(screen.getByRole('button', { name: 'Apply settings' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('apply rejected')
    expect(screen.getByRole('switch', { name: 'Accept remote inference' })).toBeChecked()
  })

  it('toggles remote mode and runs connection test in PipelinePreferences', async () => {
    vi.spyOn(backend, 'testEndpoint').mockResolvedValue({
      reachable: true,
      models: ['lama'],
      message: 'Connection successful',
    })
    const onChange = vi.fn()
    const onCredentialsChange = vi.fn()

    render(
      <PipelinePreferences
        value={{
          ...defaultPipeline,
          remote: {
            detection: {
              mode: 'remote',
              base_url: 'http://localhost:8000/v1',
              model: 'custom-det',
              timeout_seconds: 120,
              max_response_mib: 32,
            },
            ocr: { mode: 'local' },
            inpainting: { mode: 'local' },
          },
        }}
        onChange={onChange}
        credentials={defaultCredentials}
        onCredentialsChange={onCredentialsChange}
      />,
    )

    // Test connection button should be visible for remote stage
    const testBtn = await screen.findByRole('button', { name: 'Test connection' })
    fireEvent.click(testBtn)

    await waitFor(() => {
      expect(backend.testEndpoint).toHaveBeenCalledWith('detection', {
        mode: 'remote',
        base_url: 'http://localhost:8000/v1',
        model: 'custom-det',
        timeout_seconds: 120,
        max_response_mib: 32,
      })
      expect(screen.getByText('Connection successful')).toBeDefined()
    })
  })
})
