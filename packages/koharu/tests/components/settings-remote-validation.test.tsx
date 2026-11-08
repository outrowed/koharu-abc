import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { defaultModel } from '@/components/preferences/models'
import { SettingsPage } from '@/components/preferences/SettingsPage'
import * as backend from '@/lib/backend'
import { receivePreferences, useKoharuStore } from '@/lib/store'
import type {
  CredentialInput,
  DetectionModel,
  InpaintingModel,
  OcrModel,
  Preferences,
} from '@koharu/bridge/protocol'

vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: vi.fn(async () => undefined) }))

// React 19 testing shim
;(React as unknown as { act: (callback: () => unknown) => unknown }).act = (callback) => callback()

const emptyCredential = (): CredentialInput => ({
  configured: false,
  editable: true,
  environment_variable: null,
  value: null,
  clear: false,
})

const basePreferences: Preferences = {
  pipeline: {
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
  },
  providers: { entries: [] },
  typesetting: { font_families: [] },
  remote_credentials: {
    detection: emptyCredential(),
    ocr: emptyCredential(),
    inpainting: emptyCredential(),
  },
  languages: [],
}

describe('SettingsPage remote validation and autosave gate', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(backend, 'refreshPreferences').mockResolvedValue(undefined)
    vi.spyOn(backend, 'refreshTranslationModels').mockResolvedValue(undefined)
    useKoharuStore.setState({ settingsOpen: true })
    receivePreferences(basePreferences)
  })

  it('does not send autosave while remote URL is missing or partial, blocks navigation, and autosaves on complete URL', async () => {
    const saveMock = vi.spyOn(backend, 'savePreferences').mockResolvedValue(basePreferences)

    render(<SettingsPage />)

    // Navigate to Pipeline tab
    const pipelineBtn = await screen.findByRole('button', { name: 'Pipeline' })
    fireEvent.click(pipelineBtn)
    expect(await screen.findByText('Processing')).toBeDefined()

    // Toggle Detection to remote mode (initially has null base_url)
    const remoteToggle = screen.getByRole('switch', { name: 'Detection execution mode' })
    fireEvent.click(remoteToggle)

    // Inline field validation hint appears
    expect(
      await screen.findByText('Enter a valid base URL (e.g. http://host:8000/v1).'),
    ).toBeDefined()

    // Wait for autosave debounce timer to ensure no autosave was triggered
    await new Promise((resolve) => setTimeout(resolve, 350))
    expect(saveMock).not.toHaveBeenCalled()

    // Try leaving to editor via Back button while incomplete
    fireEvent.click(screen.getByRole('button', { name: 'Back to editor' }))
    expect(
      await screen.findByText(
        'Detection remote endpoint configuration is incomplete. Check the base URL.',
      ),
    ).toBeDefined()
    expect(useKoharuStore.getState().settingsOpen).toBe(true)
    expect(saveMock).not.toHaveBeenCalled()

    // Enter a partial / invalid URL
    const urlInput = screen.getByLabelText('Base URL')
    fireEvent.change(urlInput, { target: { value: 'invalid-url' } })

    expect(
      await screen.findByText(
        'URL must use http:// or https:// and cannot contain queries or credentials.',
      ),
    ).toBeDefined()
    await new Promise((resolve) => setTimeout(resolve, 350))
    expect(saveMock).not.toHaveBeenCalled()

    // Enter a valid URL
    fireEvent.change(urlInput, { target: { value: 'http://localhost:8000/v1' } })

    await waitFor(
      () => {
        expect(saveMock).toHaveBeenCalled()
      },
      { timeout: 1000 },
    )

    const savedPipeline = saveMock.mock.calls[0]![0]
    expect(savedPipeline.remote?.detection).toMatchObject({
      mode: 'remote',
      base_url: 'http://localhost:8000/v1',
    })
  })
})
