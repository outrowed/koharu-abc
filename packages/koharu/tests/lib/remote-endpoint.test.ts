import { describe, expect, it } from 'vitest'

import { invalidRemoteStage, remoteEndpointIssue } from '@/lib/remote-endpoint'
import type { PipelineConfig, StageExecution } from '@koharu/bridge/protocol'

const valid = (
  base_url = 'https://inference.example/v1',
): Extract<StageExecution, { mode: 'remote' }> => ({
  mode: 'remote',
  base_url,
  model: null,
  timeout_seconds: 300,
  max_response_mib: 64,
})

describe('remote endpoint validation', () => {
  it('distinguishes incomplete URLs from invalid URLs', () => {
    expect(remoteEndpointIssue(valid(''))).toBe('missingUrl')
    expect(remoteEndpointIssue(valid('not a url'))).toBe('invalidUrl')
  })

  it('rejects unsupported URL components and limit values', () => {
    expect(remoteEndpointIssue(valid('ftp://inference.example/v1'))).toBe('invalidUrl')
    expect(remoteEndpointIssue(valid('https://user:pass@inference.example/v1'))).toBe('invalidUrl')
    expect(remoteEndpointIssue(valid('https://inference.example/v1?token=no'))).toBe('invalidUrl')
    expect(remoteEndpointIssue(valid('https://inference.example/v1#fragment'))).toBe('invalidUrl')
    expect(
      remoteEndpointIssue({
        ...valid(),
        timeout_seconds: 0,
      }),
    ).toBe('invalidLimits')
  })

  it('identifies the first incomplete stage and permits complete settings', () => {
    const pipeline = {
      remote: {
        detection: valid(''),
        ocr: valid(),
        inpainting: { mode: 'local' },
      },
    } as unknown as PipelineConfig
    expect(invalidRemoteStage(pipeline)).toBe('detection')

    pipeline.remote!.detection = valid()
    expect(invalidRemoteStage(pipeline)).toBeNull()
  })
})
