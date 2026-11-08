import type { PipelineConfig, StageExecution } from '@koharu/bridge/protocol'

export type RemoteEndpointIssue = 'missingUrl' | 'invalidUrl' | 'invalidLimits'

export function remoteEndpointIssue(execution: StageExecution): RemoteEndpointIssue | null {
  if (execution.mode !== 'remote') return null
  const { base_url, timeout_seconds, max_response_mib } = execution
  if (!base_url?.trim()) return 'missingUrl'

  let url: URL
  try {
    url = new URL(base_url)
  } catch {
    return 'invalidUrl'
  }
  if (
    (url.protocol !== 'http:' && url.protocol !== 'https:') ||
    url.username.length > 0 ||
    url.password.length > 0 ||
    url.search.length > 0 ||
    url.hash.length > 0
  ) {
    return 'invalidUrl'
  }
  if (
    !Number.isInteger(timeout_seconds) ||
    timeout_seconds < 1 ||
    timeout_seconds > 3600 ||
    !Number.isInteger(max_response_mib) ||
    max_response_mib < 1 ||
    max_response_mib > 256
  ) {
    return 'invalidLimits'
  }
  return null
}

export function invalidRemoteStage(
  pipeline: PipelineConfig,
): 'detection' | 'ocr' | 'inpainting' | null {
  for (const stage of ['detection', 'ocr', 'inpainting'] as const) {
    const execution = pipeline.remote?.[stage]
    if (execution && remoteEndpointIssue(execution)) return stage
  }
  return null
}
