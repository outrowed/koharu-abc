# Koharu 원격/Docker 배포 가이드

Koharu는 `koharu-rpc`의 HTTP API(`/api/v1/*`)와 정적 UI(`/`)를 같은 포트에서 제공합니다. 기본 desktop 실행은 `127.0.0.1:47823`을 사용하며, 원격 접근을 위해 `0.0.0.0`에 bind하면 `KOHARU_API_TOKEN`이 필수입니다.

## Docker image 빌드

```bash
docker build -t koharu:latest .
```

API token과 provider key를 `ARG` 또는 Dockerfile `ENV`로 넣지 마세요. Build layer와 image metadata에 secret이 남을 수 있습니다. 모든 secret은 container를 시작할 때 주입합니다.

## Runtime 환경변수

`.env.example`을 복사한 뒤 필요한 값만 채웁니다.

```bash
cp .env.example .env
openssl rand -hex 32
# 출력값을 .env의 KOHARU_API_TOKEN에 입력
```

image는 `KOHARU_SECRETS=environment`로 실행되어 provider credential을 환경변수로만 읽습니다(Docker 기본 seccomp profile이 keyring syscall을 막기 때문). Settings는 설정 여부와 변수명을 표시하지만 값을 변경하거나 삭제하지 않습니다. 값을 바꾸면 container를 재시작하거나 Vast.ai instance를 재배포해야 합니다. 이 변수가 없는 데스크톱 실행은 OS credential store(Linux는 Secret Service: GNOME Keyring 또는 KWallet)에 저장하고 Settings에서 입력할 수 있으며, 아래 환경변수가 설정되어 있으면 그 값이 우선합니다.

| Provider | Environment variable |
|---|---|
| Atlas Cloud | `ATLASCLOUD_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| Grok | `XAI_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI-compatible | `OPENAI_COMPATIBLE_API_KEY` (optional) |
| OpenRouter | `OPENROUTER_API_KEY` |
| LM Studio | `LM_STUDIO_API_TOKEN` (optional) |
| DeepL | `DEEPL_API_KEY` |
| Google Cloud Translation | `GOOGLE_CLOUD_API_KEY` |
| Caiyun | `CAIYUN_API_KEY` |

`KOHARU_SECRETS=environment`에서는 writable credential store가 없으므로 Codex OAuth login/logout을 지원하지 않습니다. Translation provider와 local model serving에는 영향이 없습니다.

## 로컬 Docker 실행

```bash
docker run -d \
  --name koharu \
  --env-file .env \
  -p 47823:47823 \
  koharu:latest
```

CEF sandbox는 image에서 비활성화되어 있으므로 `--cap-add=SYS_ADMIN`, `--privileged`, `--security-opt seccomp=unconfined`이 필요하지 않습니다.

브라우저에서 다음 URL을 엽니다.

```text
http://127.0.0.1:47823/?token=<KOHARU_API_TOKEN>
```

Token이 설정된 상태에서 bare `/`는 401이 정상입니다. Query token은 초기 HTML에 runtime으로 주입되며 image나 frontend bundle에 포함되지 않습니다.

## Vast.ai custom image

1. Docker Hub/GHCR에 push한 Koharu image tag를 template의 image로 지정합니다.
2. Launch mode는 image의 `ENTRYPOINT`를 보존하는 **Entrypoint/args** 방식을 사용합니다.
3. Template environment에 `KOHARU_API_TOKEN`과 실제로 사용할 provider 변수만 추가합니다.
4. Internal TCP port `47823`을 엽니다.
5. Instance의 **IP Port Info**에서 `47823/tcp`에 배정된 외부 host/port를 확인합니다.
6. 다음 주소로 접속합니다.

```text
http://<PUBLIC_IP>:<MAPPED_PORT>/?token=<KOHARU_API_TOKEN>
```

Vast.ai template의 Docker Options가 임의의 security option을 허용하지 않아도 이 image는 기본 container 제한으로 동작합니다. 현재 Dockerfile은 Vast.ai base image의 entrypoint를 대체하므로 base image가 제공하던 SSH/Jupyter/instance portal은 자동으로 시작되지 않습니다.

## 검증

```bash
# 인증 없이 API와 root가 거부되어야 함
curl -sS -o /dev/null -w '%{http_code}\n' http://<host>:<port>/api/v1/meta
curl -sS -o /dev/null -w '%{http_code}\n' http://<host>:<port>/

# 인증된 API와 UI가 성공해야 함
curl -fsS -H "Authorization: Bearer <token>" http://<host>:<port>/api/v1/meta
curl -fsS "http://<host>:<port>/?token=<token>" > /dev/null
curl -fsS -H "Authorization: Bearer <token>" http://<host>:<port>/api/v1/config
```

Expected status는 unauthenticated 요청 401, authenticated 요청 200입니다. `/api/v1/config`에는 credential 값이 아니라 `configured`, `editable`, `environment_variable` 상태만 반환됩니다.

## 알려진 제약

- App은 headless container에서도 CEF를 Xvfb 위에서 초기화합니다.
- `KOHARU_RPC_PORT`를 변경하면 desktop CEF의 initial navigation과 맞도록 `crates/koharu/tauri.conf.json`의 `devUrl`도 build 전에 변경해야 합니다.
- Provider model discovery는 외부 API에 연결하므로 key가 유효하지 않거나 provider가 unreachable이면 해당 model 목록이 비어 있을 수 있습니다.
- `.env`는 Git에서 ignore됩니다. `.env.example`에 실제 token이나 provider key를 넣지 마세요.


## Integrated inference server

Status: implemented; use Settings > Server to configure the inference service.
The website/editor and inference API share the existing Koharu listener. Enabling
inference does not change its bind address or port, restart the app, or disable
the editor. The service is disabled by default.

The Server page configures enabled features, the selected model and its options
for each feature, CPU preference, request/image/queue limits, and inference token
requirements. Apply settings explicitly. Refresh reads current state without
replacing unsaved settings; warm-up loads the selected model, and unload releases
model memory. The inference worker and editor processing share one admission
lane so they cannot race for the same model memory. Different stages evict the
previous stage's models; repeated requests to the same stage reuse loaded models.

- `GET /api/v1/server`: configuration, active/loaded stage, queue and counters.
- `PUT /api/v1/server`: persist and apply serving configuration.
- `POST /api/v1/server/warmup`: JSON `{ "stage": "ocr" }` (or detection/inpainting).
- `POST /api/v1/server/unload`: unload worker models.
- `GET /v1/models`: configured, enabled inference models.
- `POST /v1/detect`: multipart `image`, optional `model` and thresholds.
- `POST /v1/ocr`: multipart `images` (up to 16 crops), `ids` (JSON string array),
  optional `model`; results carry matching request-local IDs.
- `POST /v1/images/edits`: multipart `image`, `mask`, optional model and generation
  settings. RGBA masks use transparent pixels for inpainting; grayscale masks use
  white pixels. `response_format=b64_json` returns the encoded PNG, otherwise the
  response is a PNG with the same dimensions.

`specs/remote-service-openapi.json` defines the inference wire formats. The server
serves exactly the model selected for each enabled feature, rather than accepting
arbitrary checkpoint paths from inference callers. Unsupported model IDs fail
explicitly. Warm-up can download the selected model and native runtime packages.

A configured `KOHARU_API_TOKEN` is required for both the UI/admin API and inference
API. On loopback-only deployments with no token, inference is still rejected by
default; the operator must explicitly turn off Require API token to allow it.
Keep that exception loopback-only. TLS and external network exposure belong to
the deployment reverse proxy, not an inference request.

## Per-feature remote execution

Settings > Pipeline selects Local or Remote separately for detection, OCR, and
inpainting. Each remote stage has its own base URL, optional model ID, credential,
timeout, and response-size limit. For another Koharu instance, use its origin or
`/v1` base URL and the model ID from its `/v1/models` catalog. The Test connection
button probes that catalog with the saved credential; it does not run inference.
Translation continues to use Settings > Providers and Settings > Translation,
including its independent OpenAI-compatible URL, key, and model selection.

The application backend sends inference requests, not the browser. Endpoint URLs
must therefore be reachable from the Koharu server process. Project storage,
scene updates, crop/tile preparation, and compositing remain on that instance.
Remote inpainting sends sequential tiles; it is not a whole-project remote job.
Image URLs returned by an image-edit service must be same-origin; use binary PNG
or `b64_json` for third-party storage outputs. Response bodies and decoded images
are bounded, requests have explicit timeouts, and redirects are rejected.

Desktop credentials remain in the OS credential store. A headless container can
opt into GUI-editable file storage with `KOHARU_SECRETS=file` and
`KOHARU_CREDENTIAL_DIR` pointing to a persistent private directory. This explicit
container mode stores credentials in plaintext with directory mode 0700 and file
mode 0600; do not share that mount. `KOHARU_SECRETS=environment` remains read-only.
Injected nonblank environment credentials always take precedence and are shown
as read-only in settings, even when a writable store is otherwise selected.
