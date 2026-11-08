//! Image inference and operator controls share the application's managed runtime.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Context as _, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::DynamicImage;
use koharu_pipeline::{
    Pipeline, RemoteEndpointConfig, Stage,
    inference::{InpaintOptions, ServerConfig, ServerStatus},
};
use serde::Deserialize;
use tauri::Manager as _;

use crate::{AppState, api::authorized};

macro_rules! ensure {
    ($condition:expr, $($message:tt)*) => {
        if !$condition {
            return Err(anyhow::anyhow!($($message)*).into());
        }
    };
}

pub fn control_router() -> Router<AppState> {
    Router::new()
        .route("/server", get(status).put(configure))
        .route("/server/unload", post(unload))
        .route("/server/warmup", post(warmup))
        .route("/endpoints/test", post(probe))
}

pub fn v1_router(token: Arc<str>) -> Router<AppState> {
    Router::new()
        .route("/models", get(models))
        .route("/detect", post(detect))
        .route("/ocr", post(ocr))
        .route("/images/edits", post(inpaint))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            token,
            require_inference_token,
        ))
}

fn inference_authorized(
    token: &str,
    require_api_token: Option<bool>,
    headers: &axum::http::HeaderMap,
    uri: &axum::http::Uri,
) -> std::result::Result<(), &'static str> {
    if token.is_empty() {
        if require_api_token.unwrap_or(true) {
            return Err(
                "inference requires KOHARU_API_TOKEN; configure the token and restart the app",
            );
        }
    } else if !authorized(headers, uri, token) {
        return Err("missing or invalid API token");
    }
    Ok(())
}

async fn require_inference_token(
    State(token): State<Arc<str>>,
    request: Request,
    next: Next,
) -> Response {
    let require_api_token = koharu_config::load::<ServerConfig>("server")
        .ok()
        .and_then(|config| config.read().ok().map(|config| config.require_api_token));
    if let Err(error) =
        inference_authorized(&token, require_api_token, request.headers(), request.uri())
    {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": error})),
        )
            .into_response();
    }
    next.run(request).await
}

fn service(app: &AppState) -> Result<koharu_pipeline::inference::InferenceService> {
    app.try_state::<Pipeline>()
        .map(|pipeline| pipeline.inference())
        .context("Koharu is still initializing")
}

struct InferenceError(StatusCode, anyhow::Error);

impl From<anyhow::Error> for InferenceError {
    fn from(error: anyhow::Error) -> Self {
        let message = error.to_string().to_ascii_lowercase();
        let status = if message.contains("disabled") || message.contains("initializ") {
            StatusCode::SERVICE_UNAVAILABLE
        } else if message.contains("queue") || message.contains("busy") {
            StatusCode::TOO_MANY_REQUESTS
        } else {
            StatusCode::BAD_REQUEST
        };
        Self(status, error)
    }
}

impl IntoResponse for InferenceError {
    fn into_response(self) -> Response {
        tracing::warn!(error = %self.1, "inference request failed");
        (
            self.0,
            Json(serde_json::json!({"error": format!("{:#}", self.1)})),
        )
            .into_response()
    }
}

type InferenceResult<T> = std::result::Result<T, InferenceError>;

async fn status(State(app): State<AppState>) -> InferenceResult<Json<ServerStatus>> {
    Ok(Json(service(&app)?.status().await?))
}

async fn configure(
    State(app): State<AppState>,
    Json(config): Json<ServerConfig>,
) -> InferenceResult<Json<ServerStatus>> {
    Ok(Json(service(&app)?.configure(config).await?))
}

async fn unload(State(app): State<AppState>) -> InferenceResult<Json<ServerStatus>> {
    Ok(Json(service(&app)?.unload().await?))
}

#[derive(Deserialize)]
struct WarmupRequest {
    stage: Stage,
}

async fn warmup(
    State(app): State<AppState>,
    Json(request): Json<WarmupRequest>,
) -> InferenceResult<Json<ServerStatus>> {
    Ok(Json(service(&app)?.warmup(request.stage).await?))
}

#[derive(Deserialize)]
struct ProbeRequest {
    stage: Stage,
    endpoint: RemoteEndpointConfig,
}

async fn probe(
    Json(request): Json<ProbeRequest>,
) -> InferenceResult<Json<koharu_pipeline::remote::EndpointProbeResult>> {
    use koharu_pipeline::remote;
    let key = match request.stage {
        Stage::Detection => remote::detection_secret_key(),
        Stage::Ocr => remote::ocr_secret_key(),
        Stage::Inpainting => remote::inpainting_secret_key(),
        Stage::Translation => {
            return Err(anyhow::anyhow!("translation endpoints use the provider settings").into());
        }
    };
    Ok(Json(
        remote::probe_endpoint(request.stage, &request.endpoint, key).await?,
    ))
}

async fn models(State(app): State<AppState>) -> InferenceResult<Json<serde_json::Value>> {
    let service = service(&app)?;
    let config = service.config()?;
    ensure!(config.enabled, "inference server is disabled");
    let data = service
        .catalog()
        .into_iter()
        .map(|id| serde_json::json!({"id": id, "object": "model", "owned_by": "koharu"}))
        .collect::<Vec<_>>();
    Ok(Json(serde_json::json!({"object": "list", "data": data})))
}

#[derive(Default)]
struct Upload {
    fields: HashMap<String, String>,
    images: Vec<Vec<u8>>,
    image: Option<Vec<u8>>,
    mask: Option<Vec<u8>>,
}

impl Upload {
    async fn read(mut multipart: Multipart, config: &ServerConfig) -> Result<Self> {
        let limit = config.max_request_mib as usize * 1024 * 1024;
        let mut total = 0usize;
        let mut upload = Self::default();
        while let Some(mut field) = multipart
            .next_field()
            .await
            .context("invalid multipart body")?
        {
            let name = field
                .name()
                .context("multipart field has no name")?
                .to_owned();
            let mut bytes = Vec::new();
            while let Some(chunk) = field
                .chunk()
                .await
                .context("failed reading multipart field")?
            {
                total = total
                    .checked_add(chunk.len())
                    .context("request size overflow")?;
                ensure!(total <= limit, "request exceeds configured byte limit");
                bytes.extend_from_slice(&chunk);
            }
            match name.as_str() {
                "image" => {
                    ensure!(upload.image.is_none(), "duplicate image field");
                    upload.image = Some(bytes.to_vec());
                }
                "mask" => {
                    ensure!(upload.mask.is_none(), "duplicate mask field");
                    upload.mask = Some(bytes.to_vec());
                }
                "images" => {
                    ensure!(
                        upload.images.len() < 16,
                        "at most 16 OCR crops are accepted"
                    );
                    upload.images.push(bytes.to_vec());
                }
                _ => {
                    ensure!(bytes.len() <= 64 * 1024, "text field exceeds limit");
                    ensure!(upload.fields.len() < 32, "too many form fields");
                    let value =
                        String::from_utf8(bytes.to_vec()).context("multipart text is not UTF-8")?;
                    ensure!(
                        upload.fields.insert(name, value).is_none(),
                        "duplicate form field"
                    );
                }
            }
        }
        Ok(upload)
    }

    fn parse<T: std::str::FromStr>(&self, field: &str) -> Result<Option<T>>
    where
        T::Err: std::fmt::Display,
    {
        self.fields
            .get(field)
            .filter(|value| !value.trim().is_empty())
            .map(|value| {
                value
                    .parse()
                    .map_err(|error| anyhow::anyhow!("invalid {field}: {error}"))
            })
            .transpose()
    }

    fn model(&self) -> Option<String> {
        self.fields
            .get("model")
            .cloned()
            .filter(|model| !model.trim().is_empty())
    }
    fn decode(&self, bytes: &[u8], config: &ServerConfig) -> Result<DynamicImage> {
        koharu_pipeline::remote::decode_image(bytes, config.max_image_pixels)
    }
}

async fn detect(
    State(app): State<AppState>,
    multipart: Multipart,
) -> InferenceResult<Json<serde_json::Value>> {
    let service = service(&app)?;
    let config = service.config()?;
    ensure!(
        config.enabled && config.features.detection,
        "detection serving is disabled"
    );
    let upload = Upload::read(multipart, &config).await?;
    let bytes = upload.image.as_ref().context("image is required")?;
    let image = upload.decode(bytes, &config)?;
    let output = service
        .detect(
            upload.model(),
            image,
            upload.parse("text_threshold")?,
            upload.parse("bubble_threshold")?,
            upload.parse("panel_threshold")?,
        )
        .await?;
    let detections = output.detections.into_iter().map(|item| serde_json::json!({
        "label_id": item.label_id, "label": item.label, "score": item.score,
        "bbox": item.bbox, "area": item.area,
        "mask": {"x": item.mask.x, "y": item.mask.y, "width": item.mask.width, "height": item.mask.height, "pixels_b64": STANDARD.encode(item.mask.pixels)}
    })).collect::<Vec<_>>();
    Ok(Json(
        serde_json::json!({"image_width": output.image_width, "image_height": output.image_height, "detections": detections}),
    ))
}

async fn ocr(
    State(app): State<AppState>,
    multipart: Multipart,
) -> InferenceResult<Json<serde_json::Value>> {
    let service = service(&app)?;
    let config = service.config()?;
    ensure!(
        config.enabled && config.features.ocr,
        "OCR serving is disabled"
    );
    let upload = Upload::read(multipart, &config).await?;
    ensure!(!upload.images.is_empty(), "OCR images are required");
    let ids: Vec<String> =
        serde_json::from_str(upload.fields.get("ids").context("OCR ids are required")?)
            .context("invalid OCR ids")?;
    ensure!(
        ids.len() == upload.images.len(),
        "OCR id count does not match image count"
    );
    let mut unique = HashSet::new();
    ensure!(
        ids.iter()
            .all(|id| !id.is_empty() && id.len() <= 256 && unique.insert(id)),
        "OCR ids must be nonempty and unique"
    );
    let mut pixels = 0u64;
    let mut crops = Vec::with_capacity(ids.len());
    for (id, bytes) in ids.into_iter().zip(&upload.images) {
        let image = upload.decode(bytes, &config)?;
        pixels += u64::from(image.width()) * u64::from(image.height());
        ensure!(
            pixels <= u64::from(config.max_image_pixels),
            "OCR crop batch exceeds pixel limit"
        );
        crops.push((id, image));
    }
    let results = service
        .ocr(upload.model(), crops)
        .await?
        .into_iter()
        .map(|(id, text)| serde_json::json!({"id": id, "text": text}))
        .collect::<Vec<_>>();
    Ok(Json(serde_json::json!({"results": results})))
}

async fn inpaint(State(app): State<AppState>, multipart: Multipart) -> InferenceResult<Response> {
    let service = service(&app)?;
    let config = service.config()?;
    ensure!(
        config.enabled && config.features.inpainting,
        "inpainting serving is disabled"
    );
    let upload = Upload::read(multipart, &config).await?;
    let image = upload.decode(upload.image.as_ref().context("image is required")?, &config)?;
    let mask = koharu_pipeline::remote::decode_openai_edit_mask_png(
        upload.mask.as_ref().context("mask is required")?,
        image.width(),
        image.height(),
        config.max_image_pixels,
    )?;
    ensure!(
        upload.parse::<u32>("n")?.is_none_or(|n| n == 1),
        "only one output image is supported"
    );
    let options = InpaintOptions {
        prompt: upload.fields.get("prompt").cloned(),
        negative_prompt: upload.fields.get("negative_prompt").cloned(),
        steps: upload.parse("steps")?,
        strength: upload.parse("strength")?,
        guidance_scale: upload.parse("guidance_scale")?,
        seed: upload.parse("seed")?,
    };
    let image = service
        .inpaint(upload.model(), image, mask, options)
        .await?;
    let bytes = koharu_pipeline::remote::encode_rgba_png(&image)?;
    if upload
        .fields
        .get("response_format")
        .is_some_and(|format| format == "b64_json")
    {
        Ok(
            Json(serde_json::json!({"data": [{"b64_json": STANDARD.encode(bytes)}]}))
                .into_response(),
        )
    } else {
        Ok(([(axum::http::header::CONTENT_TYPE, "image/png")], bytes).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_options_fail_before_model_execution() {
        let mut upload = Upload::default();
        upload
            .fields
            .insert("steps".to_owned(), "not-an-integer".to_owned());
        assert!(upload.parse::<u32>("steps").is_err());
        upload.fields.insert("steps".to_owned(), "12".to_owned());
        assert_eq!(upload.parse::<u32>("steps").unwrap(), Some(12));
    }

    #[test]
    fn inference_token_policy_defaults_closed() {
        let mut headers = axum::http::HeaderMap::new();
        let uri: axum::http::Uri = "/v1/models".parse().unwrap();
        assert!(inference_authorized("", None, &headers, &uri).is_err());
        assert!(inference_authorized("", Some(true), &headers, &uri).is_err());
        assert!(inference_authorized("", Some(false), &headers, &uri).is_ok());
        assert!(inference_authorized("secret", Some(false), &headers, &uri).is_err());
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer secret".parse().unwrap(),
        );
        assert!(inference_authorized("secret", Some(true), &headers, &uri).is_ok());
    }
}
