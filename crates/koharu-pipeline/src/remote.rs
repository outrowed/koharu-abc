use std::{io::Cursor, time::Duration};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{
    DynamicImage, ExtendedColorType, GrayImage, ImageEncoder as _, ImageFormat, Rgba, RgbaImage,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use koharu_ml::koharu_layout_rfdetr_seg_2xl::{
    KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask,
};
use koharu_secrets::{ExposeSecret, SecretKey};
use reqwest::{
    Client,
    header::CONTENT_TYPE,
    multipart::{Form, Part},
};
use serde::{Deserialize, Serialize};
use specta::Type;
use url::Url;

pub const DETECTION_API_KEY_ENV: &str = "KOHARU_DETECTION_API_KEY";
pub const OCR_API_KEY_ENV: &str = "KOHARU_OCR_API_KEY";
pub const INPAINTING_API_KEY_ENV: &str = "KOHARU_INPAINTING_API_KEY";

pub const fn detection_secret_key() -> SecretKey<'static> {
    SecretKey::environment("pipeline-detection", DETECTION_API_KEY_ENV)
}

pub const fn ocr_secret_key() -> SecretKey<'static> {
    SecretKey::environment("pipeline-ocr", OCR_API_KEY_ENV)
}

pub const fn inpainting_secret_key() -> SecretKey<'static> {
    SecretKey::environment("pipeline-inpainting", INPAINTING_API_KEY_ENV)
}

/// Where a pipeline vision stage (Detection, OCR, or Inpainting) executes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum StageExecution {
    #[default]
    Local,
    Remote(RemoteEndpointConfig),
}

impl StageExecution {
    #[must_use]
    pub const fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }

    pub fn validate(&self, stage: &str) -> Result<()> {
        if let Self::Remote(config) = self {
            config.validate(stage)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct RemoteEndpointConfig {
    pub base_url: Option<Url>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub timeout_seconds: u32,
    pub max_response_mib: u32,
}

impl Default for RemoteEndpointConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            model: None,
            timeout_seconds: 300,
            max_response_mib: 64,
        }
    }
}

pub fn http_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("failed to build inference HTTP client")
}

impl RemoteEndpointConfig {
    pub fn validate(&self, stage: &str) -> Result<()> {
        ensure!(
            (1..=3600).contains(&self.timeout_seconds),
            "{stage} timeout must be between 1 and 3600 seconds"
        );
        ensure!(
            (1..=256).contains(&self.max_response_mib),
            "{stage} response limit must be between 1 and 256 MiB"
        );
        let url = self
            .base_url
            .as_ref()
            .ok_or_else(|| anyhow!("{stage} remote execution requires a base URL"))?;
        ensure!(
            matches!(url.scheme(), "http" | "https"),
            "{stage} remote base URL must use http or https"
        );
        ensure!(
            url.username().is_empty() && url.password().is_none(),
            "{stage} remote base URL must not contain embedded user credentials"
        );
        ensure!(
            url.query().is_none(),
            "{stage} remote base URL must not contain query parameters"
        );
        ensure!(
            url.fragment().is_none(),
            "{stage} remote base URL must not contain a URL fragment"
        );
        if let Some(model) = &self.model {
            ensure!(
                !model.contains('\0'),
                "{stage} remote model identifier contains NUL"
            );
        }
        Ok(())
    }

    #[must_use]
    pub fn effective_model<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(fallback)
    }
}

/// Builds an endpoint URL from a configured base URL and a versioned suffix
/// (such as `"v1/images/edits"`, `"v1/detect"`, or `"v1/ocr"`).
///
/// If the user's `base_url` already ends in `/v1`, the duplicate `v1/` prefix
/// on `suffix` is stripped so both `http://host:8000` and `http://host:8000/v1`
/// resolve to `http://host:8000/v1/...`. If `base_url` already ends with the
/// exact endpoint path, it is used as-is.
pub fn resolve_endpoint(base_url: &Url, suffix: &str) -> String {
    let base = base_url.as_str().trim_end_matches('/');
    let clean_suffix = suffix.trim_start_matches('/');
    let without_v1 = clean_suffix.strip_prefix("v1/").unwrap_or(clean_suffix);

    if base.ends_with(&format!("/{clean_suffix}")) || base.ends_with(&format!("/{without_v1}")) {
        return base.to_owned();
    }
    if base.ends_with("/v1") {
        return format!("{base}/{without_v1}");
    }
    format!("{base}/{clean_suffix}")
}

pub struct RemoteInpaintOptions<'a> {
    pub model: &'a str,
    pub prompt: Option<&'a str>,
    pub negative_prompt: Option<&'a str>,
    pub steps: Option<u32>,
    pub strength: Option<f64>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<i64>,
}

/// Calls an OpenAI-compatible `POST /v1/images/edits` endpoint for both
/// generative and promptless (e.g. LaMa) inpainting.
///
/// The mask is encoded as an RGBA PNG of the same dimensions as `image` where
/// fully transparent pixels (`alpha = 0`) mark the region to inpaint and opaque
/// pixels (`alpha = 255`) mark the region to preserve, matching the OpenAI
/// Images Edit specification while keeping the grayscale mask in the RGB
/// channels for servers that inspect luminance.
pub async fn inpaint_via_edits(
    client: &Client,
    endpoint_config: &RemoteEndpointConfig,
    secret_key: SecretKey<'_>,
    image: &DynamicImage,
    mask: &GrayImage,
    options: &RemoteInpaintOptions<'_>,
) -> Result<DynamicImage> {
    let base_url = endpoint_config
        .base_url
        .as_ref()
        .context("inpainting remote endpoint has no base URL")?;
    let url = resolve_endpoint(base_url, "v1/images/edits");
    let model = endpoint_config.effective_model(options.model);

    let image_png = encode_rgba_png(image)?;
    let mask_png = encode_openai_edit_mask_png(mask)?;
    let (width, height) = (image.width(), image.height());

    let image_part = Part::bytes(image_png)
        .file_name("image.png")
        .mime_str("image/png")?;
    let mask_part = Part::bytes(mask_png)
        .file_name("mask.png")
        .mime_str("image/png")?;

    let mut form = Form::new()
        .part("image", image_part)
        .part("mask", mask_part)
        .text("model", model.to_owned())
        .text("n", "1")
        .text("size", format!("{width}x{height}"))
        .text("response_format", "b64_json");

    if let Some(prompt) = options.prompt.map(str::trim).filter(|p| !p.is_empty()) {
        form = form.text("prompt", prompt.to_owned());
    }
    if let Some(negative_prompt) = options
        .negative_prompt
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        form = form.text("negative_prompt", negative_prompt.to_owned());
    }
    if let Some(steps) = options.steps {
        form = form.text("steps", steps.to_string());
    }
    if let Some(strength) = options.strength {
        form = form.text("strength", strength.to_string());
    }
    if let Some(guidance_scale) = options.guidance_scale {
        form = form.text("guidance_scale", guidance_scale.to_string());
    }
    if let Some(seed) = options.seed {
        form = form.text("seed", seed.to_string());
    }

    let mut request = client
        .post(&url)
        .timeout(Duration::from_secs(endpoint_config.timeout_seconds.into()))
        .multipart(form);
    if let Some(api_key) = koharu_secrets::get(secret_key)? {
        request = request.bearer_auth(api_key.expose_secret());
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("remote inpainting request to {url} failed"))?;
    let status = response.status();
    if !status.is_success() {
        let body = String::from_utf8_lossy(&bounded_body(response, 8192).await.unwrap_or_default())
            .into_owned();
        bail!(
            "remote inpainting endpoint returned HTTP {}: {}",
            status.as_u16(),
            truncate_error(&body)
        );
    }

    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    let bytes = bounded_body(
        response,
        endpoint_config.max_response_mib as usize * 1024 * 1024,
    )
    .await
    .context("failed to read remote inpainting response body")?;

    if content_type.starts_with("image/") {
        return decode_image(&bytes, 64_000_000)
            .context("failed to decode direct image response from remote inpainting endpoint");
    }

    let parsed: ImagesEditResponse = serde_json::from_slice(&bytes)
        .context("failed to decode JSON response from remote inpainting endpoint")?;
    let item = parsed
        .data
        .into_iter()
        .next()
        .context("remote inpainting response contained no image items")?;

    if let Some(b64) = item.b64_json {
        let raw =
            decode_base64(&b64).context("invalid base64 image in remote inpainting response")?;
        return decode_image(&raw, 64_000_000)
            .context("failed to decode base64 image from remote inpainting response");
    }
    if let Some(image_url) = item.url {
        let parsed_url = Url::parse(&image_url).with_context(|| {
            format!("invalid image URL in remote inpainting response: {image_url}")
        })?;
        ensure!(
            matches!(parsed_url.scheme(), "http" | "https"),
            "inpainting image URL must use HTTP or HTTPS"
        );
        ensure!(
            parsed_url.username().is_empty() && parsed_url.password().is_none(),
            "inpainting image URL must not contain credentials"
        );
        let same_origin = base_url.origin() == parsed_url.origin();
        ensure!(
            same_origin,
            "cross-origin inpainting image URLs are not accepted; return image bytes or b64_json instead"
        );
        let mut fetch = client
            .get(parsed_url)
            .timeout(Duration::from_secs(endpoint_config.timeout_seconds.into()));
        if same_origin {
            if let Some(api_key) = koharu_secrets::get(secret_key)? {
                fetch = fetch.bearer_auth(api_key.expose_secret());
            }
        } else {
            tracing::debug!(url = %image_url, "fetching inpainted image from cross-origin location without bearer auth");
        }
        let fetched = fetch
            .send()
            .await
            .with_context(|| format!("failed to fetch inpainted image from {image_url}"))?;
        let fetch_status = fetched.status();
        if !fetch_status.is_success() {
            bail!(
                "fetching inpainted image from {} returned HTTP {}",
                image_url,
                fetch_status.as_u16()
            );
        }
        let raw = bounded_body(
            fetched,
            endpoint_config.max_response_mib as usize * 1024 * 1024,
        )
        .await?;
        return decode_image(&raw, 64_000_000)
            .context("failed to decode fetched image from remote inpainting URL");
    }

    bail!("remote inpainting response item contained neither b64_json nor url")
}

/// Calls a FastAPI/OpenAPI microservice `POST /v1/detect` endpoint.
#[allow(clippy::too_many_arguments)]
pub async fn detect_remote(
    client: &Client,
    endpoint_config: &RemoteEndpointConfig,
    secret_key: SecretKey<'_>,
    default_model: &str,
    image: &DynamicImage,
    text_threshold: Option<f32>,
    bubble_threshold: Option<f32>,
    panel_threshold: Option<f32>,
) -> Result<KoharuLayoutDetections> {
    let base_url = endpoint_config
        .base_url
        .as_ref()
        .context("detection remote endpoint has no base URL")?;
    let url = resolve_endpoint(base_url, "v1/detect");
    let model = endpoint_config.effective_model(default_model);

    let image_png = encode_rgb_png(image)?;
    let image_part = Part::bytes(image_png)
        .file_name("image.png")
        .mime_str("image/png")?;

    let mut form = Form::new()
        .part("image", image_part)
        .text("model", model.to_owned());
    if let Some(threshold) = text_threshold {
        form = form.text("text_threshold", threshold.to_string());
    }
    if let Some(threshold) = bubble_threshold {
        form = form.text("bubble_threshold", threshold.to_string());
    }
    if let Some(threshold) = panel_threshold {
        form = form.text("panel_threshold", threshold.to_string());
    }

    let mut request = client
        .post(&url)
        .timeout(Duration::from_secs(endpoint_config.timeout_seconds.into()))
        .multipart(form);
    if let Some(api_key) = koharu_secrets::get(secret_key)? {
        request = request.bearer_auth(api_key.expose_secret());
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("remote detection request to {url} failed"))?;
    let status = response.status();
    let raw = bounded_body(
        response,
        endpoint_config.max_response_mib as usize * 1024 * 1024,
    )
    .await
    .context("failed to read remote detection response")?;
    let text = String::from_utf8(raw).context("detection response is not UTF-8")?;
    if !status.is_success() {
        bail!(
            "remote detection endpoint returned HTTP {}: {}",
            status.as_u16(),
            truncate_error(&text)
        );
    }

    let wire: WireDetectionsResponse = serde_json::from_str(&text)
        .context("failed to decode JSON response from remote detection endpoint")?;
    wire.into_detections(image.width(), image.height())
}

pub struct RemoteOcrRegion {
    pub id: String,
    pub image: DynamicImage,
}

/// Calls a FastAPI/OpenAPI microservice `POST /v1/ocr` endpoint with a batch
/// of cropped text regions.
pub async fn ocr_remote(
    client: &Client,
    endpoint_config: &RemoteEndpointConfig,
    secret_key: SecretKey<'_>,
    default_model: &str,
    regions: &[RemoteOcrRegion],
) -> Result<Vec<String>> {
    if regions.is_empty() {
        return Ok(Vec::new());
    }
    let base_url = endpoint_config
        .base_url
        .as_ref()
        .context("OCR remote endpoint has no base URL")?;
    let url = resolve_endpoint(base_url, "v1/ocr");
    let model = endpoint_config.effective_model(default_model);

    let ids = regions
        .iter()
        .map(|region| region.id.clone())
        .collect::<Vec<_>>();
    let mut form = Form::new()
        .text("model", model.to_owned())
        .text("ids", serde_json::to_string(&ids)?);

    for (index, region) in regions.iter().enumerate() {
        let png = encode_rgb_png(&region.image)?;
        let part = Part::bytes(png)
            .file_name(format!("crop_{index}.png"))
            .mime_str("image/png")?;
        form = form.part("images", part);
    }

    let mut request = client
        .post(&url)
        .timeout(Duration::from_secs(endpoint_config.timeout_seconds.into()))
        .multipart(form);
    if let Some(api_key) = koharu_secrets::get(secret_key)? {
        request = request.bearer_auth(api_key.expose_secret());
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("remote OCR request to {url} failed"))?;
    let status = response.status();
    let raw = bounded_body(
        response,
        endpoint_config.max_response_mib as usize * 1024 * 1024,
    )
    .await
    .context("failed to read remote OCR response")?;
    let body = String::from_utf8(raw).context("OCR response is not UTF-8")?;
    if !status.is_success() {
        bail!(
            "remote OCR endpoint returned HTTP {}: {}",
            status.as_u16(),
            truncate_error(&body)
        );
    }

    let parsed: WireOcrResponse = serde_json::from_str(&body)
        .context("failed to decode JSON response from remote OCR endpoint")?;
    parsed.resolve(&ids)
}

#[derive(Deserialize)]
struct ImagesEditResponse {
    data: Vec<ImagesEditItem>,
}

#[derive(Deserialize)]
struct ImagesEditItem {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

#[derive(Deserialize)]
struct WireDetectionsResponse {
    #[serde(default)]
    image_width: Option<u32>,
    #[serde(default)]
    image_height: Option<u32>,
    detections: Vec<WireDetection>,
}

#[derive(Deserialize)]
struct WireDetection {
    #[serde(default)]
    label_id: Option<usize>,
    label: String,
    score: f32,
    bbox: [f32; 4],
    #[serde(default)]
    area: Option<u32>,
    #[serde(default)]
    mask: Option<WireMask>,
}

#[derive(Deserialize)]
struct WireMask {
    #[serde(default)]
    x: u32,
    #[serde(default)]
    y: u32,
    width: u32,
    height: u32,
    #[serde(default)]
    pixels: Option<Vec<u8>>,
    #[serde(default)]
    pixels_b64: Option<String>,
    #[serde(default)]
    png_b64: Option<String>,
}

impl WireDetectionsResponse {
    fn into_detections(
        self,
        fallback_width: u32,
        fallback_height: u32,
    ) -> Result<KoharuLayoutDetections> {
        let image_width = self.image_width.unwrap_or(fallback_width);
        let image_height = self.image_height.unwrap_or(fallback_height);
        ensure!(
            image_width > 0 && image_height > 0,
            "detection image dimensions must be non-zero"
        );
        ensure!(
            image_width == fallback_width && image_height == fallback_height,
            "detection dimensions differ from uploaded image"
        );
        ensure!(self.detections.len() <= 4096, "too many detections");
        let mut detections = Vec::with_capacity(self.detections.len());
        for (index, item) in self.detections.into_iter().enumerate() {
            ensure!(
                item.score.is_finite() && (0.0..=1.0).contains(&item.score),
                "detection {index} has invalid confidence score {}",
                item.score
            );
            ensure!(
                item.bbox.iter().all(|coord| coord.is_finite()),
                "detection {index} has non-finite bounding box coordinates"
            );
            let mask = item
                .mask
                .ok_or_else(|| {
                    anyhow!("detection {index} is missing an instance segmentation mask")
                })?
                .into_mask()
                .with_context(|| format!("invalid mask on detection {index}"))?;
            ensure!(
                mask.x
                    .checked_add(mask.width)
                    .is_some_and(|right| right <= image_width)
                    && mask
                        .y
                        .checked_add(mask.height)
                        .is_some_and(|bottom| bottom <= image_height),
                "detection mask exceeds uploaded image"
            );
            ensure!(
                item.bbox[0] >= 0.0
                    && item.bbox[1] >= 0.0
                    && item.bbox[2] >= item.bbox[0]
                    && item.bbox[3] >= item.bbox[1]
                    && item.bbox[2] <= image_width as f32
                    && item.bbox[3] <= image_height as f32,
                "invalid detection bounding box"
            );
            let area = item.area.unwrap_or_else(|| {
                mask.pixels
                    .iter()
                    .filter(|&&value| value != 0)
                    .count()
                    .min(u32::MAX as usize) as u32
            });
            let label_id = item.label_id.unwrap_or(match item.label.as_str() {
                "text" => 0,
                "onomatopoeia" => 1,
                "bubble" => 2,
                "panel" => 3,
                _ => 4,
            });
            detections.push(KoharuLayoutDetection {
                label_id,
                label: item.label,
                score: item.score,
                bbox: item.bbox,
                area,
                mask,
            });
        }
        Ok(KoharuLayoutDetections {
            image_width,
            image_height,
            detections,
        })
    }
}

impl WireMask {
    fn into_mask(self) -> Result<KoharuLayoutMask> {
        let expected_len = (self.width as usize)
            .checked_mul(self.height as usize)
            .context("mask dimensions overflow")?;
        ensure!(expected_len <= 64_000_000, "mask exceeds pixel limit");
        if expected_len == 0 {
            return Ok(KoharuLayoutMask {
                x: self.x,
                y: self.y,
                width: 0,
                height: 0,
                pixels: Vec::new(),
            });
        }
        let pixels = if let Some(pixels) = self.pixels {
            pixels
        } else if let Some(b64) = self.pixels_b64 {
            decode_base64(&b64).context("failed to decode mask pixels_b64")?
        } else if let Some(png_b64) = self.png_b64 {
            let raw_png = decode_base64(&png_b64).context("failed to decode mask png_b64")?;
            let decoded = decode_image(&raw_png, 64_000_000)
                .context("failed to decode PNG mask")?
                .to_luma8();
            ensure!(
                decoded.dimensions() == (self.width, self.height),
                "decoded PNG mask dimensions {:?} do not match declared {:?}",
                decoded.dimensions(),
                (self.width, self.height)
            );
            decoded.into_raw()
        } else {
            bail!("mask must provide pixels, pixels_b64, or png_b64");
        };
        ensure!(
            pixels.len() == expected_len,
            "mask pixel count {} does not match {}x{} ({expected_len})",
            pixels.len(),
            self.width,
            self.height
        );
        Ok(KoharuLayoutMask {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            pixels,
        })
    }
}

#[derive(Deserialize)]
struct WireOcrResponse {
    #[serde(default)]
    results: Option<Vec<WireOcrItem>>,
    #[serde(default)]
    texts: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct WireOcrItem {
    #[serde(default)]
    id: Option<String>,
    text: String,
}

impl WireOcrResponse {
    fn resolve(self, expected_ids: &[String]) -> Result<Vec<String>> {
        if let Some(texts) = self.texts {
            ensure!(
                texts.len() == expected_ids.len(),
                "remote OCR returned {} texts for {} regions",
                texts.len(),
                expected_ids.len()
            );
            return Ok(texts);
        }
        if let Some(items) = self.results {
            ensure!(
                items.len() == expected_ids.len(),
                "remote OCR returned {} results for {} regions",
                items.len(),
                expected_ids.len()
            );
            let mut by_id = std::collections::HashMap::with_capacity(items.len());
            for item in items {
                let id = item.id.context("every OCR result must include an ID")?;
                if by_id.insert(id.clone(), item.text).is_some() {
                    bail!("remote OCR returned duplicate result id {id}");
                }
            }
            let mut ordered = Vec::with_capacity(expected_ids.len());
            for id in expected_ids {
                let text = by_id
                    .remove(id)
                    .with_context(|| format!("remote OCR omitted result for region id {id}"))?;
                ordered.push(text);
            }
            return Ok(ordered);
        }
        bail!("remote OCR response must contain 'results' or 'texts'")
    }
}

pub async fn bounded_body(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>> {
    ensure!(
        response
            .content_length()
            .is_none_or(|size| size <= limit as u64),
        "response exceeds byte limit"
    );
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            body.len().saturating_add(chunk.len()) <= limit,
            "response exceeds byte limit"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

pub fn decode_image(bytes: &[u8], max_pixels: u32) -> Result<DynamicImage> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let (width, height) = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .into_dimensions()?;
    ensure!(
        width > 0 && height > 0 && u64::from(width) * u64::from(height) <= u64::from(max_pixels),
        "image exceeds pixel limit"
    );
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(width);
    limits.max_image_height = Some(height);
    limits.max_alloc = Some(u64::from(max_pixels) * 16);
    reader.limits(limits);
    reader.decode().context("failed to decode bounded image")
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct EndpointProbeResult {
    pub reachable: bool,
    pub models: Vec<String>,
    pub message: String,
}

#[derive(Deserialize)]
struct ModelsListResponse {
    #[serde(default)]
    data: Vec<ModelListItem>,
}

#[derive(Deserialize)]
struct ModelListItem {
    id: String,
}

pub async fn probe_endpoint(
    stage: crate::Stage,
    endpoint_config: &RemoteEndpointConfig,
    secret_key: SecretKey<'_>,
) -> Result<EndpointProbeResult> {
    endpoint_config.validate(&stage.to_string())?;
    let base_url = endpoint_config
        .base_url
        .as_ref()
        .context("endpoint has no base URL")?;
    let url = resolve_endpoint(base_url, "v1/models");
    let client = http_client()?;
    let mut request = client.get(&url).timeout(Duration::from_secs(
        endpoint_config.timeout_seconds.min(30).into(),
    ));
    if let Some(api_key) = koharu_secrets::get(secret_key)? {
        request = request.bearer_auth(api_key.expose_secret());
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            return Ok(EndpointProbeResult {
                reachable: false,
                models: Vec::new(),
                message: format!("connection failed: {error:#}"),
            });
        }
    };
    let status = response.status();
    let raw = bounded_body(response, 1024 * 1024)
        .await
        .unwrap_or_default();
    if status.is_success() {
        let models = serde_json::from_slice::<ModelsListResponse>(&raw)
            .map(|parsed| {
                parsed
                    .data
                    .into_iter()
                    .map(|item| item.id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        return Ok(EndpointProbeResult {
            reachable: true,
            models,
            message: format!("HTTP {}", status.as_u16()),
        });
    }
    if matches!(
        status,
        reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::METHOD_NOT_ALLOWED
    ) {
        return Ok(EndpointProbeResult {
            reachable: true,
            models: Vec::new(),
            message: format!("server reachable (HTTP {} on /v1/models)", status.as_u16()),
        });
    }
    let text = String::from_utf8_lossy(&raw);
    Ok(EndpointProbeResult {
        reachable: false,
        models: Vec::new(),
        message: format!("HTTP {}: {}", status.as_u16(), truncate_error(&text)),
    })
}

pub fn encode_rgba_png(image: &DynamicImage) -> Result<Vec<u8>> {
    let rgba = image.to_rgba8();
    let mut bytes = Cursor::new(Vec::new());
    PngEncoder::new_with_quality(&mut bytes, CompressionType::Fast, FilterType::NoFilter)
        .write_image(
            rgba.as_raw(),
            rgba.width(),
            rgba.height(),
            ExtendedColorType::Rgba8,
        )
        .context("failed to encode RGBA PNG")?;
    Ok(bytes.into_inner())
}

pub fn encode_rgb_png(image: &DynamicImage) -> Result<Vec<u8>> {
    let rgb = image.to_rgb8();
    let mut bytes = Cursor::new(Vec::new());
    PngEncoder::new_with_quality(&mut bytes, CompressionType::Fast, FilterType::NoFilter)
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            ExtendedColorType::Rgb8,
        )
        .context("failed to encode RGB PNG")?;
    Ok(bytes.into_inner())
}

pub fn decode_openai_edit_mask_png(
    bytes: &[u8],
    expected_width: u32,
    expected_height: u32,
    max_pixels: u32,
) -> Result<GrayImage> {
    let image = decode_image(bytes, max_pixels).context("invalid mask image")?;
    ensure!(
        image.width() == expected_width && image.height() == expected_height,
        "mask dimensions {}x{} do not match image {}x{}",
        image.width(),
        image.height(),
        expected_width,
        expected_height
    );
    let has_alpha = image.color().has_alpha();
    let rgba = image.to_rgba8();
    let mut mask = GrayImage::new(expected_width, expected_height);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let inpaint = if has_alpha {
            pixel[3] < 128
        } else {
            pixel[0] >= 127 || pixel[1] >= 127 || pixel[2] >= 127
        };
        mask.put_pixel(x, y, image::Luma([if inpaint { 255 } else { 0 }]));
    }
    Ok(mask)
}

/// Encodes a `GrayImage` cleanup mask into an OpenAI `/v1/images/edits` RGBA
/// PNG mask: pixels with mask value >= 127 (to be inpainted) get `alpha = 0`
/// (transparent) and `RGB = 255`, while pixels with mask value < 127 (to be
/// preserved) get `alpha = 255` (opaque) and `RGB = 0`.
pub fn encode_openai_edit_mask_png(mask: &GrayImage) -> Result<Vec<u8>> {
    let mut rgba = RgbaImage::new(mask.width(), mask.height());
    for (x, y, pixel) in mask.enumerate_pixels() {
        let inpaint = pixel[0] >= 127;
        let target = if inpaint {
            Rgba([255, 255, 255, 0])
        } else {
            Rgba([0, 0, 0, 255])
        };
        rgba.put_pixel(x, y, target);
    }
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(rgba)
        .write_to(&mut bytes, ImageFormat::Png)
        .context("failed to encode OpenAI edit mask PNG")?;
    Ok(bytes.into_inner())
}

fn decode_base64(input: &str) -> Result<Vec<u8>> {
    let trimmed = input.trim();
    let payload = trimmed
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
        .map_or(trimmed, |(_, data)| data);
    STANDARD.decode(payload).map_err(Into::into)
}

fn truncate_error(text: &str) -> String {
    text.chars().take(2048).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_resolution_handles_with_and_without_v1_prefix() {
        let root = Url::parse("http://gpu.internal:8000").unwrap();
        let with_v1 = Url::parse("http://gpu.internal:8000/v1/").unwrap();
        let exact = Url::parse("http://gpu.internal:8000/v1/images/edits").unwrap();

        assert_eq!(
            resolve_endpoint(&root, "v1/images/edits"),
            "http://gpu.internal:8000/v1/images/edits"
        );
        assert_eq!(
            resolve_endpoint(&with_v1, "v1/images/edits"),
            "http://gpu.internal:8000/v1/images/edits"
        );
        assert_eq!(
            resolve_endpoint(&exact, "v1/images/edits"),
            "http://gpu.internal:8000/v1/images/edits"
        );
        assert_eq!(
            resolve_endpoint(&with_v1, "v1/detect"),
            "http://gpu.internal:8000/v1/detect"
        );
    }

    #[test]
    fn openai_edit_mask_marks_inpaint_pixels_transparent() {
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(0, 0, image::Luma([255]));
        mask.put_pixel(1, 0, image::Luma([0]));

        let encoded = encode_openai_edit_mask_png(&mask).unwrap();
        let decoded = image::load_from_memory(&encoded).unwrap().to_rgba8();

        assert_eq!(decoded.get_pixel(0, 0), &Rgba([255, 255, 255, 0]));
        assert_eq!(decoded.get_pixel(1, 0), &Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn wire_ocr_response_orders_by_id_and_rejects_mismatches() {
        let ids = vec!["r1".to_owned(), "r2".to_owned()];
        let out_of_order: WireOcrResponse = serde_json::from_value(serde_json::json!({
            "results": [
                { "id": "r2", "text": "two" },
                { "id": "r1", "text": "one" }
            ]
        }))
        .unwrap();
        assert_eq!(out_of_order.resolve(&ids).unwrap(), vec!["one", "two"]);

        let missing: WireOcrResponse = serde_json::from_value(serde_json::json!({
            "results": [
                { "id": "r1", "text": "one" }
            ]
        }))
        .unwrap();
        assert!(missing.resolve(&ids).is_err());
    }

    #[test]
    fn endpoint_limits_are_validated() {
        let mut config = RemoteEndpointConfig {
            base_url: Some(Url::parse("http://gpu.internal:8000").unwrap()),
            ..Default::default()
        };
        assert!(config.validate("OCR").is_ok());
        config.timeout_seconds = 0;
        assert!(config.validate("OCR").is_err());
        config.timeout_seconds = 300;
        config.max_response_mib = 257;
        assert!(config.validate("OCR").is_err());
    }

    #[test]
    fn oversized_images_fail_before_decoding_pixels() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(4, 4));
        let png = encode_rgb_png(&image).unwrap();
        assert!(decode_image(&png, 15).is_err());
        assert!(decode_image(&png, 16).is_ok());
    }

    #[test]
    fn openai_mask_round_trips_and_checks_dimensions() {
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(0, 0, image::Luma([255]));
        let png = encode_openai_edit_mask_png(&mask).unwrap();
        assert_eq!(decode_openai_edit_mask_png(&png, 2, 1, 4).unwrap(), mask);
        assert!(decode_openai_edit_mask_png(&png, 1, 2, 4).is_err());
    }

    #[test]
    fn opaque_white_rgba_mask_preserves_all_pixels() {
        let mask =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([255, 255, 255, 255])));
        let png = encode_rgba_png(&mask).unwrap();
        assert!(
            decode_openai_edit_mask_png(&png, 2, 2, 4)
                .unwrap()
                .pixels()
                .all(|pixel| pixel[0] == 0)
        );
    }

    #[tokio::test]
    async fn response_body_limit_is_enforced_without_content_length() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let worker = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            stream.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n8\r\n12345678\r\n0\r\n\r\n").await.unwrap();
        });
        let response = http_client()
            .unwrap()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap();
        assert!(bounded_body(response, 7).await.is_err());
        worker.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires a configured live Koharu inference instance"]
    async fn live_server_round_trip() {
        let base_url = std::env::var("KOHARU_TEST_ENDPOINT").expect("KOHARU_TEST_ENDPOINT");
        let endpoint = RemoteEndpointConfig {
            base_url: Some(Url::parse(&base_url).unwrap()),
            ..Default::default()
        };
        let client = http_client().unwrap();
        let page = image::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../koharu-ml/benches/fixtures/object_detection/1.jpg"),
        )
        .unwrap();
        let detections = detect_remote(
            &client,
            &endpoint,
            detection_secret_key(),
            "koharu-layout-rfdetr-seg-2xl",
            &page,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        assert!(!detections.detections.is_empty());
        assert_eq!(
            (detections.image_width, detections.image_height),
            (page.width(), page.height())
        );
        let crop = image::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../koharu-ml/benches/fixtures/ocr/title.png"),
        )
        .unwrap();
        let recognized = ocr_remote(
            &client,
            &endpoint,
            ocr_secret_key(),
            "manga-ocr",
            &[RemoteOcrRegion {
                id: "title".into(),
                image: crop,
            }],
        )
        .await
        .unwrap();
        assert_eq!(recognized.len(), 1);
        assert!(!recognized[0].is_empty());
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            128,
            128,
            image::Rgb([220, 220, 220]),
        ));
        let mut mask = GrayImage::new(128, 128);
        for y in 40..80 {
            for x in 40..80 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        let options = RemoteInpaintOptions {
            model: "lama",
            prompt: None,
            negative_prompt: None,
            steps: None,
            strength: None,
            guidance_scale: None,
            seed: None,
        };
        let result = inpaint_via_edits(
            &client,
            &endpoint,
            inpainting_secret_key(),
            &image,
            &mask,
            &options,
        )
        .await
        .unwrap();
        assert_eq!((result.width(), result.height()), (128, 128));
    }

    #[tokio::test]
    #[ignore = "requires a configured live Koharu inference instance"]
    async fn live_remote_pipeline_commits_scene_results() {
        use koharu_scene::{
            AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session, Snapshot,
        };
        use std::{collections::BTreeMap, sync::Arc};
        struct SessionCommitter<'a>(&'a mut Session);
        #[async_trait::async_trait]
        impl crate::Committer for SessionCommitter<'_> {
            async fn commit(&mut self, output: crate::StageOutput) -> Result<Snapshot> {
                self.0.commit(output.patch).await?;
                Ok(self.0.snapshot())
            }
        }
        let endpoint = RemoteEndpointConfig {
            base_url: Some(
                Url::parse(&std::env::var("KOHARU_TEST_ENDPOINT").expect("KOHARU_TEST_ENDPOINT"))
                    .unwrap(),
            ),
            ..Default::default()
        };
        let mut config = crate::PipelineConfig::default();
        config.ocr = crate::OcrModel::MangaOcr;
        config.remote = crate::RemotePipelineConfig {
            detection: StageExecution::Remote(endpoint.clone()),
            ocr: StageExecution::Remote(endpoint.clone()),
            inpainting: StageExecution::Remote(endpoint),
        };
        let source = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../koharu-ml/benches/fixtures/object_detection/1.jpg"),
        )
        .unwrap();
        let image = image::load_from_memory(&source).unwrap();
        let mut session = Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    PageDraft::new(
                        "remote-test",
                        f64::from(image.width()),
                        f64::from(image.height()),
                    ),
                    At::End,
                )?;
                edit.set_asset(
                    id,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        Arc::<[u8]>::from(source),
                        "image/jpeg",
                        AssetMetadata {
                            width: Some(image.width()),
                            height: Some(image.height()),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                page = Some(id);
                Ok(())
            })
            .unwrap();
        session.commit(patch).await.unwrap();
        let pipeline = crate::Pipeline::from_config(
            koharu_config::Config::memory(config),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
            koharu_ml::Device::cpu(),
        )
        .unwrap();
        let report = pipeline
            .execute(
                session.snapshot(),
                crate::Request {
                    operation: crate::Operation::Stages {
                        stages: vec![
                            crate::Stage::Detection,
                            crate::Stage::Ocr,
                            crate::Stage::Inpainting,
                        ],
                    },
                    scope: crate::Scope::Pages(vec![page.unwrap()]),
                    ..Default::default()
                },
                &mut SessionCommitter(&mut session),
            )
            .await
            .unwrap();
        assert_eq!(report.status, crate::RunStatus::Completed);
        assert_eq!(report.completed, 3);
        assert_eq!(report.total, 3);
        let snapshot = session.snapshot();
        let contents = snapshot
            .page(page.unwrap())
            .unwrap()
            .text_group()
            .unwrap()
            .unwrap()
            .text_layers()
            .unwrap();
        assert!(contents.len() > 0);
    }
}
