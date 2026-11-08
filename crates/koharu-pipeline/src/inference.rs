use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use image::{DynamicImage, GenericImageView as _, GrayImage};
use koharu_config::Config;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{DetectionModel, InpaintingModel, OcrModel, Stage};

fn default_mib() -> u32 {
    64
}
fn default_pixels() -> u32 {
    32_000_000
}
fn default_queue() -> u32 {
    4
}
fn default_require_token() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ServerFeatures {
    pub detection: bool,
    pub ocr: bool,
    pub inpainting: bool,
}
impl Default for ServerFeatures {
    fn default() -> Self {
        Self {
            detection: true,
            ocr: true,
            inpainting: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ServerConfig {
    pub enabled: bool,
    pub cpu: bool,
    pub features: ServerFeatures,
    pub detection: DetectionModel,
    pub ocr: OcrModel,
    pub inpainting: InpaintingModel,
    #[serde(default = "default_mib")]
    pub max_request_mib: u32,
    #[serde(default = "default_pixels")]
    pub max_image_pixels: u32,
    #[serde(default = "default_queue")]
    pub max_queue: u32,
    #[serde(default = "default_require_token")]
    pub require_api_token: bool,
}
impl Default for ServerConfig {
    fn default() -> Self {
        let pipeline = crate::PipelineConfig::default();
        Self {
            enabled: false,
            cpu: false,
            features: ServerFeatures::default(),
            detection: pipeline.detection,
            ocr: pipeline.ocr,
            inpainting: pipeline.inpainting,
            max_request_mib: default_mib(),
            max_image_pixels: default_pixels(),
            max_queue: default_queue(),
            require_api_token: true,
        }
    }
}
impl ServerConfig {
    pub fn load() -> Result<Config<Self>> {
        koharu_config::load("server")
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=256).contains(&self.max_request_mib),
            "max_request_mib must be between 1 and 256"
        );
        ensure!(
            (1..=64_000_000).contains(&self.max_image_pixels),
            "max_image_pixels must be between 1 and 64000000"
        );
        ensure!(self.max_queue <= 32, "max_queue must not exceed 32");
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &self.detection;
        for (name, value) in [
            ("text", settings.text_threshold),
            ("bubble", settings.bubble_threshold),
            ("panel", settings.panel_threshold),
        ] {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "{name} threshold must be between zero and one"
                );
            }
        }
        if let OcrModel::PaddleOcrVl1_6MangaLora(settings) = &self.ocr {
            settings.validate()?;
        }
        self.inpainting.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct InpaintOptions {
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub steps: Option<u32>,
    pub strength: Option<f64>,
    pub guidance_scale: Option<f32>,
    #[specta(type = f64)]
    pub seed: Option<i64>,
}
impl Default for InpaintOptions {
    fn default() -> Self {
        Self {
            prompt: None,
            negative_prompt: None,
            steps: None,
            strength: None,
            guidance_scale: None,
            seed: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ServerStatus {
    pub config: ServerConfig,
    pub active: Option<Stage>,
    pub loaded: Option<Stage>,
    pub queued: u32,
    #[specta(type = f64)]
    pub completed: u64,
    #[specta(type = f64)]
    pub failed: u64,
    pub last_error: Option<String>,
}
impl Default for ServerStatus {
    fn default() -> Self {
        Self {
            config: ServerConfig::default(),
            active: None,
            loaded: None,
            queued: 0,
            completed: 0,
            failed: 0,
            last_error: None,
        }
    }
}

/// In-process inference facade. Construction does not load model weights.
#[derive(Clone)]
pub struct InferenceService {
    config: Config<ServerConfig>,
    execution: Arc<tokio::sync::Mutex<()>>,
    device: koharu_ml::Device,
    state: Arc<tokio::sync::Mutex<ServerStatus>>,
    pipeline: Arc<
        std::sync::Mutex<
            Option<std::sync::Weak<arc_swap::ArcSwap<crate::stage_runner::StageRunner>>>,
        >,
    >,
    runtime: Arc<tokio::sync::Mutex<Option<crate::stages::Stages>>>,
    queued: Arc<std::sync::atomic::AtomicU32>,
}
impl InferenceService {
    pub fn from_config(
        config: Config<ServerConfig>,
        execution: Arc<tokio::sync::Mutex<()>>,
        device: &koharu_ml::Device,
    ) -> Result<Self> {
        let value = config.read()?.clone();
        value.validate()?;
        let actual_device = if value.cpu {
            koharu_ml::Device::cpu()
        } else {
            device.clone()
        };
        let runtime = build_stages(&value, &actual_device)?;
        Ok(Self {
            config,
            execution,
            device: device.clone(),
            state: Arc::new(tokio::sync::Mutex::new(ServerStatus {
                config: value,
                ..ServerStatus::default()
            })),
            pipeline: Arc::new(std::sync::Mutex::new(None)),
            runtime: Arc::new(tokio::sync::Mutex::new(Some(runtime))),
            queued: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        })
    }
    pub(crate) fn attach_pipeline(
        &self,
        runner: &Arc<arc_swap::ArcSwap<crate::stage_runner::StageRunner>>,
    ) {
        if let Ok(mut pipeline) = self.pipeline.lock() {
            *pipeline = Some(Arc::downgrade(runner));
        }
    }
    pub(crate) async fn evict_models(&self) {
        if let Some(stages) = self.runtime.lock().await.as_ref() {
            stages.unload_all();
        }
        let mut status = self.state.lock().await;
        status.loaded = None;
    }
    pub fn config(&self) -> Result<ServerConfig> {
        Ok(self.config.read()?.clone())
    }
    pub fn catalog(&self) -> Vec<String> {
        let Ok(config) = self.config.read() else {
            return Vec::new();
        };
        let mut models = Vec::new();
        if config.features.detection {
            models.push(detection_id(&config.detection).to_owned());
        }
        if config.features.ocr {
            models.push(ocr_id(&config.ocr).to_owned());
        }
        if config.features.inpainting {
            models.push(inpainting_id(&config.inpainting).to_owned());
        }
        models
    }
    pub async fn status(&self) -> Result<ServerStatus> {
        let mut status = self.state.lock().await.clone();
        status.queued = self.queued.load(std::sync::atomic::Ordering::Acquire);
        Ok(status)
    }
    pub async fn configure(&self, config: ServerConfig) -> Result<ServerStatus> {
        config.validate()?;
        let _lane = self.execution.lock().await;
        let actual_device = if config.cpu {
            koharu_ml::Device::cpu()
        } else {
            self.device.clone()
        };
        let stages = build_stages(&config, &actual_device)?;
        if let Some(old) = self.runtime.lock().await.as_ref() {
            old.unload_all();
        }
        let previous = self.config()?;
        {
            let mut write = self.config.write()?;
            *write = config.clone();
            drop(write);
            if let Err(error) = self.config.save() {
                *self.config.write()? = previous;
                return Err(error);
            }
        }
        *self.runtime.lock().await = Some(stages);
        let mut status = self.state.lock().await;
        status.config = config;
        status.active = None;
        status.loaded = None;
        status.last_error = None;
        Ok(status.clone())
    }
    pub async fn unload(&self) -> Result<ServerStatus> {
        let _lane = self.execution.lock().await;
        if let Some(stages) = self.runtime.lock().await.as_ref() {
            for stage in Stage::ALL {
                stages.unload(stage);
            }
        }
        let mut status = self.state.lock().await;
        status.loaded = None;
        status.active = None;
        Ok(status.clone())
    }
    pub async fn warmup(&self, stage: Stage) -> Result<ServerStatus> {
        let service = self.clone();
        tokio::spawn(async move { service.warmup_inner(stage).await })
            .await
            .map_err(|error| anyhow!("warm-up task failed: {error}"))?
    }

    async fn warmup_inner(&self, stage: Stage) -> Result<ServerStatus> {
        let _lane = self.execution.lock().await;
        let config = self.config()?;
        ensure!(config.enabled, "inference server is disabled");
        let feature = match stage {
            Stage::Detection => config.features.detection,
            Stage::Ocr => config.features.ocr,
            Stage::Inpainting => config.features.inpainting,
            Stage::Translation => false,
        };
        ensure!(feature, "{stage} inference is disabled");
        {
            let mut status = self.state.lock().await;
            status.active = Some(stage);
        }
        let load_result = async {
            let mut runtime = self.runtime.lock().await;
            let stages = runtime
                .as_mut()
                .ok_or_else(|| anyhow!("inference runtime is unavailable"))?;
            unload_pipeline(&self.pipeline);
            for other in Stage::ALL {
                if other != stage {
                    stages.unload(other);
                }
            }
            stages.load(stage).await
        }
        .await;
        let mut status = self.state.lock().await;
        status.active = None;
        status.config = config;
        if let Err(error) = load_result {
            status.failed += 1;
            status.loaded = None;
            status.last_error = Some(format!("{error:#}"));
            return Err(error);
        }
        status.loaded = Some(stage);
        status.last_error = None;
        Ok(status.clone())
    }
    pub async fn detect(
        &self,
        model: Option<String>,
        image: DynamicImage,
        text: Option<f32>,
        bubble: Option<f32>,
        panel: Option<f32>,
    ) -> Result<koharu_ml::koharu_layout_rfdetr_seg_2xl::KoharuLayoutDetections> {
        let service = self.clone();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            service
                .detect_inner(model, image, text, bubble, panel, sender)
                .await
        });
        let _guard = CancelOnDrop(Some(receiver));
        task.await
            .map_err(|error| anyhow!("inference task failed: {error}"))?
    }
    async fn detect_inner(
        &self,
        model: Option<String>,
        image: DynamicImage,
        text: Option<f32>,
        bubble: Option<f32>,
        panel: Option<f32>,
        cancel: tokio::sync::oneshot::Sender<()>,
    ) -> Result<koharu_ml::koharu_layout_rfdetr_seg_2xl::KoharuLayoutDetections> {
        let _lane = self.acquire_lane(cancel).await?;
        let config = self.config()?;
        ensure!(
            config.enabled && config.features.detection,
            "detection inference is disabled"
        );
        check_model(model.as_deref(), detection_id(&config.detection))?;
        check_image(&image, &config)?;
        self.state.lock().await.active = Some(Stage::Detection);
        let mut runtime = self.runtime.lock().await;
        let stages = runtime
            .as_mut()
            .ok_or_else(|| anyhow!("inference runtime is unavailable"))?;
        let result = async {
            unload_pipeline(&self.pipeline);
            for stage in Stage::ALL {
                if stage != Stage::Detection {
                    stages.unload(stage);
                }
            }
            stages.load(Stage::Detection).await?;
            stages.detect(image, text, bubble, panel).await
        }
        .await;
        if result.is_err() {
            stages.unload(Stage::Detection);
        }
        self.complete(Stage::Detection, result.is_ok(), result.as_ref().err())
            .await;
        result
    }
    pub async fn ocr(
        &self,
        model: Option<String>,
        images: Vec<(String, DynamicImage)>,
    ) -> Result<Vec<(String, String)>> {
        let service = self.clone();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move { service.ocr_inner(model, images, sender).await });
        let _guard = CancelOnDrop(Some(receiver));
        task.await
            .map_err(|error| anyhow!("inference task failed: {error}"))?
    }
    async fn ocr_inner(
        &self,
        model: Option<String>,
        images: Vec<(String, DynamicImage)>,
        cancel: tokio::sync::oneshot::Sender<()>,
    ) -> Result<Vec<(String, String)>> {
        let _lane = self.acquire_lane(cancel).await?;
        let config = self.config()?;
        ensure!(
            config.enabled && config.features.ocr,
            "OCR inference is disabled"
        );
        check_model(model.as_deref(), ocr_id(&config.ocr))?;
        ensure!(images.len() <= 256, "OCR request has too many images");
        for (_, image) in &images {
            check_image(image, &config)?;
        }
        self.state.lock().await.active = Some(Stage::Ocr);
        let mut runtime = self.runtime.lock().await;
        let stages = runtime
            .as_mut()
            .ok_or_else(|| anyhow!("inference runtime is unavailable"))?;
        let result = async {
            unload_pipeline(&self.pipeline);
            for stage in Stage::ALL {
                if stage != Stage::Ocr {
                    stages.unload(stage);
                }
            }
            stages.load(Stage::Ocr).await?;
            stages.ocr(images).await
        }
        .await;
        if result.is_err() {
            stages.unload(Stage::Ocr);
        }
        self.complete(Stage::Ocr, result.is_ok(), result.as_ref().err())
            .await;
        result
    }
    pub async fn inpaint(
        &self,
        model: Option<String>,
        image: DynamicImage,
        mask: GrayImage,
        options: InpaintOptions,
    ) -> Result<DynamicImage> {
        let service = self.clone();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            service
                .inpaint_inner(model, image, mask, options, sender)
                .await
        });
        let _guard = CancelOnDrop(Some(receiver));
        task.await
            .map_err(|error| anyhow!("inference task failed: {error}"))?
    }
    async fn inpaint_inner(
        &self,
        model: Option<String>,
        image: DynamicImage,
        mask: GrayImage,
        options: InpaintOptions,
        cancel: tokio::sync::oneshot::Sender<()>,
    ) -> Result<DynamicImage> {
        let _lane = self.acquire_lane(cancel).await?;
        let config = self.config()?;
        ensure!(
            config.enabled && config.features.inpainting,
            "inpainting inference is disabled"
        );
        check_model(model.as_deref(), inpainting_id(&config.inpainting))?;
        check_image(&image, &config)?;
        ensure!(
            mask.dimensions() == image.dimensions(),
            "image and mask dimensions differ"
        );
        validate_inpaint_options(&config.inpainting, &options)?;
        self.state.lock().await.active = Some(Stage::Inpainting);
        let mut runtime = self.runtime.lock().await;
        let stages = runtime
            .as_mut()
            .ok_or_else(|| anyhow!("inference runtime is unavailable"))?;
        let result = async {
            unload_pipeline(&self.pipeline);
            for stage in Stage::ALL {
                if stage != Stage::Inpainting {
                    stages.unload(stage);
                }
            }
            stages.load(Stage::Inpainting).await?;
            stages.inpaint(image, mask, options).await
        }
        .await;
        if result.is_err() {
            stages.unload(Stage::Inpainting);
        }
        self.complete(Stage::Inpainting, result.is_ok(), result.as_ref().err())
            .await;
        result
    }
    async fn acquire_lane(
        &self,
        mut cancel: tokio::sync::oneshot::Sender<()>,
    ) -> Result<tokio::sync::OwnedMutexGuard<()>> {
        let config = self.config()?;
        ensure!(config.enabled, "inference server is disabled");
        if let Ok(guard) = self.execution.clone().try_lock_owned() {
            return Ok(guard);
        }
        ensure!(
            config.max_queue > 0,
            "inference lane is busy and queueing is disabled"
        );
        let permit = self.admit(&config)?;
        let lock = self.execution.clone().lock_owned();
        tokio::select! {
            guard = lock => { drop(permit); Ok(guard) }
            _ = cancel.closed() => { drop(permit); Err(anyhow!("inference request cancelled before execution")) }
        }
    }
    fn admit(&self, config: &ServerConfig) -> Result<QueuePermit> {
        ensure!(config.enabled, "inference server is disabled");
        let mut current = self.queued.load(std::sync::atomic::Ordering::Acquire);
        loop {
            ensure!(current < config.max_queue, "inference queue is full");
            match self.queued.compare_exchange_weak(
                current,
                current + 1,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            ) {
                Ok(_) => return Ok(QueuePermit(self.queued.clone())),
                Err(actual) => current = actual,
            }
        }
    }
    async fn complete(&self, stage: Stage, success: bool, error: Option<&anyhow::Error>) {
        let mut status = self.state.lock().await;
        status.active = None;
        if success {
            status.completed += 1;
            status.loaded = Some(stage);
            status.last_error = None;
        } else {
            status.failed += 1;
            status.loaded = None;
            status.last_error = error.map(|error| format!("{error:#}"));
        }
    }
}

#[allow(dead_code)]
struct CancelOnDrop(Option<tokio::sync::oneshot::Receiver<()>>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        // Dropping the receiver closes the cancellation channel while queued.
        self.0.take();
    }
}

fn validate_inpaint_options(model: &InpaintingModel, options: &InpaintOptions) -> Result<()> {
    if let Some(prompt) = options.prompt.as_deref() {
        ensure!(!prompt.contains('\0'), "prompt contains NUL byte");
        ensure!(!prompt.trim().is_empty(), "prompt cannot be blank");
    }
    if let Some(neg) = options.negative_prompt.as_deref() {
        ensure!(!neg.contains('\0'), "negative_prompt contains NUL byte");
    }
    if let Some(steps) = options.steps {
        ensure!(
            steps > 0 && steps <= 1000,
            "steps must be between 1 and 1000"
        );
    }
    if let Some(strength) = options.strength {
        ensure!(
            strength.is_finite() && strength > 0.0 && strength <= 1.0,
            "strength must be between 0 and 1"
        );
    }
    if let Some(guidance) = options.guidance_scale {
        ensure!(
            guidance.is_finite() && guidance > 0.0 && guidance <= 100.0,
            "guidance_scale must be between 0 and 100"
        );
    }
    match model {
        InpaintingModel::LaMa(_)
        | InpaintingModel::MiGan(_)
        | InpaintingModel::MangaInpaintor(_)
        | InpaintingModel::AotInpainting {} => {
            ensure!(
                options.prompt.is_none()
                    && options.negative_prompt.is_none()
                    && options.steps.is_none()
                    && options.strength.is_none()
                    && options.guidance_scale.is_none()
                    && options.seed.is_none(),
                "generation overrides are not supported for this promptless inpainting model"
            );
        }
        InpaintingModel::Flux2Klein(_) | InpaintingModel::Flux3Klein(_) => {
            ensure!(
                options.negative_prompt.is_none() && options.guidance_scale.is_none(),
                "negative_prompt and guidance_scale overrides are not supported for FLUX Klein"
            );
        }
        InpaintingModel::PowerPaint(_) => {
            ensure!(
                options.prompt.is_none() && options.negative_prompt.is_none(),
                "prompt overrides are not supported for PowerPaint"
            );
        }
        InpaintingModel::RoremMixed(_) => {}
    }
    Ok(())
}

struct QueuePermit(Arc<std::sync::atomic::AtomicU32>);
impl Drop for QueuePermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}
fn detection_id(_: &DetectionModel) -> &'static str {
    "koharu-layout-rfdetr-seg-2xl"
}
fn ocr_id(model: &OcrModel) -> &'static str {
    match model {
        OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
        OcrModel::PaddleOcrVl1_6MangaLora(_) => "paddleocr-vl-1.6-manga-lora",
        OcrModel::MangaOcr => "manga-ocr",
        OcrModel::MangaOcrCptFlex => "manga-ocr-cpt-flex",
        OcrModel::BaberuOcr => "baberu-ocr",
        OcrModel::HayaiOcr => "hayai-ocr",
    }
}
fn inpainting_id(model: &InpaintingModel) -> &'static str {
    match model {
        InpaintingModel::LaMa(_) => "lama",
        InpaintingModel::MiGan(_) => "mi-gan",
        InpaintingModel::MangaInpaintor(_) => "manga-inpaintor",
        InpaintingModel::AotInpainting {} => "aot-inpainting",
        InpaintingModel::Flux2Klein(_) => "flux2-klein",
        InpaintingModel::Flux3Klein(_) => "flux3-klein",
        InpaintingModel::RoremMixed(_) => "rorem-mixed",
        InpaintingModel::PowerPaint(_) => "powerpaint",
    }
}

fn unload_pipeline(
    pipeline: &std::sync::Mutex<
        Option<std::sync::Weak<arc_swap::ArcSwap<crate::stage_runner::StageRunner>>>,
    >,
) {
    let runner = pipeline
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().and_then(std::sync::Weak::upgrade));
    if let Some(runner) = runner {
        runner.load().unload_all();
    }
}
fn build_stages(
    config: &ServerConfig,
    device: &koharu_ml::Device,
) -> Result<crate::stages::Stages> {
    let pipeline = crate::PipelineConfig {
        detection: config.detection.clone(),
        ocr: config.ocr.clone(),
        inpainting: config.inpainting.clone(),
        ..crate::PipelineConfig::default()
    };
    let translator = koharu_translator::Translator::from_config(
        device.clone(),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
    )?;
    crate::stages::Stages::new(
        &pipeline,
        translator,
        device,
        crate::resources::ResourceMonitor::new(device),
    )
}
fn check_model(requested: Option<&str>, expected: &str) -> Result<()> {
    if let Some(requested) = requested {
        ensure!(
            requested == expected,
            "unsupported model '{requested}'; configured model is '{expected}'"
        );
    }
    Ok(())
}
fn check_image(image: &DynamicImage, config: &ServerConfig) -> Result<()> {
    let (width, height) = image.dimensions();
    ensure!(
        width > 0
            && height > 0
            && u64::from(width) * u64::from(height) <= u64::from(config.max_image_pixels),
        "image exceeds max_image_pixels"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn idle_lane_bypasses_queue_before_bounding_waiters() {
        let config = Config::memory(ServerConfig {
            enabled: true,
            max_queue: 1,
            ..ServerConfig::default()
        });
        let execution = Arc::new(tokio::sync::Mutex::new(()));
        let service =
            InferenceService::from_config(config, execution, &koharu_ml::Device::cpu()).unwrap();
        let (s1, _r1) = tokio::sync::oneshot::channel();
        let active = service.acquire_lane(s1).await.unwrap();
        let service_wait = service.clone();
        let (s2, _r2) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async move { service_wait.acquire_lane(s2).await });
        for _ in 0..20 {
            if service.status().await.unwrap().queued == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(service.status().await.unwrap().queued, 1);
        let (s3, _r3) = tokio::sync::oneshot::channel();
        assert!(service.acquire_lane(s3).await.is_err());
        drop(active);
        let second = waiter.await.unwrap().unwrap();
        assert_eq!(service.status().await.unwrap().queued, 0);
        drop(second);
    }

    #[tokio::test]
    async fn acquire_lane_with_zero_max_queue_rejects_when_busy() {
        let config = Config::memory(ServerConfig {
            enabled: true,
            max_queue: 0,
            ..ServerConfig::default()
        });
        let execution = Arc::new(tokio::sync::Mutex::new(()));
        let service =
            InferenceService::from_config(config, execution.clone(), &koharu_ml::Device::cpu())
                .unwrap();
        let held = execution.lock().await;
        let (_s, _r) = tokio::sync::oneshot::channel();
        assert!(service.acquire_lane(_s).await.is_err());
        drop(held);
        let (_s2, _r2) = tokio::sync::oneshot::channel();
        assert!(service.acquire_lane(_s2).await.is_ok());
    }

    #[tokio::test]
    async fn acquire_lane_cancels_if_caller_drops_before_admitted() {
        let config = Config::memory(ServerConfig {
            enabled: true,
            max_queue: 4,
            ..ServerConfig::default()
        });
        let execution = Arc::new(tokio::sync::Mutex::new(()));
        let service =
            InferenceService::from_config(config, execution.clone(), &koharu_ml::Device::cpu())
                .unwrap();
        let held = execution.lock().await;
        let (s, r) = tokio::sync::oneshot::channel();
        drop(r); // caller dropped before lock was acquired
        assert!(service.acquire_lane(s).await.is_err());
        drop(held);
        assert_eq!(service.queued.load(std::sync::atomic::Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn evict_models_clears_loaded_status() {
        let config = Config::memory(ServerConfig {
            enabled: true,
            ..ServerConfig::default()
        });
        let execution = Arc::new(tokio::sync::Mutex::new(()));
        let service =
            InferenceService::from_config(config, execution, &koharu_ml::Device::cpu()).unwrap();
        service.state.lock().await.loaded = Some(Stage::Inpainting);
        service.evict_models().await;
        assert_eq!(service.state.lock().await.loaded, None);
    }

    use super::*;

    #[test]
    fn default_config_is_disabled_and_valid() {
        let config = ServerConfig::default();
        assert!(!config.enabled);
        assert!(config.require_api_token);
        assert_eq!(config.max_queue, 4);
        assert_eq!(config.max_request_mib, 64);
        config.validate().unwrap();
    }

    #[test]
    fn config_validation_rejects_out_of_bounds() {
        let mut config = ServerConfig::default();
        config.max_queue = 33;
        assert!(config.validate().is_err());
        config.max_queue = 4;
        config.max_request_mib = 0;
        assert!(config.validate().is_err());
        config.max_request_mib = 300;
        assert!(config.validate().is_err());
        config.max_request_mib = 64;
        config.max_image_pixels = 0;
        assert!(config.validate().is_err());
        config.max_image_pixels = 70_000_000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validation_checks_threshold_bounds() {
        let mut config = ServerConfig::default();
        config.detection =
            DetectionModel::KoharuLayoutRFDetrSeg2XL(crate::KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(1.5),
                bubble_threshold: None,
                panel_threshold: None,
            });
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn disabled_server_refuses_admit_and_warmup() {
        let config = Config::memory(ServerConfig::default());
        let execution = Arc::new(tokio::sync::Mutex::new(()));
        let service =
            InferenceService::from_config(config, execution, &koharu_ml::Device::cpu()).unwrap();
        assert!(service.warmup(Stage::Detection).await.is_err());
        assert!(service.admit(&service.config().unwrap()).is_err());
    }

    #[test]
    fn queue_permit_tracks_bounded_queue() {
        let config = Config::memory(ServerConfig {
            enabled: true,
            max_queue: 2,
            ..ServerConfig::default()
        });
        let execution = Arc::new(tokio::sync::Mutex::new(()));
        let service =
            InferenceService::from_config(config, execution, &koharu_ml::Device::cpu()).unwrap();
        let p1 = service.admit(&service.config().unwrap()).unwrap();
        let p2 = service.admit(&service.config().unwrap()).unwrap();
        assert!(service.admit(&service.config().unwrap()).is_err());
        drop(p1);
        let _p3 = service.admit(&service.config().unwrap()).unwrap();
        drop(p2);
    }

    #[test]
    fn unsupported_inpaint_overrides_are_rejected_explicitly() {
        let options = InpaintOptions {
            seed: Some(1),
            ..Default::default()
        };
        assert!(
            validate_inpaint_options(
                &InpaintingModel::LaMa(crate::LaMaConfig::default()),
                &options
            )
            .is_err()
        );
        let options = InpaintOptions {
            guidance_scale: Some(2.0),
            ..Default::default()
        };
        assert!(
            validate_inpaint_options(
                &InpaintingModel::Flux2Klein(crate::Flux2KleinConfig::default()),
                &options
            )
            .is_err()
        );
    }
}
