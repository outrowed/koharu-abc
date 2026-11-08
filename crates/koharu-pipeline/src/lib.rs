//! In-process, scene-native model orchestration for Koharu.

mod accelerator;
mod config;
mod error;
mod execution;
mod images;
pub mod inference;
mod model_cell;
mod pipeline;
mod progress;
pub mod remote;
mod report;
mod request;
mod resources;
mod scheduler;
mod scope;
mod stage;
mod stage_runner;
mod stages;

pub use config::{
    DetectionModel, InpaintingModel, OcrModel, PipelineConfig, ProcessorConfig,
    RemotePipelineConfig, TranslationConfig,
};
pub use error::{ErrorKind, PipelineError};
pub use inference::{InferenceService, InpaintOptions, ServerConfig, ServerFeatures, ServerStatus};
pub use pipeline::Pipeline;
pub use progress::{Progress, ProgressSink};
pub use remote::{RemoteEndpointConfig, StageExecution};
pub use report::{Committer, Report, RunStatus, StageOutput};
pub use request::{InpaintingMask, Operation, Request, StopToken};
pub use resources::{
    Bytes, DeviceResources, Estimate, MeasuredSource, MemoryScope, Provenance, ResourceSnapshot,
    Tier, file_size,
};
pub use scope::{Bounds, Scope};
pub use stage::Stage;
pub use stages::{
    Flux2KleinConfig, Flux3KleinConfig, Flux3KleinSourceConfig, KoharuLayoutRFDetrSeg2XLConfig,
    LaMaConfig, MangaInpaintorConfig, MiGanConfig, PaddleOcrVl1_6MangaLoraConfig, PowerPaintConfig,
    RoremMixedConfig, WeightsFormatConfig,
};

use images::ImageCache;
use model_cell::ModelCell;

#[cfg(test)]
mod tests;
