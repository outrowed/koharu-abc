//! File downloads and progress events.

use std::{
    future::Future,
    path::Path,
    sync::{
        Arc, LazyLock, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use backon::{ExponentialBuilder, Retryable};
use futures::{StreamExt, TryStreamExt, stream};
use reqwest::{StatusCode, header};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncSeekExt, AsyncWriteExt, BufWriter},
    sync::broadcast,
};

use crate::network;

const EVENT_CAPACITY: usize = 256;
const CHUNK_SIZE: u64 = 64 * 1024 * 1024;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(50);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(5);
const RETRY_MIN_DELAY: Duration = Duration::from_millis(250);
const WRITE_BUFFER_SIZE: usize = 256 * 1024;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static EVENTS: LazyLock<broadcast::Sender<Event>> =
    LazyLock::new(|| broadcast::channel(EVENT_CAPACITY).0);
type DownloadClient = Arc<reqwest_middleware::ClientWithMiddleware>;

static CLIENT: OnceLock<DownloadClient> = OnceLock::new();

struct Activity {
    id: u64,
    name: String,
}

#[derive(Clone, Copy)]
struct Chunk {
    start: u64,
    end: u64,
}

impl Chunk {
    const fn len(self) -> u64 {
        self.end - self.start + 1
    }
}

impl Activity {
    fn start(name: String) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let _ = EVENTS.send(Event::Started {
            id,
            name: name.clone(),
        });
        Self { id, name }
    }

    fn progress(&self, completed: u64, total: u64) {
        let _ = EVENTS.send(Event::Progress {
            id: self.id,
            name: self.name.clone(),
            completed,
            total,
        });
    }

    async fn finish(self, destination: &Path, result: Result<()>) -> Result<()> {
        match result {
            Ok(()) => {
                let _ = EVENTS.send(Event::Finished { id: self.id });
                Ok(())
            }
            Err(error) => {
                let _ = tokio::fs::remove_file(destination).await;
                let _ = EVENTS.send(Event::Failed {
                    id: self.id,
                    name: self.name,
                    error: error.to_string(),
                });
                Err(error)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Event {
    Started {
        id: u64,
        name: String,
    },
    Progress {
        id: u64,
        name: String,
        completed: u64,
        total: u64,
    },
    Finished {
        id: u64,
    },
    Failed {
        id: u64,
        name: String,
        error: String,
    },
}

#[must_use]
pub fn subscribe() -> broadcast::Receiver<Event> {
    EVENTS.subscribe()
}

fn client(http: reqwest::Client, max_retries: usize) -> DownloadClient {
    CLIENT
        .get_or_init(|| {
            Arc::new(
                reqwest_middleware::ClientBuilder::new(http)
                    .with(reqwest_retry::RetryTransientMiddleware::new_with_policy(
                        reqwest_retry::policies::ExponentialBackoff::builder()
                            .build_with_max_retries(max_retries as u32),
                    ))
                    .build(),
            )
        })
        .clone()
}

pub(crate) async fn fetch(url: &str, destination: &Path) -> Result<()> {
    let name = reqwest::Url::parse(url)
        .ok()
        .and_then(|url| {
            url.path_segments()
                .and_then(|mut segments| segments.next_back())
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "download".to_owned());
    let config = network::config()?;
    let http = network::http()?;
    let read_timeout = Duration::from_secs(config.read_timeout.max(1));
    let max_retries = config.max_retries as usize;
    let client = client(http, max_retries);
    let activity = Activity::start(name);
    let result = fetch_http(
        &activity,
        &client,
        url,
        destination,
        read_timeout,
        max_retries,
    )
    .await;
    activity.finish(destination, result).await
}

/// Download the named asset from an exact release through GitHub's asset API.
/// The API resolves the signed storage URL without depending on the browser
/// download route, which intermediary caches can reject for large runtimes.
pub(crate) async fn github_release(
    repo: &str,
    tag: &str,
    asset: &str,
    destination: &Path,
) -> Result<()> {
    github_release_from("https://api.github.com", repo, tag, asset, destination).await
}

async fn github_release_from(
    api_base: &str,
    repo: &str,
    tag: &str,
    asset: &str,
    destination: &Path,
) -> Result<()> {
    #[derive(Deserialize)]
    struct Release {
        assets: Vec<Asset>,
    }
    #[derive(Deserialize)]
    struct Asset {
        name: String,
        url: String,
        size: u64,
    }
    let http = network::http()?;
    let timeout = Duration::from_secs(network::config()?.read_timeout.max(1));
    let metadata: Release = http
        .get(format!("{api_base}/repos/{repo}/releases/tags/{tag}"))
        .timeout(timeout)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let selected = metadata
        .assets
        .into_iter()
        .find(|candidate| candidate.name == asset)
        .with_context(|| format!("release {repo}@{tag} has no asset {asset}"))?;
    let client = client(http, network::config()?.max_retries as usize);
    let activity = Activity::start(asset.to_owned());
    let result = async {
        let response = tokio::time::timeout(
            timeout,
            client
                .get(&selected.url)
                .header(header::ACCEPT, "application/octet-stream")
                .send(),
        )
        .await
        .context("runtime download request timed out")??
        .error_for_status()?;
        ensure!(
            !response
                .headers()
                .get(header::CONTENT_TYPE)
                .is_some_and(|value| value.to_str().unwrap_or("").contains("json")),
            "GitHub returned metadata instead of runtime bytes"
        );
        write_stream(&activity, destination, selected.size, timeout, response.bytes_stream()).await
    }
    .await;
    activity.finish(destination, result).await
}

pub(crate) async fn receive<F, S, B, E>(name: &str, destination: &Path, source: F) -> Result<()>
where
    F: Future<Output = Result<(Option<u64>, S)>>,
    S: futures::Stream<Item = std::result::Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::error::Error + Send + Sync + 'static,
{
    let read_timeout = Duration::from_secs(network::config()?.read_timeout.max(1));
    let activity = Activity::start(name.to_owned());
    let result = async {
        let (total, stream) = source.await?;
        write_stream(
            &activity,
            destination,
            total.unwrap_or(0),
            read_timeout,
            stream,
        )
        .await
    }
    .await;
    activity.finish(destination, result).await
}

async fn fetch_http(
    activity: &Activity,
    client: &reqwest_middleware::ClientWithMiddleware,
    url: &str,
    destination: &Path,
    read_timeout: Duration,
    max_retries: usize,
) -> Result<()> {
    let response = tokio::time::timeout(
        read_timeout,
        client
            .get(url)
            .header(header::RANGE, "bytes=0-0")
            .header(header::ACCEPT_ENCODING, "identity")
            .send(),
    )
    .await
    .context("download probe timed out")??;
    if response.status() == StatusCode::PARTIAL_CONTENT {
        let content_range = response
            .headers()
            .get(header::CONTENT_RANGE)
            .context("range probe omitted Content-Range")?
            .to_str()?;
        let (_, total) = content_range
            .split_once('/')
            .context("range probe returned invalid Content-Range")?;
        let total = total.parse::<u64>()?;
        if total == 0 {
            tokio::fs::File::create(destination).await?;
            return Ok(());
        }
        return fetch_ranges(
            activity,
            client,
            url,
            destination,
            total,
            read_timeout,
            max_retries,
        )
        .await;
    }
    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE
        && response
            .headers()
            .get(header::CONTENT_RANGE)
            .is_some_and(|value| value == "bytes */0")
    {
        tokio::fs::File::create(destination).await?;
        return Ok(());
    }
    let response = response.error_for_status()?;
    drop(response);
    (|| fetch_stream(activity, client, url, destination, read_timeout))
        .retry(backoff(max_retries))
        .when(retryable)
        .notify(|error, delay| tracing::warn!(?delay, %error, "retrying download stream"))
        .await
}

async fn fetch_stream(
    activity: &Activity,
    client: &reqwest_middleware::ClientWithMiddleware,
    url: &str,
    destination: &Path,
    read_timeout: Duration,
) -> Result<()> {
    let response = tokio::time::timeout(
        read_timeout,
        client
            .get(url)
            .header(header::ACCEPT_ENCODING, "identity")
            .send(),
    )
    .await
    .context("download request timed out")??
    .error_for_status()?;
    let total = response.content_length().unwrap_or(0);
    write_stream(
        activity,
        destination,
        total,
        read_timeout,
        response.bytes_stream(),
    )
    .await
}

async fn fetch_ranges(
    activity: &Activity,
    client: &reqwest_middleware::ClientWithMiddleware,
    url: &str,
    destination: &Path,
    total: u64,
    read_timeout: Duration,
    max_retries: usize,
) -> Result<()> {
    tokio::fs::File::create(destination)
        .await?
        .set_len(total)
        .await?;
    let completed = Arc::new(AtomicU64::new(0));
    let chunks = (0..total).step_by(CHUNK_SIZE as usize).map(|start| Chunk {
        start,
        end: (start + CHUNK_SIZE).min(total) - 1,
    });
    stream::iter(chunks)
        .map(|chunk| {
            let completed = Arc::clone(&completed);
            async move {
                (|| fetch_range(client, url, destination, total, chunk, read_timeout))
                    .retry(backoff(max_retries))
                    .when(retryable)
                    .notify(|error, delay| {
                        tracing::warn!(start = chunk.start, end = chunk.end, ?delay, %error, "retrying download range");
                    })
                    .await?;
                let progress = completed.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
                activity.progress(progress, total);
                Ok::<(), anyhow::Error>(())
            }
        })
        .buffer_unordered(connection_count())
        .try_collect::<()>()
        .await?;
    Ok(())
}

async fn fetch_range(
    client: &reqwest_middleware::ClientWithMiddleware,
    url: &str,
    destination: &Path,
    total: u64,
    chunk: Chunk,
    read_timeout: Duration,
) -> Result<()> {
    let response = tokio::time::timeout(
        read_timeout,
        client
            .get(url)
            .header(
                header::RANGE,
                format!("bytes={}-{}", chunk.start, chunk.end),
            )
            .header(header::ACCEPT_ENCODING, "identity")
            .send(),
    )
    .await
    .context("download range request timed out")??
    .error_for_status()?;
    let expected = format!("bytes {}-{}/{}", chunk.start, chunk.end, total);
    let actual = response
        .headers()
        .get(header::CONTENT_RANGE)
        .context("download range omitted Content-Range")?
        .to_str()?;
    if actual != expected {
        bail!("download range returned {actual}, expected {expected}");
    }

    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(destination)
        .await?;
    file.seek(std::io::SeekFrom::Start(chunk.start)).await?;
    let mut file = BufWriter::with_capacity(WRITE_BUFFER_SIZE, file);
    let mut body = response.bytes_stream();
    let mut received = 0;
    while let Some(bytes) = tokio::time::timeout(read_timeout, body.next())
        .await
        .context("download range read timed out")?
    {
        let bytes = bytes?;
        received += bytes.len() as u64;
        file.write_all(&bytes).await?;
    }
    file.flush().await?;
    ensure!(
        received == chunk.len(),
        "download range ended after {received} bytes"
    );
    Ok(())
}

async fn write_stream<S, B, E>(
    activity: &Activity,
    destination: &Path,
    total: u64,
    read_timeout: Duration,
    mut body: S,
) -> Result<()>
where
    S: futures::Stream<Item = std::result::Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::error::Error + Send + Sync + 'static,
{
    let mut output = BufWriter::with_capacity(
        WRITE_BUFFER_SIZE,
        tokio::fs::File::create(destination).await?,
    );
    let mut completed = 0;
    let mut reported = 0;
    let mut last_report = Instant::now();
    while let Some(bytes) = tokio::time::timeout(read_timeout, body.next())
        .await
        .context("download read timed out")?
    {
        let bytes = bytes?;
        output.write_all(bytes.as_ref()).await?;
        completed += bytes.as_ref().len() as u64;
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            activity.progress(completed, total);
            reported = completed;
            last_report = Instant::now();
        }
    }
    output.flush().await?;
    if reported != completed {
        activity.progress(completed, total);
    }
    if total > 0 {
        ensure!(
            completed == total,
            "download ended after {completed} of {total} bytes"
        );
    }
    Ok(())
}

fn backoff(max_retries: usize) -> ExponentialBuilder {
    ExponentialBuilder::default()
        .with_min_delay(RETRY_MIN_DELAY)
        .with_max_delay(RETRY_MAX_DELAY)
        .with_max_times(max_retries)
        .with_jitter()
}

fn connection_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8)
        .saturating_mul(2)
        .clamp(8, 32)
}

fn retryable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
        .map(|error| error.is_body() || error.is_decode())
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn github_release_streams_named_asset() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let worker = tokio::spawn(async move {
            for step in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 2048];
                let n = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..n]);
                if step == 0 {
                    assert!(request.starts_with("GET /repos/koharu-rs/torch/releases/tags/v1 "));
                    let body = format!(
                        "{{\"assets\":[{{\"name\":\"cpu.tar.gz\",\"url\":\"http://{address}/assets/1\",\"size\":6}}]}}"
                    );
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).await.unwrap();
                } else {
                    assert!(
                        request.contains("GET /assets/1 ")
                            && request
                                .to_ascii_lowercase()
                                .contains("accept: application/octet-stream")
                    );
                    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 6\r\nConnection: close\r\n\r\ntorch!").await.unwrap();
                }
            }
        });
        let destination = tempfile::NamedTempFile::new().unwrap();
        github_release_from(
            &format!("http://{address}"),
            "koharu-rs/torch",
            "v1",
            "cpu.tar.gz",
            destination.path(),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(destination.path()).unwrap(), b"torch!");
        worker.await.unwrap();
    }
}
