use secrecy::{SecretBox, zeroize::Zeroize};
use serde::{Deserialize, Serialize, Serializer};

pub use secrecy::{ExposeSecret, SerializableSecret};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SecretString(SecretBox<SecretValue>);

impl Default for SecretString {
    fn default() -> Self {
        Self::from(String::new())
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(SecretBox::new(Box::new(SecretValue(value))))
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self::from(value.to_owned())
    }
}

impl ExposeSecret<str> for SecretString {
    fn expose_secret(&self) -> &str {
        &self.0.expose_secret().0
    }
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct SecretValue(String);

impl Zeroize for SecretValue {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl secrecy::CloneableSecret for SecretValue {}

impl Serialize for SecretValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("[REDACTED]")
    }
}

impl SerializableSecret for SecretValue {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SecretKey<'a> {
    name: &'a str,
    environment_variable: Option<&'a str>,
}

impl<'a> SecretKey<'a> {
    #[must_use]
    pub const fn environment(name: &'a str, environment_variable: &'a str) -> Self {
        Self {
            name,
            environment_variable: Some(environment_variable),
        }
    }

    #[must_use]
    pub const fn stored(name: &'a str) -> Self {
        Self {
            name,
            environment_variable: None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'a str {
        self.name
    }

    #[must_use]
    pub const fn environment_variable(self) -> Option<&'a str> {
        self.environment_variable
    }
}

/// Environment variable that switches every secret to read-only, environment-only storage.
///
/// Container deployments set it to `environment`: Docker's default seccomp profile blocks the
/// keyring syscalls, and provider credentials are injected at run time instead. Desktops leave it
/// unset and keep the platform credential store.
const MODE_VARIABLE: &str = "KOHARU_SECRETS";

/// Whether secrets come only from environment variables and cannot be changed from Koharu.
#[must_use]
pub fn is_read_only() -> bool {
    environment_only(std::env::var_os(MODE_VARIABLE))
}

fn environment_only(mode: Option<std::ffi::OsString>) -> bool {
    mode.is_some_and(|mode| mode == "environment")
}

/// Load a secret. A non-blank environment variable mapped to the key wins over the credential
/// store, so injected credentials work in both modes.
pub fn get(key: SecretKey<'_>) -> anyhow::Result<Option<SecretString>> {
    if let Some(variable) = key.environment_variable()
        && let Some(secret) = environment_secret(variable, std::env::var_os(variable))?
    {
        return Ok(Some(secret));
    }
    if is_read_only() {
        return Ok(None);
    }
    if file_store_directory().is_some() {
        return file_store_get(key);
    }
    match entry(key.name())?.get_password() {
        Ok(value) => Ok(Some(SecretString::from(value))),
        Err(keyring_core::Error::NoEntry) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub fn set(key: SecretKey<'_>, secret: &SecretString) -> anyhow::Result<()> {
    if is_read_only() {
        anyhow::bail!(
            "secret '{}' is managed by {} and cannot be changed from Koharu",
            key.name(),
            key.environment_variable().unwrap_or("the environment")
        );
    }
    if file_store_directory().is_some() {
        return file_store_set(key, secret);
    }
    entry(key.name())?.set_password(secret.expose_secret())?;
    Ok(())
}

pub fn delete(key: SecretKey<'_>) -> anyhow::Result<()> {
    if is_read_only() {
        anyhow::bail!(
            "secret '{}' is managed by {} and cannot be cleared from Koharu",
            key.name(),
            key.environment_variable().unwrap_or("the environment")
        );
    }
    if file_store_directory().is_some() {
        return file_store_delete(key);
    }
    match entry(key.name())?.delete_credential() {
        Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn file_store_directory() -> Option<std::path::PathBuf> {
    (std::env::var_os(MODE_VARIABLE).is_some_and(|mode| mode == "file")).then(|| {
        std::env::var_os("KOHARU_CREDENTIAL_DIR").map_or_else(
            || {
                dirs::data_local_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("Koharu/credentials")
            },
            std::path::PathBuf::from,
        )
    })
}

fn file_store_path(key: SecretKey<'_>) -> anyhow::Result<std::path::PathBuf> {
    anyhow::ensure!(
        key.name()
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
        "invalid credential name"
    );
    Ok(file_store_directory()
        .ok_or_else(|| anyhow::anyhow!("file credential store is not selected"))?
        .join(key.name()))
}

fn file_store_get(key: SecretKey<'_>) -> anyhow::Result<Option<SecretString>> {
    let path = file_store_path(key)?;
    match std::fs::read_to_string(&path) {
        Ok(value) => Ok(Some(value.into())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn file_store_set(key: SecretKey<'_>, secret: &SecretString) -> anyhow::Result<()> {
    let path = file_store_path(key)?;
    write_secret_file(&path, secret)
}

fn write_secret_file(path: &std::path::Path, secret: &SecretString) -> anyhow::Result<()> {
    use std::io::Write;
    let directory = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("credential directory is missing"))?;
    std::fs::create_dir_all(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
    }
    let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temporary
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    temporary.write_all(secret.expose_secret().as_bytes())?;
    temporary.as_file().sync_all()?;
    temporary.persist(path)?;
    Ok(())
}

fn file_store_delete(key: SecretKey<'_>) -> anyhow::Result<()> {
    match std::fs::remove_file(file_store_path(key)?) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn environment_secret(
    variable: &str,
    value: Option<std::ffi::OsString>,
) -> anyhow::Result<Option<SecretString>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{variable} contains non-Unicode data"))?;
    Ok((!value.trim().is_empty()).then(|| SecretString::from(value)))
}

const SERVICE: &str = "koharu";

// Linux has no default credential store. Secret Service (GNOME Keyring, KWallet) keeps secrets
// in the user's login keyring on disk. It is registered on first use, so the environment-only
// mode never connects to the session bus.
#[cfg(target_os = "linux")]
static LINUX_STORE: std::sync::LazyLock<Result<(), String>> = std::sync::LazyLock::new(|| {
    zbus_secret_service_keyring_store::Store::new()
        .map(|store| keyring_core::set_default_store(store))
        .map_err(|error| error.to_string())
});

fn entry(key: &str) -> anyhow::Result<keyring_core::Entry> {
    #[cfg(target_os = "linux")]
    {
        if let Err(error) = &*LINUX_STORE {
            anyhow::bail!("failed to connect to the Secret Service: {error}");
        }
        Ok(keyring_core::Entry::new(SERVICE, key)?)
    }
    #[cfg(not(target_os = "linux"))]
    Ok(keyring::Entry::new(SERVICE, key)?.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_values_are_loaded_without_exposing_them() {
        let secret = environment_secret("TEST_KEY", Some("  secret  ".into()))
            .unwrap()
            .unwrap();
        assert_eq!(secret.expose_secret(), "  secret  ");
        assert_eq!(serde_json::to_string(&secret).unwrap(), "\"[REDACTED]\"");
    }

    #[test]
    fn missing_and_blank_environment_values_are_absent() {
        assert!(environment_secret("TEST_KEY", None).unwrap().is_none());
        assert!(
            environment_secret("TEST_KEY", Some(" \t\n".into()))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn only_the_environment_mode_is_read_only() {
        assert!(environment_only(Some("environment".into())));
        assert!(!environment_only(None));
        assert!(!environment_only(Some("keyring".into())));
    }

    #[test]
    fn container_credentials_are_written_atomically_and_private() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("credentials");
        let path = root.join("pipeline-ocr");
        write_secret_file(&path, &SecretString::from("first")).unwrap();
        write_secret_file(&path, &SecretString::from("second")).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&root).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}
