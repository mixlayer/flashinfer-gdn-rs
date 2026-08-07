use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::{Error, Result};

#[derive(Debug, Clone)]
enum Argument {
    Literal(OsString),
    OutputDirectory,
}

/// Out-of-process compiler invocation whose output is redirected to build.log.
#[derive(Debug, Clone)]
pub struct CompilerCommand {
    program: PathBuf,
    arguments: Vec<Argument>,
    environment: BTreeMap<OsString, OsString>,
    current_directory: Option<PathBuf>,
    timeout: Duration,
}

impl CompilerCommand {
    /// Creates a compiler command with a fifteen-minute timeout.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            arguments: Vec::new(),
            environment: BTreeMap::new(),
            current_directory: None,
            timeout: Duration::from_secs(15 * 60),
        }
    }

    /// Appends a literal worker argument.
    #[must_use]
    pub fn arg(mut self, value: impl AsRef<OsStr>) -> Self {
        self.arguments
            .push(Argument::Literal(value.as_ref().to_owned()));
        self
    }

    /// Appends the staging artifact directory as a worker argument.
    #[must_use]
    pub fn output_directory_arg(mut self) -> Self {
        self.arguments.push(Argument::OutputDirectory);
        self
    }

    /// Adds or replaces one environment variable for the worker.
    #[must_use]
    pub fn env(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        self.environment
            .insert(key.as_ref().to_owned(), value.as_ref().to_owned());
        self
    }

    /// Sets the worker's current directory.
    #[must_use]
    pub fn current_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.current_directory = Some(path.into());
        self
    }

    /// Replaces the compiler timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Runs the worker and waits for successful completion.
    pub fn run(&self, output_directory: &Path) -> Result<()> {
        fs::create_dir_all(output_directory).map_err(|error| {
            Error::io(
                format!(
                    "failed to create compiler output directory {}",
                    output_directory.display()
                ),
                error,
            )
        })?;
        let log_path = output_directory.join("build.log");
        let mut log = File::create(&log_path).map_err(|error| {
            Error::io(
                format!("failed to create compiler log {}", log_path.display()),
                error,
            )
        })?;

        let resolved_arguments: Vec<OsString> = self
            .arguments
            .iter()
            .map(|argument| match argument {
                Argument::Literal(value) => value.clone(),
                Argument::OutputDirectory => output_directory.as_os_str().to_owned(),
            })
            .collect();
        writeln!(
            log,
            "program: {:?}\narguments: {:?}",
            self.program, resolved_arguments
        )
        .map_err(|error| {
            Error::io(
                format!("failed to write compiler log {}", log_path.display()),
                error,
            )
        })?;
        let stdout = log.try_clone().map_err(|error| {
            Error::io(
                format!("failed to clone compiler log {}", log_path.display()),
                error,
            )
        })?;

        let mut command = Command::new(&self.program);
        command
            .args(&resolved_arguments)
            .envs(&self.environment)
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(log));
        if let Some(directory) = &self.current_directory {
            command.current_dir(directory);
        }

        let mut child = command.spawn().map_err(|source| Error::CompilerSpawn {
            program: self.program.clone(),
            source,
        })?;
        let started = Instant::now();
        loop {
            if let Some(status) = child.try_wait().map_err(|error| {
                Error::io(
                    format!("failed to wait for compiler worker {}", child.id()),
                    error,
                )
            })? {
                return if status.success() {
                    Ok(())
                } else {
                    Err(Error::CompilerExited { status })
                };
            }
            if started.elapsed() >= self.timeout {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::CompilerTimedOut {
                    timeout: self.timeout,
                });
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let count = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cutedsl-command-test-{}-{timestamp}-{count}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn captures_worker_output() {
        let directory = TestDirectory::new();
        CompilerCommand::new("/bin/sh")
            .arg("-c")
            .arg("printf worker-output")
            .run(&directory.0)
            .unwrap();
        let log = fs::read_to_string(directory.0.join("build.log")).unwrap();
        assert!(log.contains("worker-output"));
    }

    #[test]
    fn reports_nonzero_worker_status() {
        let directory = TestDirectory::new();
        let error = CompilerCommand::new("/bin/sh")
            .arg("-c")
            .arg("exit 7")
            .run(&directory.0)
            .unwrap_err();
        assert!(matches!(error, Error::CompilerExited { .. }));
    }

    #[test]
    fn kills_worker_at_timeout() {
        let directory = TestDirectory::new();
        let error = CompilerCommand::new("/bin/sleep")
            .arg("2")
            .timeout(Duration::from_millis(20))
            .run(&directory.0)
            .unwrap_err();
        assert!(matches!(error, Error::CompilerTimedOut { .. }));
    }
}
