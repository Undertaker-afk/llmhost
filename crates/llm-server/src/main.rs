//! Standalone llmhost server.
//!
//! Binds `127.0.0.1:8080` by default. Override with `--addr <ip:port>` or
//! `--addr=<ip:port>`. OpenAI and Anthropic generate routes land in M3.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

const DEFAULT_ADDR: &str = "127.0.0.1:8080";

#[derive(Serialize)]
struct ModelsList {
    object: &'static str,
    data: Vec<serde_json::Value>,
}

fn parse_addr<I, S>(args: I) -> Result<String, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut addr = DEFAULT_ADDR.to_string();
    let mut iter = args.into_iter();
    while let Some(raw) = iter.next() {
        let arg = raw.as_ref();
        if let Some(value) = arg.strip_prefix("--addr=") {
            if value.is_empty() {
                return Err("--addr requires a value".to_string());
            }
            addr = value.to_string();
        } else if arg == "--addr" {
            match iter.next() {
                Some(value) if !value.as_ref().is_empty() => {
                    addr = value.as_ref().to_string();
                }
                _ => return Err("--addr requires a value".to_string()),
            }
        } else {
            return Err(format!("unknown argument: {arg}"));
        }
    }
    Ok(addr)
}

fn models_list() -> ModelsList {
    ModelsList {
        object: "list",
        data: Vec::new(),
    }
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn list_models() -> Json<ModelsList> {
    Json(models_list())
}

fn app() -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
}

#[tokio::main]
async fn main() {
    let addr = match parse_addr(std::env::args().skip(1)) {
        Ok(addr) => addr,
        Err(err) => {
            eprintln!("{err}");
            eprintln!("usage: llm-server [--addr HOST:PORT]");
            std::process::exit(2);
        }
    };
    let bind: std::net::SocketAddr = addr.parse().unwrap_or_else(|err| {
        eprintln!("invalid --addr {addr:?}: {err}");
        std::process::exit(2);
    });
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .unwrap_or_else(|err| {
            eprintln!("failed to bind {bind}: {err}");
            std::process::exit(1);
        });
    axum::serve(listener, app()).await.unwrap_or_else(|err| {
        eprintln!("server error: {err}");
        std::process::exit(1);
    });
}

#[cfg(test)]
mod tests {
    use super::{models_list, parse_addr, DEFAULT_ADDR};

    #[test]
    fn default_addr_is_localhost() {
        assert_eq!(parse_addr(Vec::<&str>::new()).unwrap(), DEFAULT_ADDR);
        assert_eq!(DEFAULT_ADDR, "127.0.0.1:8080");
    }

    #[test]
    fn addr_flag_forms() {
        assert_eq!(
            parse_addr(["--addr", "127.0.0.1:9"]).unwrap(),
            "127.0.0.1:9"
        );
        assert_eq!(parse_addr(["--addr=10.0.0.1:80"]).unwrap(), "10.0.0.1:80");
        assert!(parse_addr(["--addr"]).is_err());
        assert!(parse_addr(["--nope"]).is_err());
    }

    #[test]
    fn models_json_is_empty_openai_list() {
        let value = serde_json::to_value(models_list()).unwrap();
        assert_eq!(value["object"], "list");
        assert_eq!(value["data"], serde_json::json!([]));
    }
}
