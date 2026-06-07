import { Command } from "commander";
import { spawn, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  statSync,
  createWriteStream,
  unlinkSync,
  readFileSync,
  writeFileSync,
  readdirSync,
  rmSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve, basename, normalize, isAbsolute } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { get, del, type ClientOpts } from "../client.js";
import { resolveModelId } from "../resolve.js";
import { SUPPORTED_BASE_MODELS } from "../base-models.js";
import { DEFAULT_SPEC_FILE } from "./init.js";
import type { LocalSpec } from "../eval/types.js";
import { startManagedModelServer } from "../serve-manager.js";
import {
  printTable,
  printDetail,
  printSuccess,
  printJson,
  isJsonMode,
  formatDate,
  shortId,
  truncate,
  printWarning,
} from "../output.js";

interface Model {
  id: string;
  name: string;
  provider: string;
  provider_model_id: string;
  base_model: string;
  description: string | null;
  created_at: string;
}

interface ModelDownload {
  url: string;
  filename: string;
  expires_at: string;
}

interface ServeTarget {
  modelPath: string;
  modelName: string;
  source: "local-directory" | "local-archive" | "downloaded-model";
  modelId?: string;
}

const REFERENCE_SERVER_SCRIPT = String.raw`#!/usr/bin/env python3
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_PATH = os.environ["TT_MODEL_PATH"]
MODEL_NAME = os.environ.get("TT_MODEL_NAME", os.path.basename(MODEL_PATH.rstrip("/")) or "tuned-tensor-model")
SYSTEM_PROMPT = os.environ.get("TT_SYSTEM_PROMPT", "").strip()
HOST = os.environ.get("TT_HOST", "127.0.0.1")
PORT = int(os.environ.get("TT_PORT", "8000"))
DEFAULT_MAX_TOKENS = int(os.environ.get("TT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("TT_TEMPERATURE", "0.7"))
TRUST_REMOTE_CODE = os.environ.get("TT_TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes", "y"}
REQUESTED_DEVICE = os.environ.get("TT_DEVICE", "auto").lower()
JSON_REPAIR_ATTEMPTS = int(os.environ.get("TT_JSON_REPAIR_ATTEMPTS", "1"))
DEFAULT_JSON_SCHEMA = None
if os.environ.get("TT_JSON_SCHEMA"):
    DEFAULT_JSON_SCHEMA = json.loads(os.environ["TT_JSON_SCHEMA"])


def choose_device(torch):
    if REQUESTED_DEVICE not in {"auto", "cpu", "cuda", "mps"}:
        raise RuntimeError("TT_DEVICE must be one of: auto, cpu, cuda, mps")
    if REQUESTED_DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TT_DEVICE=cuda requested, but CUDA is not available")
    if REQUESTED_DEVICE == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("TT_DEVICE=mps requested, but Apple MPS is not available")
    if REQUESTED_DEVICE != "auto":
        return REQUESTED_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing Python serving dependencies. Install them with: "
            "python -m pip install torch transformers accelerate"
        ) from exc

    try:
        processor = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    except Exception:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=TRUST_REMOTE_CODE)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = choose_device(torch)
    kwargs = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        kwargs["device_map"] = "auto"
    elif device == "mps":
        kwargs["torch_dtype"] = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return torch, tokenizer, model, device


torch, tokenizer, model, DEVICE = load_model()


def content_to_text(content):
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def normalize_messages(raw_messages):
    messages = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        messages.append({"role": role, "content": content_to_text(item.get("content", ""))})

    if SYSTEM_PROMPT and not any(message["role"] == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def response_json_contract(body):
    response_format = body.get("response_format")
    schema = DEFAULT_JSON_SCHEMA
    mode = "json_schema" if schema else "text"

    if isinstance(response_format, dict):
        format_type = response_format.get("type")
        if format_type == "json_object":
            mode = "json_object"
        elif format_type == "json_schema":
            mode = "json_schema"
            json_schema = response_format.get("json_schema")
            if isinstance(json_schema, dict):
                schema = json_schema.get("schema") if isinstance(json_schema.get("schema"), dict) else json_schema

    return mode, schema


def with_json_contract(messages, mode, schema):
    if mode == "text":
        return messages

    if schema:
        contract = (
            "Return only a valid JSON object matching this JSON Schema. "
            "Do not include markdown, code fences, comments, or prose outside the JSON object.\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
    else:
        contract = (
            "Return only a valid JSON object. Do not include markdown, code fences, "
            "comments, or prose outside the JSON object."
        )

    contracted = [dict(message) for message in messages]
    for message in contracted:
        if message["role"] == "system":
            message["content"] = (message["content"].rstrip() + "\n\n" + contract).strip()
            return contracted

    contracted.insert(0, {"role": "system", "content": contract})
    return contracted


def render_prompt(messages):
    chat_template = getattr(tokenizer, "chat_template", None)
    if hasattr(tokenizer, "apply_chat_template") and chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)


def generate_completion(messages, max_tokens, temperature):
    prompt = render_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = getattr(model, "device", None)
    if input_device is None:
        input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    prompt_tokens = int(inputs["input_ids"].shape[-1])
    completion_ids = output_ids[0][prompt_tokens:]
    content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return content, prompt_tokens, int(completion_ids.shape[-1])


def extract_json_value(content):
    decoder = json.JSONDecoder()
    text = content.strip()
    try:
        return json.loads(text), None
    except Exception:
        pass

    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
            trailing = text[index + end :].strip()
            if trailing:
                continue
            return value, None
        except Exception:
            continue
    return None, "response is not valid JSON"


def json_type_matches(value, expected):
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate_json_schema(value, schema, path="$"):
    if not isinstance(schema, dict):
        return []

    errors = []
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if not any(json_type_matches(value, item) for item in expected_type):
            errors.append(f"{path} has wrong type")
            return errors
    elif isinstance(expected_type, str) and not json_type_matches(value, expected_type):
        errors.append(f"{path} must be {expected_type}")
        return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")

    if isinstance(value, dict):
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")

        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(validate_json_schema(value[key], child_schema, f"{path}.{key}"))

        if schema.get("additionalProperties") is False:
            extras = sorted(set(value.keys()) - set(properties.keys()))
            for key in extras:
                errors.append(f"{path}.{key} is not allowed")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(validate_json_schema(item, schema["items"], f"{path}[{index}]"))

    return errors


def normalize_json_content(content, mode, schema):
    if mode == "text":
        return content, []

    value, error = extract_json_value(content)
    if error:
        return content, [error]
    if mode in {"json_object", "json_schema"} and not isinstance(value, dict):
        return content, ["response must be a JSON object"]

    errors = validate_json_schema(value, schema) if schema else []
    if errors:
        return content, errors

    return json.dumps(value, ensure_ascii=False, separators=(",", ":")), []


def generate_validated_completion(messages, max_tokens, temperature, mode, schema):
    contracted_messages = with_json_contract(messages, mode, schema)
    attempts = max(0, JSON_REPAIR_ATTEMPTS) + 1
    last_content = ""
    last_prompt_tokens = 0
    last_completion_tokens = 0
    last_errors = []

    for attempt in range(attempts):
        content, prompt_tokens, completion_tokens = generate_completion(
            contracted_messages,
            max_tokens,
            temperature,
        )
        normalized, errors = normalize_json_content(content, mode, schema)
        if not errors:
            return normalized, prompt_tokens, completion_tokens, []

        last_content = content
        last_prompt_tokens = prompt_tokens
        last_completion_tokens = completion_tokens
        last_errors = errors

        if attempt < attempts - 1:
            contracted_messages = contracted_messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "The previous response was invalid: "
                        + "; ".join(errors[:5])
                        + ". Return only a corrected JSON object."
                    ),
                },
            ]

    return last_content, last_prompt_tokens, last_completion_tokens, last_errors


class Handler(BaseHTTPRequestHandler):
    server_version = "TunedTensorReferenceServer/0.1"

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(204, {})

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "model": MODEL_NAME,
                "spec_prompt": bool(SYSTEM_PROMPT),
                "device": DEVICE,
            })
            return
        self.send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self):
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_json(404, {"error": {"message": "Not found"}})
            return

        try:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if body.get("stream"):
                self.send_json(400, {"error": {"message": "Streaming is not supported by tt models serve yet."}})
                return

            raw_messages = body.get("messages")
            if not isinstance(raw_messages, list) or not raw_messages:
                self.send_json(400, {"error": {"message": "Request must include a non-empty messages array."}})
                return

            max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
            temperature = float(body.get("temperature") if body.get("temperature") is not None else DEFAULT_TEMPERATURE)
            messages = normalize_messages(raw_messages)
            json_mode, json_schema = response_json_contract(body)
            content, prompt_tokens, completion_tokens, validation_errors = generate_validated_completion(
                messages,
                max_tokens,
                temperature,
                json_mode,
                json_schema,
            )
            if validation_errors:
                self.send_json(422, {
                    "error": {
                        "message": "Model did not produce valid JSON for the requested response_format.",
                        "details": validation_errors,
                        "content": content,
                    }
                })
                return

            self.send_json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
        except Exception as exc:
            self.send_json(500, {"error": {"message": str(exc)}})

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


def main():
    print(f"Serving {MODEL_NAME} from {MODEL_PATH}")
    print(f"Listening on http://{HOST}:{PORT}/v1/chat/completions")
    if SYSTEM_PROMPT:
        print("Behavior spec prompt: enabled")
    else:
        print("Behavior spec prompt: disabled")
    print(f"Device: {DEVICE}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
`;

function resolveOutputPath(output: string | undefined, filename: string): string {
  if (!output) return filename;
  if (existsSync(output) && statSync(output).isDirectory()) {
    return join(output, filename);
  }
  return output;
}

function parseContentLength(value: string | null): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const precision = unit === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[unit]}`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "--";
  const rounded = Math.max(1, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function chunkLength(chunk: unknown): number {
  if (typeof chunk === "string") return Buffer.byteLength(chunk);
  if (chunk instanceof Uint8Array) return chunk.byteLength;
  return Buffer.byteLength(String(chunk));
}

function createDownloadProgress(totalBytes: number | null, enabled: boolean) {
  const stream = process.stderr;
  const start = Date.now();
  let lastRender = 0;
  let rendered = false;

  function render(downloadedBytes: number, force = false) {
    if (!enabled) return;
    const now = Date.now();
    if (!force && now - lastRender < 100 && downloadedBytes !== totalBytes) return;
    lastRender = now;
    rendered = true;

    const elapsedSeconds = Math.max((now - start) / 1000, 0.001);
    const rate = downloadedBytes / elapsedSeconds;
    const rateText = `${formatBytes(rate)}/s`;

    let line: string;
    if (totalBytes != null && totalBytes > 0) {
      const ratio = Math.min(downloadedBytes / totalBytes, 1);
      const barWidth = 24;
      const filled = Math.round(ratio * barWidth);
      const bar = "#".repeat(filled) + "-".repeat(barWidth - filled);
      const etaSeconds = rate > 0 ? (totalBytes - downloadedBytes) / rate : Number.NaN;
      line = `Downloading [${bar}] ${(ratio * 100).toFixed(1).padStart(5)}% ${formatBytes(downloadedBytes)}/${formatBytes(totalBytes)} ${rateText} ETA ${formatDuration(etaSeconds)}`;
    } else {
      line = `Downloading ${formatBytes(downloadedBytes)} ${rateText} elapsed ${formatDuration(elapsedSeconds)}`;
    }

    const columns = stream.columns || 120;
    stream.write(`\r\x1b[K${line.slice(0, Math.max(columns - 1, 0))}`);
  }

  function stop() {
    if (enabled && rendered) stream.write("\n");
  }

  return { render, stop };
}

async function downloadUrlToFile(url: string, outputPath: string): Promise<number | null> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }
  if (!res.body) {
    throw new Error("Download failed: response body was empty");
  }

  mkdirSync(dirname(outputPath), { recursive: true });
  const file = createWriteStream(outputPath);
  const contentLength = parseContentLength(res.headers.get("content-length"));
  const progress = createDownloadProgress(
    contentLength,
    !isJsonMode() && process.stderr.isTTY === true,
  );
  let downloadedBytes = 0;
  const progressStream = new Transform({
    transform(chunk, _encoding, callback) {
      downloadedBytes += chunkLength(chunk);
      progress.render(downloadedBytes);
      callback(null, chunk);
    },
  });

  try {
    progress.render(0, true);
    await pipeline(Readable.fromWeb(res.body), progressStream, file);
    progress.render(downloadedBytes, true);
    progress.stop();
  } catch (err) {
    progress.stop();
    try {
      unlinkSync(outputPath);
    } catch {
      // Best effort cleanup for partially written downloads.
    }
    throw err;
  }

  return contentLength;
}

function defaultCacheDir(): string {
  const root = process.env.XDG_CACHE_HOME || join(homedir(), ".cache");
  return join(root, "tuned-tensor", "models");
}

function runtimeDir(cacheDir: string): string {
  return join(cacheDir, "_runtime");
}

function runtimePythonPath(cacheDir: string): string {
  const dir = runtimeDir(cacheDir);
  return process.platform === "win32"
    ? join(dir, "Scripts", "python.exe")
    : join(dir, "bin", "python");
}

function resolveServePython(cacheDir: string, explicitPython?: string): string {
  if (explicitPython) return explicitPython;
  const managedPython = runtimePythonPath(cacheDir);
  return existsSync(managedPython) ? managedPython : "python3";
}

function commandExists(command: string): boolean {
  try {
    execFileSync(command, ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function pythonMinorVersion(command: string): number | null {
  try {
    const out = execFileSync(
      command,
      ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] },
    ).trim();
    const [major, minor] = out.split(".").map((part) => Number(part));
    if (major !== 3 || !Number.isFinite(minor)) return null;
    return minor;
  } catch {
    return null;
  }
}

function pickRuntimePython(explicitPython?: string): string {
  const candidates = explicitPython
    ? [explicitPython]
    : ["python3.12", "python3.11", "python3.10", "python3"];
  for (const candidate of candidates) {
    if (!commandExists(candidate)) continue;
    const minor = pythonMinorVersion(candidate);
    if (minor != null && minor >= 10 && minor <= 12) return candidate;
    if (explicitPython) {
      throw new Error(
        `${candidate} is Python 3.${minor ?? "?"}. The local serving runtime requires Python 3.10, 3.11, or 3.12 because torch wheels may not be available for newer versions.`,
      );
    }
  }
  throw new Error(
    "Could not find Python 3.10, 3.11, or 3.12. Install one, then run `tt models setup-runtime --python <path>`.",
  );
}

function ensureServingRuntime(python: string, cacheDir: string): void {
  const check = `
import importlib.util
import json
required = ["torch", "transformers", "accelerate", "safetensors"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(json.dumps({"missing": missing}))
`;
  let missing: string[];
  try {
    const out = execFileSync(python, ["-c", check], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    missing = JSON.parse(out).missing;
  } catch (err) {
    const suffix = python === runtimePythonPath(cacheDir)
      ? " Re-run `tt models setup-runtime` to repair it."
      : " Run `tt models setup-runtime` to create an isolated serving runtime.";
    throw new Error(`Python serving runtime check failed for ${python}.${suffix}`);
  }

  if (missing.length > 0) {
    const managedPython = runtimePythonPath(cacheDir);
    const installHint = python === managedPython
      ? "Run `tt models setup-runtime --force` to repair it."
      : "Run `tt models setup-runtime` first, or pass --python <path> to a Python environment with the serving dependencies installed.";
    throw new Error(
      `Python serving runtime is missing: ${missing.join(", ")}. ${installHint}`,
    );
  }
}

function setupRuntimeCommands(python: string, venvDir: string, deps: string[]): string[][] {
  const venvPython = process.platform === "win32"
    ? join(venvDir, "Scripts", "python.exe")
    : join(venvDir, "bin", "python");
  return [
    [python, "-m", "venv", venvDir],
    [venvPython, "-m", "pip", "install", "--upgrade", "pip"],
    [venvPython, "-m", "pip", "install", ...deps],
  ];
}

function safeSegment(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

function directoryHasFiles(path: string): boolean {
  return existsSync(path) && statSync(path).isDirectory() && readdirSync(path).length > 0;
}

export function validateArchiveEntryPath(entry: string): string {
  const normalized = normalize(entry).replace(/^[\\/]+/, "");
  if (
    !entry.trim() ||
    isAbsolute(entry) ||
    normalized === ".." ||
    normalized.startsWith(`..${"/"}`) ||
    normalized.startsWith(`..${"\\"}`)
  ) {
    throw new Error(`Unsafe archive entry path: ${entry}`);
  }
  return normalized;
}

function validateArchiveEntries(archivePath: string) {
  const listing = execFileSync("tar", ["-tzf", archivePath], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  for (const entry of listing.split(/\r?\n/)) {
    if (entry.trim()) validateArchiveEntryPath(entry);
  }
}

function extractArchive(archivePath: string, targetDir: string, force: boolean): string {
  if (directoryHasFiles(targetDir) && !force) return targetDir;
  validateArchiveEntries(archivePath);
  if (existsSync(targetDir)) rmSync(targetDir, { recursive: true, force: true });
  mkdirSync(targetDir, { recursive: true });
  execFileSync("tar", ["-xzf", archivePath, "-C", targetDir], { stdio: "inherit" });
  return targetDir;
}

function localArchiveCacheDir(archivePath: string, cacheDir: string): string {
  const resolved = resolve(archivePath);
  const digest = createHash("sha256").update(resolved).digest("hex").slice(0, 12);
  return join(cacheDir, `local-${safeSegment(basename(archivePath))}-${digest}`);
}

function buildSystemPrompt(spec: Pick<LocalSpec, "system_prompt" | "guidelines" | "constraints">): string {
  const parts: string[] = [];
  if (spec.system_prompt?.trim()) parts.push(spec.system_prompt.trim());
  if (Array.isArray(spec.guidelines) && spec.guidelines.length > 0) {
    parts.push(`Guidelines:\n${spec.guidelines.map((g) => `- ${g}`).join("\n")}`);
  }
  if (Array.isArray(spec.constraints) && spec.constraints.length > 0) {
    parts.push(`Constraints:\n${spec.constraints.map((c) => `- ${c}`).join("\n")}`);
  }
  return parts.join("\n\n");
}

function findSpecPath(explicitPath: string | undefined, modelPath: string): string | null {
  const candidates = explicitPath
    ? [explicitPath]
    : [join(process.cwd(), DEFAULT_SPEC_FILE), join(modelPath, DEFAULT_SPEC_FILE)];
  for (const candidate of candidates) {
    const resolved = resolve(candidate);
    if (existsSync(resolved) && statSync(resolved).isFile()) return resolved;
  }
  return null;
}

function loadSystemPromptFromSpec(specPath: string): string {
  const spec = JSON.parse(readFileSync(specPath, "utf8")) as LocalSpec;
  return buildSystemPrompt(spec);
}

function loadJsonSchemaForServe(schemaPath: string): string {
  const resolved = resolve(schemaPath);
  if (!existsSync(resolved) || !statSync(resolved).isFile()) {
    throw new Error(`JSON schema file not found: ${schemaPath}`);
  }

  try {
    const schema = JSON.parse(readFileSync(resolved, "utf8"));
    return JSON.stringify(schema);
  } catch (err) {
    throw new Error(`JSON schema file is not valid JSON: ${schemaPath}`);
  }
}

function writeReferenceServerScript(cacheDir: string): string {
  const scriptDir = join(cacheDir, "_server");
  mkdirSync(scriptDir, { recursive: true });
  const scriptPath = join(scriptDir, "openai_reference_server.py");
  writeFileSync(scriptPath, REFERENCE_SERVER_SCRIPT, "utf8");
  return scriptPath;
}

async function resolveServeTarget(
  target: string,
  opts: ClientOpts,
  cacheDir: string,
  forceDownload: boolean,
): Promise<ServeTarget> {
  if (existsSync(target)) {
    const resolved = resolve(target);
    const stats = statSync(resolved);
    if (stats.isDirectory()) {
      return {
        modelPath: resolved,
        modelName: basename(resolved),
        source: "local-directory",
      };
    }
    if (stats.isFile() && (resolved.endsWith(".tar.gz") || resolved.endsWith(".tgz"))) {
      const extractDir = join(localArchiveCacheDir(resolved, cacheDir), "model");
      return {
        modelPath: extractArchive(resolved, extractDir, forceDownload),
        modelName: basename(resolved).replace(/\.t(ar\.)?gz$/, ""),
        source: "local-archive",
      };
    }
    throw new Error(`Unsupported model path: ${target}. Use a model directory or .tar.gz artifact.`);
  }

  const fullId = await resolveModelId(target, opts);
  const { data: model } = await get<Model>(`/models/${fullId}`, undefined, opts);
  const { data } = await get<ModelDownload>(`/models/${fullId}/download`, undefined, opts);
  const modelCacheDir = join(cacheDir, fullId);
  const archivePath = join(modelCacheDir, data.filename);
  const extractDir = join(modelCacheDir, "model");

  if (!existsSync(archivePath) || forceDownload) {
    await downloadUrlToFile(data.url, archivePath);
  }
  extractArchive(archivePath, extractDir, forceDownload);

  return {
    modelPath: extractDir,
    modelName: model.name || fullId,
    source: "downloaded-model",
    modelId: fullId,
  };
}

function quoteCommandPart(part: string): string {
  if (/^[a-zA-Z0-9_./:=@-]+$/.test(part)) return part;
  return `'${part.replace(/'/g, "'\\''")}'`;
}

function parseServeInteger(value: string | number, label: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer.`);
  }
  return parsed;
}

function printServeCommand(
  command: string,
  args: string[],
  env: Record<string, string>,
  managed?: {
    enabled: boolean;
    host: string;
    port: number;
    idle_timeout_seconds: number;
    restart_after_requests: number;
    gate_field: string;
    log_file?: string;
  },
) {
  const envKeys = Object.keys(env).sort();
  const commandLine = [command, ...args].map(quoteCommandPart).join(" ");
  if (isJsonMode()) {
    return printJson({ command, args, env_keys: envKeys, command_line: commandLine, managed });
  }
  const fields: [string, string | undefined][] = [
    ["Command", commandLine],
    ["Environment", envKeys.join(", ")],
  ];
  if (managed?.enabled) {
    fields.push(
      ["Managed", "enabled"],
      ["Wrapper", `http://${managed.host}:${managed.port}`],
      ["Idle timeout", `${managed.idle_timeout_seconds}s`],
      ["Restart after", managed.restart_after_requests === 0 ? "disabled" : `${managed.restart_after_requests} requests`],
      ["Gate field", managed.gate_field],
      ["Log file", managed.log_file],
    );
  }
  printDetail(fields);
}

// Quant types that `convert_hf_to_gguf.py` can emit directly via --outtype,
// so they skip the separate llama-quantize step.
const CONVERT_NATIVE_OUTTYPES = new Set([
  "f32",
  "f16",
  "bf16",
  "q8_0",
  "tq1_0",
  "tq2_0",
]);

// Quant types that require llama-quantize applied to an f16 intermediate.
const QUANTIZE_TYPES = new Set([
  "q4_0",
  "q4_1",
  "q5_0",
  "q5_1",
  "q2_k",
  "q2_k_s",
  "q3_k_s",
  "q3_k_m",
  "q3_k_l",
  "q4_k_s",
  "q4_k_m",
  "q5_k_s",
  "q5_k_m",
  "q6_k",
  "iq1_s",
  "iq1_m",
  "iq2_xxs",
  "iq2_xs",
  "iq2_s",
  "iq2_m",
  "iq3_xxs",
  "iq3_s",
  "iq3_m",
  "iq4_nl",
  "iq4_xs",
]);

interface QuantPlan {
  quant: string;
  requiresQuantize: boolean;
  convertOuttype: string;
  quantizeType?: string;
}

function planQuant(quant: string): QuantPlan {
  const lower = quant.trim().toLowerCase();
  if (!lower) {
    throw new Error("--quant requires a value (e.g. q4_k_m, q8_0, f16).");
  }
  if (CONVERT_NATIVE_OUTTYPES.has(lower)) {
    return { quant: lower, requiresQuantize: false, convertOuttype: lower };
  }
  if (QUANTIZE_TYPES.has(lower)) {
    return {
      quant: lower,
      requiresQuantize: true,
      convertOuttype: "f16",
      quantizeType: lower.toUpperCase(),
    };
  }
  const supported = [...CONVERT_NATIVE_OUTTYPES, ...QUANTIZE_TYPES].sort().join(", ");
  throw new Error(`Unsupported --quant "${quant}". Supported types: ${supported}.`);
}

// Ollama model tags must be lowercase and limited to [a-z0-9._-].
function ollamaSlug(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "");
  return slug || "model";
}

function firstExisting(candidates: string[]): string | null {
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate) && statSync(candidate).isFile()) {
      return resolve(candidate);
    }
  }
  return null;
}

function llamaCppDir(explicit?: string): string | undefined {
  return explicit || process.env.LLAMA_CPP_DIR || process.env.LLAMA_CPP_HOME || undefined;
}

function resolveConvertScript(
  opts: { convertScript?: string; llamaCpp?: string },
  required: boolean,
): string {
  if (opts.convertScript) {
    const resolved = resolve(opts.convertScript);
    if (required && !existsSync(resolved)) {
      throw new Error(`Conversion script not found: ${opts.convertScript}`);
    }
    return resolved;
  }

  const dir = llamaCppDir(opts.llamaCpp);
  const candidates = dir
    ? [join(dir, "convert_hf_to_gguf.py"), join(dir, "convert-hf-to-gguf.py")]
    : [];
  const found = firstExisting(candidates);
  if (found) return found;

  if (required) {
    throw new Error(
      "Could not find convert_hf_to_gguf.py. Pass --llama-cpp <dir> (a llama.cpp checkout) " +
        "or --convert-script <path>. Get it from https://github.com/ggml-org/llama.cpp.",
    );
  }
  return dir ? join(dir, "convert_hf_to_gguf.py") : "convert_hf_to_gguf.py";
}

function resolveQuantizeBin(
  opts: { quantizeBin?: string; llamaCpp?: string },
  required: boolean,
): string {
  if (opts.quantizeBin) {
    const resolved = resolve(opts.quantizeBin);
    if (required && !existsSync(resolved)) {
      throw new Error(`llama-quantize binary not found: ${opts.quantizeBin}`);
    }
    return resolved;
  }

  const dir = llamaCppDir(opts.llamaCpp);
  if (dir) {
    const candidates = [
      join(dir, "build", "bin", "llama-quantize"),
      join(dir, "build", "bin", "quantize"),
      join(dir, "llama-quantize"),
      join(dir, "quantize"),
    ];
    const found = firstExisting(candidates);
    if (found) return found;
  }

  if (required && !commandExists("llama-quantize")) {
    throw new Error(
      "Could not find the llama-quantize binary. Build llama.cpp and pass --llama-cpp <dir> " +
        "or --quantize-bin <path>, or add llama-quantize to your PATH.",
    );
  }
  return "llama-quantize";
}

function buildModelfile(ggufBasename: string, systemPrompt: string): string {
  const lines = [`FROM ./${ggufBasename}`];
  if (systemPrompt.trim()) {
    const escaped = systemPrompt.trim().replace(/"""/g, '\\"\\"\\"');
    lines.push("", `SYSTEM """${escaped}"""`);
  }
  return lines.join("\n") + "\n";
}

interface ExportStep {
  name: "convert" | "quantize" | "ollama-create";
  command: string;
  args: string[];
  command_line: string;
}

function buildExportStep(name: ExportStep["name"], command: string, args: string[]): ExportStep {
  return {
    name,
    command,
    args,
    command_line: [command, ...args].map(quoteCommandPart).join(" "),
  };
}

interface ExportPlan {
  format: "gguf";
  quant: string;
  model: { name: string; path: string; source: ServeTarget["source"]; id?: string };
  output_dir: string;
  gguf_path: string;
  intermediate_path?: string;
  steps: ExportStep[];
  ollama?: {
    name: string;
    modelfile_path: string;
    modelfile: string;
    create: boolean;
  };
}

function printExportPlan(plan: ExportPlan) {
  if (isJsonMode()) return printJson(plan);

  printDetail([
    ["Model", `${plan.model.name} (${plan.model.source})`],
    ["Format", plan.format],
    ["Quant", plan.quant],
    ["Output dir", plan.output_dir],
    ["GGUF", plan.gguf_path],
  ]);
  console.log("\nPlanned steps:");
  for (const step of plan.steps) {
    console.log(`  ${step.name}: ${step.command_line}`);
  }
  if (plan.ollama) {
    console.log(`\nModelfile (${plan.ollama.modelfile_path}):`);
    console.log(plan.ollama.modelfile.replace(/^/gm, "  "));
  }
}

function printOpenClawHints(ollamaName: string) {
  console.log("\nUse it with OpenClaw via Ollama's native /api/chat (not /v1):");
  console.log(
    [
      "  {",
      "    models: { providers: { ollama: {",
      '      api: "ollama",',
      '      baseUrl: "http://127.0.0.1:11434",',
      `      models: [{ id: "${ollamaName}" }]`,
      "    }}}",
      "  }",
    ].join("\n"),
  );
  console.log("\nThen run it through the infer surface:");
  console.log(
    `  openclaw infer model run --local --model ollama/${ollamaName} \\\n` +
      '    --prompt "<payload>" --json',
  );
}

export function registerModelsCommands(parent: Command) {
  const models = parent.command("models").description("Manage fine-tuned models");

  models
    .command("base")
    .description("List supported base models")
    .action(async () => {
      const data = SUPPORTED_BASE_MODELS.map((model) => ({
        id: model,
        name: model.split("/").pop() || model,
        type: "base" as const,
      }));

      if (isJsonMode()) return printJson({ data });

      printTable(
        ["ID", "Name", "Type"],
        data.map((m) => [m.id, m.name, m.type]),
      );
    });

  models
    .command("list")
    .description("List fine-tuned models")
    .option("-p, --page <n>", "Page number", "1")
    .option("--per-page <n>", "Results per page", "20")
    .action(async (cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const { data, meta } = await get<Model[]>(
        "/models",
        { page: cmdOpts.page, per_page: cmdOpts.perPage },
        opts,
      );

      if (isJsonMode()) return printJson({ data, meta });

      printTable(
        ["ID", "Name", "Base Model", "Provider", "Created"],
        data.map((m) => [
          shortId(m.id),
          truncate(m.name, 30),
          m.base_model.split("/").pop() || m.base_model,
          m.provider,
          formatDate(m.created_at),
        ]),
        meta,
      );
    });

  models
    .command("get")
    .description("Show model details")
    .argument("<id>", "Model ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveModelId(id, opts);
      const { data } = await get<Model>(`/models/${fullId}`, undefined, opts);

      if (isJsonMode()) return printJson(data);

      printDetail([
        ["ID", data.id],
        ["Name", data.name],
        ["Base Model", data.base_model],
        ["Provider", data.provider],
        ["Hosted Model", data.provider_model_id],
        ["Description", data.description ?? undefined],
        ["Created", formatDate(data.created_at)],
      ]);
    });

  models
    .command("download")
    .description("Download a fine-tuned model artifact")
    .argument("<id>", "Model ID (full UUID or 4+ char prefix)")
    .option("-o, --output <path>", "Output file or directory")
    .option("-f, --force", "Overwrite the output file if it already exists")
    .action(async (id: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveModelId(id, opts);
      const { data } = await get<ModelDownload>(
        `/models/${fullId}/download`,
        undefined,
        opts,
      );

      const outputPath = resolveOutputPath(cmdOpts.output, data.filename);
      if (existsSync(outputPath) && !cmdOpts.force) {
        throw new Error(`Output file already exists: ${outputPath}. Use --force to overwrite.`);
      }

      const bytes = await downloadUrlToFile(data.url, outputPath);

      if (isJsonMode()) {
        return printJson({
          id: fullId,
          output_path: outputPath,
          filename: data.filename,
          bytes,
          expires_at: data.expires_at,
        });
      }

      const size = bytes == null ? "" : ` (${bytes} bytes)`;
      printSuccess(`Model downloaded to ${outputPath}${size}`);
    });

  models
    .command("export")
    .description("Export a fine-tuned model to GGUF and (optionally) package it for Ollama")
    .argument("<target>", "Model ID/prefix, model directory, or .tar.gz artifact")
    .option("--format <format>", "Export format", "gguf")
    .option("--quant <type>", "Quantization type (e.g. q4_k_m, q8_0, f16)", "q4_k_m")
    .option("-o, --output <dir>", "Output directory for the GGUF (and Modelfile)")
    .option("--ollama", "Also write an Ollama Modelfile and run `ollama create`")
    .option("--ollama-name <name>", "Ollama model name (default: tt-<slug>)")
    .option("--no-ollama-create", "With --ollama, write the Modelfile but skip `ollama create`")
    .option("--spec <path>", "Behavior spec JSON to embed as the Ollama SYSTEM prompt")
    .option("--no-spec-prompt", "Do not embed a behavior spec system prompt in the Modelfile")
    .option("--llama-cpp <dir>", "Path to a llama.cpp checkout/build (convert script + quantize binary)")
    .option("--convert-script <path>", "Path to convert_hf_to_gguf.py")
    .option("--quantize-bin <path>", "Path to the llama-quantize binary")
    .option("--ollama-bin <path>", "Path to the ollama binary", "ollama")
    .option("--python <path>", "Python executable to run the conversion script", "python3")
    .option("--cache-dir <path>", "Cache directory for downloaded/extracted models")
    .option("--force-download", "Re-download and re-extract model artifacts")
    .option("-f, --force", "Overwrite existing GGUF/Modelfile outputs")
    .option("--keep-intermediate", "Keep the intermediate f16 GGUF after quantization")
    .option("--print-command", "Print the planned commands without executing them")
    .action(async (target: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;

      const format = String(cmdOpts.format).toLowerCase();
      if (format !== "gguf") {
        throw new Error(`Unsupported --format "${cmdOpts.format}". Only "gguf" is supported.`);
      }
      const quantPlan = planQuant(String(cmdOpts.quant));
      const printOnly = Boolean(cmdOpts.printCommand);

      const cacheDir = resolve(cmdOpts.cacheDir || defaultCacheDir());
      mkdirSync(cacheDir, { recursive: true });
      const serveTarget = await resolveServeTarget(
        target,
        opts,
        cacheDir,
        Boolean(cmdOpts.forceDownload),
      );

      const slug = ollamaSlug(serveTarget.modelName);
      const outputDir = resolve(cmdOpts.output || join(process.cwd(), `${slug}-gguf`));
      const finalPath = join(outputDir, `${slug}.${quantPlan.quant}.gguf`);
      const intermediatePath = quantPlan.requiresQuantize
        ? join(outputDir, `${slug}.f16.gguf`)
        : undefined;

      const convertScript = resolveConvertScript(cmdOpts, !printOnly);
      const convertOutfile = intermediatePath ?? finalPath;
      const steps: ExportStep[] = [
        buildExportStep("convert", cmdOpts.python, [
          convertScript,
          serveTarget.modelPath,
          "--outfile",
          convertOutfile,
          "--outtype",
          quantPlan.convertOuttype,
        ]),
      ];

      let quantizeBin: string | undefined;
      if (quantPlan.requiresQuantize) {
        quantizeBin = resolveQuantizeBin(cmdOpts, !printOnly);
        steps.push(
          buildExportStep("quantize", quantizeBin, [
            intermediatePath as string,
            finalPath,
            quantPlan.quantizeType as string,
          ]),
        );
      }

      const wantOllama = Boolean(cmdOpts.ollama);
      const ollamaName = cmdOpts.ollamaName || `tt-${slug}`;
      let ollamaInfo: ExportPlan["ollama"];
      let modelfileContent = "";
      let modelfilePath = "";
      if (wantOllama) {
        let systemPrompt = "";
        if (cmdOpts.specPrompt !== false) {
          const specPath = findSpecPath(cmdOpts.spec, serveTarget.modelPath);
          if (cmdOpts.spec && !specPath) {
            throw new Error(`Spec file not found: ${cmdOpts.spec}`);
          }
          if (specPath) systemPrompt = loadSystemPromptFromSpec(specPath);
        }
        modelfilePath = join(outputDir, "Modelfile");
        modelfileContent = buildModelfile(basename(finalPath), systemPrompt);
        const createModel = cmdOpts.ollamaCreate !== false;
        if (createModel) {
          steps.push(
            buildExportStep("ollama-create", cmdOpts.ollamaBin, [
              "create",
              ollamaName,
              "-f",
              modelfilePath,
            ]),
          );
        }
        ollamaInfo = {
          name: ollamaName,
          modelfile_path: modelfilePath,
          modelfile: modelfileContent,
          create: createModel,
        };
      }

      const plan: ExportPlan = {
        format: "gguf",
        quant: quantPlan.quant,
        model: {
          name: serveTarget.modelName,
          path: serveTarget.modelPath,
          source: serveTarget.source,
          id: serveTarget.modelId,
        },
        output_dir: outputDir,
        gguf_path: finalPath,
        intermediate_path: intermediatePath,
        steps,
        ollama: ollamaInfo,
      };

      if (printOnly) return printExportPlan(plan);

      if (existsSync(finalPath) && !cmdOpts.force) {
        throw new Error(`Output already exists: ${finalPath}. Use --force to overwrite.`);
      }

      mkdirSync(outputDir, { recursive: true });

      const runStep = (command: string, args: string[]) =>
        execFileSync(command, args, { stdio: "inherit" });

      if (!isJsonMode()) printSuccess(`Converting ${serveTarget.modelName} → ${quantPlan.convertOuttype} GGUF`);
      runStep(cmdOpts.python, [
        convertScript,
        serveTarget.modelPath,
        "--outfile",
        convertOutfile,
        "--outtype",
        quantPlan.convertOuttype,
      ]);

      if (quantPlan.requiresQuantize && quantizeBin) {
        if (!isJsonMode()) printSuccess(`Quantizing → ${quantPlan.quantizeType}`);
        runStep(quantizeBin, [
          intermediatePath as string,
          finalPath,
          quantPlan.quantizeType as string,
        ]);
        if (!cmdOpts.keepIntermediate && intermediatePath && existsSync(intermediatePath)) {
          rmSync(intermediatePath, { force: true });
        }
      }

      if (ollamaInfo) {
        writeFileSync(modelfilePath, modelfileContent, "utf8");
        if (!isJsonMode()) printSuccess(`Wrote Modelfile to ${modelfilePath}`);
        if (ollamaInfo.create) {
          if (!isJsonMode()) printSuccess(`Creating Ollama model ${ollamaName}`);
          runStep(cmdOpts.ollamaBin, ["create", ollamaName, "-f", modelfilePath]);
        }
      }

      if (isJsonMode()) {
        return printJson({
          format: "gguf",
          quant: quantPlan.quant,
          output_dir: outputDir,
          gguf_path: finalPath,
          ollama: ollamaInfo
            ? { name: ollamaName, modelfile_path: modelfilePath, created: ollamaInfo.create }
            : undefined,
        });
      }

      printSuccess(`GGUF written to ${finalPath}`);
      if (ollamaInfo?.create) {
        console.log(`Run it: ollama run ${ollamaName}`);
        printOpenClawHints(ollamaName);
      } else if (ollamaInfo) {
        console.log(`Create the Ollama model: ollama create ${ollamaName} -f ${modelfilePath}`);
      }
    });

  models
    .command("setup-runtime")
    .description("Install an isolated Python runtime for local model serving")
    .option("--python <path>", "Python 3.10-3.12 executable to create the runtime")
    .option("--cache-dir <path>", "Cache directory for the managed runtime")
    .option("-f, --force", "Recreate the runtime if it already exists")
    .option("--print-command", "Print the setup commands without running them")
    .action(async (cmdOpts) => {
      const cacheDir = resolve(cmdOpts.cacheDir || defaultCacheDir());
      const venvDir = runtimeDir(cacheDir);
      const venvPython = runtimePythonPath(cacheDir);
      const python = pickRuntimePython(cmdOpts.python);
      const deps = ["torch", "transformers", "accelerate", "safetensors"];
      const commands = setupRuntimeCommands(python, venvDir, deps);

      if (cmdOpts.printCommand) {
        const commandLines = commands.map((cmd) => cmd.map(quoteCommandPart).join(" "));
        if (isJsonMode()) {
          return printJson({
            python,
            runtime_dir: venvDir,
            runtime_python: venvPython,
            commands: commandLines,
          });
        }
        printDetail([
          ["Python", python],
          ["Runtime", venvDir],
          ["Commands", commandLines.join("\n")],
        ]);
        return;
      }

      if (existsSync(venvDir) && cmdOpts.force) {
        rmSync(venvDir, { recursive: true, force: true });
      }
      if (!existsSync(venvPython)) {
        mkdirSync(cacheDir, { recursive: true });
        execFileSync(commands[0][0], commands[0].slice(1), { stdio: "inherit" });
      }

      execFileSync(commands[1][0], commands[1].slice(1), { stdio: "inherit" });
      execFileSync(commands[2][0], commands[2].slice(1), { stdio: "inherit" });
      ensureServingRuntime(venvPython, cacheDir);

      if (isJsonMode()) {
        return printJson({
          runtime_dir: venvDir,
          runtime_python: venvPython,
          installed: true,
          dependencies: deps,
        });
      }

      printSuccess(`Serving runtime ready at ${venvDir}`);
      console.log(`Use it with: tt models serve <model-id> --spec ${DEFAULT_SPEC_FILE}`);
    });

  models
    .command("serve")
    .description("Serve a downloaded model with an OpenAI-compatible local API")
    .argument("<target>", "Model ID/prefix, model directory, or .tar.gz artifact")
    .option("--spec <path>", "Behavior spec JSON to apply as the default system prompt")
    .option("--no-spec-prompt", "Do not auto-apply a behavior spec system prompt")
    .option("--host <host>", "Host to bind", "127.0.0.1")
    .option("--port <port>", "Port to bind", "8000")
    .option("--python <path>", "Python executable")
    .option("--cache-dir <path>", "Cache directory for downloaded/extracted models")
    .option("--device <device>", "Inference device: auto, cpu, cuda, or mps", "auto")
    .option("--force-download", "Re-download and re-extract model artifacts")
    .option("--max-tokens <n>", "Default max completion tokens", "512")
    .option("--temperature <n>", "Default sampling temperature", "0.7")
    .option("--json-schema <path>", "JSON Schema file to enforce by default for chat completions")
    .option("--json-repair-attempts <n>", "Number of JSON/schema repair retries", "1")
    .option("--trust-remote-code", "Pass trust_remote_code=True to transformers")
    .option("--managed", "Run a local lifecycle manager in front of the model server")
    .option("--idle-timeout <seconds>", "Managed mode idle timeout before stopping the model", "300")
    .option("--restart-after-requests <n>", "Managed mode restart threshold; 0 disables request-count restarts", "100")
    .option("--gate-field <field>", "Response JSON field to log as the managed serving gate result", "should_process")
    .option("--log-file <path>", "Write managed serving JSONL request logs to a file")
    .option("--print-command", "Print the underlying Python command without starting it")
    .action(async (target: string, cmdOpts) => {
      const opts = parent.opts() as ClientOpts;
      const cacheDir = resolve(cmdOpts.cacheDir || defaultCacheDir());
      mkdirSync(cacheDir, { recursive: true });
      const python = resolveServePython(cacheDir, cmdOpts.python);
      if (!cmdOpts.printCommand) ensureServingRuntime(python, cacheDir);

      const serveTarget = await resolveServeTarget(
        target,
        opts,
        cacheDir,
        Boolean(cmdOpts.forceDownload),
      );
      const serverScript = writeReferenceServerScript(cacheDir);

      let specPath: string | null = null;
      let systemPrompt = "";
      if (cmdOpts.specPrompt !== false) {
        specPath = findSpecPath(cmdOpts.spec, serveTarget.modelPath);
        if (cmdOpts.spec && !specPath) {
          throw new Error(`Spec file not found: ${cmdOpts.spec}`);
        }
        if (specPath) {
          systemPrompt = loadSystemPromptFromSpec(specPath);
        }
      }

      const env: Record<string, string> = {
        TT_MODEL_PATH: serveTarget.modelPath,
        TT_MODEL_NAME: serveTarget.modelName,
        TT_HOST: cmdOpts.host,
        TT_PORT: String(cmdOpts.port),
        TT_MAX_TOKENS: String(cmdOpts.maxTokens),
        TT_TEMPERATURE: String(cmdOpts.temperature),
        TT_JSON_REPAIR_ATTEMPTS: String(cmdOpts.jsonRepairAttempts),
        TT_TRUST_REMOTE_CODE: cmdOpts.trustRemoteCode ? "true" : "false",
        TT_DEVICE: String(cmdOpts.device),
      };
      if (systemPrompt) env.TT_SYSTEM_PROMPT = systemPrompt;
      if (cmdOpts.jsonSchema) env.TT_JSON_SCHEMA = loadJsonSchemaForServe(cmdOpts.jsonSchema);

      const args = [serverScript];
      const idleTimeoutSeconds = parseServeInteger(cmdOpts.idleTimeout, "--idle-timeout");
      const restartAfterRequests = parseServeInteger(
        cmdOpts.restartAfterRequests,
        "--restart-after-requests",
      );
      const publicPort = parseServeInteger(cmdOpts.port, "--port");
      const managedConfig = cmdOpts.managed
        ? {
            enabled: true,
            host: String(cmdOpts.host),
            port: publicPort,
            idle_timeout_seconds: idleTimeoutSeconds,
            restart_after_requests: restartAfterRequests,
            gate_field: String(cmdOpts.gateField),
            log_file: cmdOpts.logFile ? resolve(cmdOpts.logFile) : undefined,
          }
        : undefined;
      if (cmdOpts.printCommand) return printServeCommand(python, args, env, managedConfig);

      if (!isJsonMode()) {
        printSuccess(
          cmdOpts.managed
            ? `Serving ${serveTarget.modelName} through a managed local wrapper`
            : `Serving ${serveTarget.modelName} from ${serveTarget.modelPath}`,
        );
        if (systemPrompt && specPath) {
          printSuccess(`Behavior spec prompt loaded from ${specPath}`);
        } else if (cmdOpts.specPrompt !== false) {
          printWarning(
            `No ${DEFAULT_SPEC_FILE} found. Pass --spec <path> to preserve the behavior prompt.`,
          );
        }
        console.log(`OpenAI-compatible endpoint: http://${cmdOpts.host}:${cmdOpts.port}/v1/chat/completions`);
        console.log(`Health check: http://${cmdOpts.host}:${cmdOpts.port}/health`);
        console.log(`Python runtime: ${python}`);
        if (cmdOpts.managed) {
          console.log(`Managed idle timeout: ${idleTimeoutSeconds}s`);
          console.log(
            `Managed restart threshold: ${
              restartAfterRequests === 0 ? "disabled" : `${restartAfterRequests} requests`
            }`,
          );
          console.log(`Managed gate field: ${cmdOpts.gateField}`);
        }
      }

      if (cmdOpts.managed) {
        await startManagedModelServer({
          publicHost: String(cmdOpts.host),
          publicPort,
          python,
          args,
          env,
          modelName: serveTarget.modelName,
          modelPath: serveTarget.modelPath,
          idleTimeoutSeconds,
          restartAfterRequests,
          gateField: String(cmdOpts.gateField),
          logFile: cmdOpts.logFile ? resolve(cmdOpts.logFile) : undefined,
        });
        return;
      }

      await new Promise<void>((resolvePromise, reject) => {
        const child = spawn(python, args, {
          stdio: "inherit",
          env: { ...process.env, ...env },
        });
        child.on("error", reject);
        child.on("exit", (code, signal) => {
          if (code === 0) return resolvePromise();
          reject(new Error(`Model server exited with ${signal ?? `code ${code}`}`));
        });
      });
    });

  models
    .command("delete")
    .description("Delete a model")
    .argument("<id>", "Model ID (full UUID or 4+ char prefix)")
    .action(async (id: string) => {
      const opts = parent.opts() as ClientOpts;
      const fullId = await resolveModelId(id, opts);
      await del(`/models/${fullId}`, opts);

      if (isJsonMode()) return printJson({ id: fullId, deleted: true });
      printSuccess(`Model deleted: ${shortId(fullId)}`);
    });
}
