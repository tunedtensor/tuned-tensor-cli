import { readFileSync, statSync } from "node:fs";
import { basename } from "node:path";
import { getApiKey, getBaseUrl } from "./config.js";

export interface ApiResponse<T> {
  data: T;
  meta?: { page: number; per_page: number; total: number };
}

export interface ApiErrorBody {
  error: { code: string; message: string };
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface ClientOpts {
  apiKey?: string;
  baseUrl?: string;
}

function resolveKey(opts?: ClientOpts): string {
  const key = getApiKey(opts);
  if (!key) {
    throw new ApiError(
      401,
      "NO_API_KEY",
      "No API key found. Run `tt auth login` or set TUNED_TENSOR_API_KEY.",
    );
  }
  return key;
}

function buildUrl(
  path: string,
  query?: Record<string, string | number | undefined>,
  opts?: ClientOpts,
): string {
  const base = getBaseUrl(opts);
  const url = new URL(`/api/v1${path}`, base);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  return url.toString();
}

async function request<T>(
  method: string,
  path: string,
  options: {
    query?: Record<string, string | number | undefined>;
    body?: unknown;
    opts?: ClientOpts;
  } = {},
): Promise<ApiResponse<T>> {
  const key = resolveKey(options.opts);
  const url = buildUrl(path, options.query, options.opts);

  const headers: Record<string, string> = {
    Authorization: `Bearer ${key}`,
  };

  let bodyPayload: string | undefined;
  if (options.body) {
    headers["Content-Type"] = "application/json";
    bodyPayload = JSON.stringify(options.body);
  }

  const res = await fetch(url, { method, headers, body: bodyPayload });

  if (!res.ok) {
    let errBody: ApiErrorBody;
    try {
      errBody = (await res.json()) as ApiErrorBody;
    } catch {
      throw new ApiError(res.status, "UNKNOWN", res.statusText);
    }
    throw new ApiError(
      res.status,
      errBody.error?.code || "UNKNOWN",
      errBody.error?.message || res.statusText,
    );
  }

  if (res.status === 204) return { data: null as T };
  return (await res.json()) as ApiResponse<T>;
}

export function get<T>(
  path: string,
  query?: Record<string, string | number | undefined>,
  opts?: ClientOpts,
) {
  return request<T>("GET", path, { query, opts });
}

export function post<T>(
  path: string,
  body?: unknown,
  opts?: ClientOpts,
) {
  return request<T>("POST", path, { body, opts });
}

export function put<T>(
  path: string,
  body?: unknown,
  opts?: ClientOpts,
) {
  return request<T>("PUT", path, { body, opts });
}

export function patch<T>(
  path: string,
  body?: unknown,
  opts?: ClientOpts,
) {
  return request<T>("PATCH", path, { body, opts });
}

export function del<T>(path: string, opts?: ClientOpts) {
  return request<T>("DELETE", path, { opts });
}

export async function upload<T>(
  path: string,
  filePath: string,
  fields: Record<string, string>,
  opts?: ClientOpts,
): Promise<ApiResponse<T>> {
  const key = resolveKey(opts);
  const url = buildUrl(path, undefined, opts);

  const form = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    form.append(k, v);
  }

  const fileBytes = readFileSync(filePath);
  const fileSize = statSync(filePath).size;
  const blob = new Blob([fileBytes], { type: "application/jsonl" });
  form.append("file", blob, basename(filePath));

  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}` },
    body: form,
  });

  if (!res.ok) {
    let errBody: ApiErrorBody;
    try {
      errBody = (await res.json()) as ApiErrorBody;
    } catch {
      throw new ApiError(res.status, "UNKNOWN", res.statusText);
    }
    throw new ApiError(
      res.status,
      errBody.error?.code || "UNKNOWN",
      errBody.error?.message || res.statusText,
    );
  }

  return (await res.json()) as ApiResponse<T>;
}
