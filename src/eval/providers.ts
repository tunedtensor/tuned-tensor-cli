import type { ProviderConfig } from "./types.js";

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface ChatResponse {
  content: string;
  latency_ms: number;
}

const DEFAULT_OLLAMA_URL = "http://localhost:11434";
const DEFAULT_OPENAI_URL = "https://api.openai.com";

function resolveBaseUrl(config: ProviderConfig): string {
  if (config.baseUrl) return config.baseUrl;
  switch (config.provider) {
    case "ollama":
      return process.env.OLLAMA_HOST || DEFAULT_OLLAMA_URL;
    case "openai":
      return DEFAULT_OPENAI_URL;
    case "custom":
      throw new Error("Custom provider requires --base-url");
  }
}

function resolveApiKey(config: ProviderConfig): string | undefined {
  if (config.apiKey) return config.apiKey;
  switch (config.provider) {
    case "ollama":
      return undefined;
    case "openai":
      return process.env.OPENAI_API_KEY;
    case "custom":
      return process.env.PROVIDER_API_KEY;
  }
}

export async function chatCompletion(
  messages: ChatMessage[],
  config: ProviderConfig,
): Promise<ChatResponse> {
  const baseUrl = resolveBaseUrl(config);
  const apiKey = resolveApiKey(config);
  const url = `${baseUrl}/v1/chat/completions`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  const start = performance.now();
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: config.model,
      messages,
      temperature: 0.7,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    if (config.provider === "ollama" && (res.status === 0 || text.includes("ECONNREFUSED"))) {
      throw new Error(
        `Cannot connect to Ollama at ${baseUrl}. Is it running? Start with: ollama serve`,
      );
    }
    throw new Error(
      `${config.provider} API error [${res.status}]: ${text}`,
    );
  }

  const json = await res.json() as {
    choices: { message: { content: string } }[];
  };
  const latency_ms = Math.round(performance.now() - start);

  return {
    content: json.choices[0]?.message?.content ?? "",
    latency_ms,
  };
}

export async function checkProviderAvailability(
  config: ProviderConfig,
): Promise<{ available: boolean; error?: string }> {
  try {
    const baseUrl = resolveBaseUrl(config);

    if (config.provider === "ollama") {
      const res = await fetch(`${baseUrl}/api/tags`, {
        signal: AbortSignal.timeout(3000),
      });
      if (!res.ok) return { available: false, error: "Ollama is not responding" };
      const data = await res.json() as { models?: { name: string }[] };
      const models = data.models?.map((m) => m.name) ?? [];
      const hasModel = models.some(
        (m) => m === config.model || m.startsWith(config.model + ":"),
      );
      if (!hasModel) {
        return {
          available: false,
          error: `Model "${config.model}" not found in Ollama. Pull it with: ollama pull ${config.model}`,
        };
      }
    }

    return { available: true };
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("ECONNREFUSED") || msg.includes("fetch failed")) {
      return {
        available: false,
        error: `Cannot connect to ${config.provider}. Is it running?`,
      };
    }
    return { available: false, error: msg };
  }
}
