import {
  formatDate,
  formatStatus,
  printDetail,
  printError,
  printJson,
  printSuccess,
  printTable,
  printWarning,
  shortId,
  truncate,
} from "./output.js";

export interface LocalOutputPayload {
  args: string[];
  exitCode: number;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  hasJson: boolean;
  json: unknown;
  streamingStdout: boolean;
  droppedSpecKeys: string[];
  errorMessage?: string;
}

export interface RenderLocalOutputOptions {
  jsonMode?: boolean;
  stdout?: NodeJS.WritableStream;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function text(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return isRecord(value) ? value : undefined;
}

function rows(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function displayId(value: unknown): string {
  const id = text(value);
  return id ? shortId(id) : "—";
}

function displayNumber(value: unknown, digits = 4): string | undefined {
  const parsed = number(value);
  if (parsed === undefined) return undefined;
  return parsed.toFixed(digits).replace(/\.?0+$/, "");
}

function displayBytes(value: unknown): string | undefined {
  const bytes = number(value);
  if (bytes === undefined || bytes < 0) return undefined;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = bytes;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  const precision = amount >= 10 || unit === 0 ? 0 : 1;
  return `${amount.toFixed(precision)} ${units[unit]}`;
}

function displayCommand(value: unknown): string | undefined {
  if (!Array.isArray(value) || !value.every((part) => typeof part === "string")) {
    return undefined;
  }
  return value.map((part) =>
    /[\s"'\\]/.test(part) ? JSON.stringify(part) : part
  ).join(" ");
}

function status(value: unknown): string {
  const parsed = text(value) ?? "unknown";
  return formatStatus(parsed);
}

function gateFrom(value: unknown): Record<string, unknown> | undefined {
  return record(record(value)?.general_regression);
}

function renderGate(value: unknown): void {
  const gate = gateFrom(value);
  if (!gate) {
    printWarning(
      "General regression was not run; this model cannot be activated yet.",
    );
    return;
  }
  if (gate.passed === true) {
    printSuccess("General regression gate passed");
    return;
  }

  printWarning(
    "The local run completed, but the general regression gate failed.",
  );
  const failures = Array.isArray(gate.failures)
    ? gate.failures.filter((failure): failure is string =>
      typeof failure === "string"
    )
    : [];
  for (const failure of failures) printWarning(failure);
}

function renderDoctor(value: Record<string, unknown>): void {
  const checks = rows(value.checks);
  printTable(
    ["Check", "Status", "Details"],
    checks.map((check) => [
      text(check.name) ?? "unknown",
      check.ok === true ? formatStatus("completed") : formatStatus("failed"),
      truncate(text(check.message) ?? text(check.detail) ?? "—", 100),
    ]),
  );
  if (value.ok === true) {
    printSuccess("Local environment is ready");
  } else {
    printError("Local environment needs attention");
  }
  if (value.config_path) {
    printDetail([["Config", text(value.config_path)]]);
  }
}

function renderInit(value: Record<string, unknown>): void {
  printSuccess("Created local Tuned Tensor project");
  printDetail([
    ["Path", text(value.path)],
    ["Name", text(value.name)],
    ["ID", text(value.id)],
    ["Base model", text(value.base_model)],
    ["Config", text(value.config_path)],
  ]);
}

function renderValidate(value: Record<string, unknown>): void {
  printSuccess("Local behavior spec is valid");
  printDetail([
    ["Input", text(value.input_path)],
    ["Config", text(value.config_path)],
    ["Spec", text(value.behavior_spec_id)],
    ["Base model", text(value.base_model)],
    ["Dataset", text(value.dataset_format)],
    ["Artifacts", text(value.artifact_root)],
    ["Store", text(value.store_root)],
    ["Dry run", text(value.dry_run)],
  ]);
}

function renderRun(value: Record<string, unknown>): void {
  const runStatus = text(value.status) ?? "unknown";
  if (runStatus === "completed") {
    printSuccess("Local run completed");
  } else {
    printWarning(`Local run finished with status ${runStatus}`);
  }
  const comparison = record(value.comparison);
  printDetail([
    ["Run", text(value.run_id)],
    ["Status", status(runStatus)],
    ["Model", text(value.model_id) ?? text(value.fine_tuned_model_id)],
    ["Report", text(value.report_path)],
    ["Artifacts", text(value.artifact_dir)],
    ["Score delta", displayNumber(comparison?.avg_score_delta)],
    ["Pass-rate delta", displayNumber(comparison?.pass_rate_delta)],
  ]);
  renderGate(value);
}

function renderRunsList(value: unknown): void {
  const runRows = rows(value);
  if (runRows.length === 0) {
    printWarning("No local runs found.");
    return;
  }
  printTable(
    ["Run", "Spec", "Status", "Stage", "Model", "Updated"],
    runRows.map((run) => [
      displayId(run.id),
      truncate(text(run.spec_name) ?? displayId(run.behavior_spec_id), 28),
      status(run.status),
      text(run.current_stage) ?? "—",
      displayId(run.model_id),
      formatDate(text(run.updated_at)),
    ]),
  );
}

function renderRunEvents(value: unknown): void {
  const eventRows = rows(value);
  if (eventRows.length === 0) {
    printWarning("No local run events found.");
    return;
  }
  printTable(
    ["Time", "Stage", "Status", "Message"],
    eventRows.map((event) => [
      formatDate(text(event.occurred_at)),
      text(event.stage) ?? "—",
      status(event.status),
      truncate(text(event.message) ?? "—", 80),
    ]),
  );
}

function renderRunRecord(value: Record<string, unknown>): void {
  printDetail([
    ["Run", text(value.id) ?? text(value.run_id)],
    ["Spec", text(value.spec_name) ?? text(value.behavior_spec_id)],
    ["Status", status(value.status)],
    ["Stage", text(value.current_stage)],
    ["Model", text(value.model_id) ?? text(value.fine_tuned_model_id)],
    ["Base model", text(value.base_model)],
    ["Report", text(value.report_path)],
    ["Artifacts", text(value.artifact_dir)],
    ["Updated", formatDate(text(value.updated_at) ?? text(value.created_at))],
  ]);
}

function renderRunReport(value: Record<string, unknown>): void {
  const baseline = record(value.baseline);
  const candidate = record(value.candidate);
  const comparison = record(value.comparison);
  const metadata = record(value.run_metadata);
  printDetail([
    ["Run", text(value.run_id)],
    ["Status", status(value.status)],
    ["Base model", text(value.base_model)],
    ["Tuned model", text(value.fine_tuned_model_id)],
    ["Training examples", text(metadata?.training_example_count)],
    ["Evaluation examples", text(metadata?.eval_examples_used)],
    ["Baseline score", displayNumber(baseline?.avg_score)],
    ["Candidate score", displayNumber(candidate?.avg_score)],
    ["Score delta", displayNumber(comparison?.avg_score_delta)],
    ["Pass-rate delta", displayNumber(comparison?.pass_rate_delta)],
    ["Regressions", text(comparison?.regressions)],
    ["Improvements", text(comparison?.improvements)],
  ]);
  renderGate(value);
}

function renderRunComparison(value: Record<string, unknown>): void {
  const runA = record(value.run_a);
  const runB = record(value.run_b);
  const shared = record(value.shared);
  const sharedA = record(shared?.run_a);
  const sharedB = record(shared?.run_b);
  printTable(
    ["Run", "Candidate score", "Token F1", "Examples"],
    [
      [
        displayId(runA?.run_id),
        displayNumber(sharedA?.candidate_avg_score) ?? "—",
        displayNumber(sharedA?.candidate_avg_token_f1) ?? "—",
        text(shared?.examples) ?? "0",
      ],
      [
        displayId(runB?.run_id),
        displayNumber(sharedB?.candidate_avg_score) ?? "—",
        displayNumber(sharedB?.candidate_avg_token_f1) ?? "—",
        text(shared?.examples) ?? "0",
      ],
    ],
  );
  const notes = Array.isArray(value.notes)
    ? value.notes.filter((note): note is string => typeof note === "string")
    : [];
  for (const note of notes) printWarning(note);
}

function renderModelsList(value: unknown): void {
  const modelRows = rows(value);
  if (modelRows.length === 0) {
    printWarning("No local models found.");
    return;
  }
  printTable(
    ["Model", "Base model", "Run", "Provider", "Created"],
    modelRows.map((model) => [
      displayId(model.id),
      truncate(text(model.base_model) ?? "—", 32),
      displayId(model.run_id),
      text(model.provider) ?? "—",
      formatDate(text(model.created_at)),
    ]),
  );
}

function renderModelRecord(value: Record<string, unknown>): void {
  const model = record(value.model) ?? value;
  const artifact = record(value.artifact);
  const integrity = record(value.integrity);
  printDetail([
    ["Model", text(model.id) ?? text(value.active)],
    ["Base model", text(model.base_model)],
    ["Run", text(model.run_id)],
    ["Provider", text(model.provider)],
    ["Artifact", text(model.artifact_uri) ?? text(artifact?.path)],
    ["Manifest", text(value.manifest_path)],
    ["Verified files", text(integrity?.checked)],
    ["Created", formatDate(text(model.created_at))],
  ]);
}

function renderModelPrefetch(value: Record<string, unknown>): void {
  printSuccess(
    value.local_files_only === true
      ? "Verified the local base-model snapshot"
      : "Base model is ready locally",
  );
  printDetail([
    ["Base model", text(value.base_model)],
    ["Revision", text(value.snapshot_revision)],
    ["Snapshot", text(value.snapshot_path)],
    ["Files", text(value.file_count)],
    ["Size", displayBytes(value.size_bytes)],
    ["Cache", text(value.model_cache) ?? text(value.hf_home)],
  ]);
}

function renderActiveModel(value: Record<string, unknown>): void {
  const active = text(value.active) ?? "base";
  const pointer = record(value.pointer);
  if (pointer?.action === "activate") {
    printSuccess(`Activated local model ${active}`);
  } else if (pointer?.action === "rollback") {
    printSuccess(`Rolled back to ${active}`);
  }
  printDetail([
    ["Active", active],
    ["Run", text(pointer?.run_id)],
    ["Previous", text(pointer?.previous_model_id) ?? "base"],
    ["Changed", formatDate(text(pointer?.activated_at))],
    ["Manifest", text(value.manifest_path)],
  ]);
}

function renderServePlan(value: Record<string, unknown>): void {
  printSuccess("Local serving plan is valid");
  printDetail([
    ["Model", text(value.model_id) ?? text(value.base_model)],
    ["URL", text(value.url)],
    ["Artifact", text(value.artifact_path)],
    ["Manifest", text(value.manifest_path)],
    ["Command", displayCommand(value.command)],
  ]);
}

function renderKnownJson(args: string[], value: unknown): boolean {
  const command = args[0];
  const subcommand = args[1];
  if (command === "doctor" && isRecord(value)) {
    renderDoctor(value);
    return true;
  }
  if (command === "init" && isRecord(value)) {
    renderInit(value);
    return true;
  }
  if (command === "validate" && isRecord(value)) {
    renderValidate(value);
    return true;
  }
  if (command === "run" && isRecord(value)) {
    renderRun(value);
    return true;
  }
  if (command === "runs") {
    if ((subcommand === undefined || subcommand === "list") && Array.isArray(value)) {
      renderRunsList(value);
      return true;
    }
    if (subcommand === "events" && Array.isArray(value)) {
      renderRunEvents(value);
      return true;
    }
    if (subcommand === "get" && isRecord(value)) {
      renderRunRecord(value);
      return true;
    }
    if (subcommand === "report" && isRecord(value)) {
      renderRunReport(value);
      return true;
    }
    if (subcommand === "compare" && isRecord(value)) {
      renderRunComparison(value);
      return true;
    }
  }
  if (command === "models") {
    if ((subcommand === undefined || subcommand === "list") && Array.isArray(value)) {
      renderModelsList(value);
      return true;
    }
    if ((subcommand === "get" || subcommand === "verify") && isRecord(value)) {
      if (subcommand === "verify") printSuccess("Local model artifact is valid");
      renderModelRecord(value);
      return true;
    }
    if (
      (subcommand === "prefetch" || subcommand === "verify-base")
      && isRecord(value)
    ) {
      renderModelPrefetch(value);
      return true;
    }
    if (
      ["active", "activate", "rollback"].includes(subcommand ?? "")
      && isRecord(value)
    ) {
      renderActiveModel(value);
      return true;
    }
    if (subcommand === "serve" && isRecord(value)) {
      renderServePlan(value);
      return true;
    }
  }
  if (command === "serve" && isRecord(value)) {
    renderServePlan(value);
    return true;
  }
  return false;
}

function lastNonEmptyLine(value: string): string | undefined {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .at(-1);
}

export function adaptLocalCliText(value: string): string {
  return value
    .replace(
      /(?<![.@/\w])tuned-tensor-local(?=$|[\s:`\]])/g,
      "tt local",
    )
    .replace(/(?<![.@/\w])tt-local(?=$|[\s:`\]])/g, "tt local");
}

export function localCliErrorEnvelope(
  payload: Pick<
    LocalOutputPayload,
    "exitCode" | "signal" | "stderr" | "stdout" | "errorMessage"
  >,
): Record<string, unknown> {
  const rawMessage = payload.errorMessage
    ?? (payload.stderr.trim() || undefined)
    ?? lastNonEmptyLine(payload.stdout)
    ?? `tt local exited with code ${payload.exitCode}.`;
  return {
    error: {
      status: null,
      code: "LOCAL_CLI_ERROR",
      message: adaptLocalCliText(rawMessage),
      exit_code: payload.exitCode,
      signal: payload.signal,
    },
  };
}

export function localCliTextEnvelope(
  payload: Pick<LocalOutputPayload, "args" | "stdout">,
): Record<string, unknown> {
  const output = adaptLocalCliText(payload.stdout).trim();
  if (payload.args[0] === "info") {
    const lines = output.split(/\r?\n/);
    const heading = lines[0]?.match(/^([^:]+):\s*(.+)$/);
    const fields = Object.fromEntries(
      lines.slice(1).flatMap((line) => {
        const match = line.match(/^([^:]+):\s*(.*)$/);
        return match
          ? [[match[1]!.trim().toLowerCase().replaceAll(" ", "_"), match[2]!.trim()]]
          : [];
      }),
    );
    if (heading) {
      return {
        data: {
          name: heading[1],
          description: heading[2],
          ...fields,
        },
      };
    }
  }
  return { data: { output } };
}

export function renderLocalOutput(
  payload: LocalOutputPayload,
  options: RenderLocalOutputOptions = {},
): void {
  if (
    payload.streamingStdout
    && payload.exitCode === 0
    && !payload.errorMessage
  ) {
    return;
  }
  const output = options.stdout ?? process.stdout;

  if (options.jsonMode) {
    if (payload.hasJson) {
      output.write(payload.stdout);
      return;
    }
    if (payload.exitCode !== 0 || payload.errorMessage) {
      output.write(`${JSON.stringify(localCliErrorEnvelope(payload), null, 2)}\n`);
      return;
    }
    output.write(`${JSON.stringify(localCliTextEnvelope(payload), null, 2)}\n`);
    return;
  }

  if (payload.droppedSpecKeys.length > 0) {
    printWarning(
      `Ignored project field(s) unsupported by the local runtime: ${
        payload.droppedSpecKeys.join(", ")
      }.`,
    );
  }

  if (payload.hasJson && renderKnownJson(payload.args, payload.json)) return;
  if (payload.hasJson) {
    printJson(payload.json);
    return;
  }
  if (payload.stdout) output.write(adaptLocalCliText(payload.stdout));
  if (payload.exitCode !== 0 && !payload.stderr && !payload.stdout) {
    printError(
      payload.errorMessage
        ? adaptLocalCliText(payload.errorMessage)
        : `tt local exited with code ${payload.exitCode}.`,
    );
  }
}
