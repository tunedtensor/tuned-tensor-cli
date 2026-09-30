import chalk from "chalk";
import { sanitizeTerminalText, terminalWidth, wrapTerminalLine } from "./terminal-markdown.js";

/**
 * Human-readable rendering of tunedtensor.json. The JSON file stays the source
 * of truth; this view groups its fields into the sections that drive training
 * (behavior, examples, training settings, evaluation, runtime) and highlights
 * them so people can scan a spec without reading raw JSON.
 */

const accent = chalk.hex("#8B5CF6");
const soft = chalk.hex("#A78BFA");
const keyColor = chalk.hex("#C4B5FD");
const stringColor = chalk.hex("#86EFAC");
const numberColor = chalk.hex("#FCD34D");
const literalColor = chalk.hex("#F9A8D4");

export const SPEC_SECTIONS = [
  "identity",
  "behavior",
  "examples",
  "training",
  "evaluation",
  "runtime",
  "pipeline",
] as const;

export type SpecSection = typeof SPEC_SECTIONS[number];

const SECTION_ALIASES: Record<string, SpecSection> = {
  id: "identity",
  about: "identity",
  prompt: "behavior",
  system: "behavior",
  guidelines: "behavior",
  constraints: "behavior",
  example: "examples",
  data: "examples",
  hyperparameters: "training",
  hparams: "training",
  foundation: "training",
  train: "training",
  eval: "evaluation",
  gpu: "runtime",
};

export function resolveSpecSection(name: string): SpecSection | undefined {
  const normalized = name.trim().toLowerCase();
  if ((SPEC_SECTIONS as readonly string[]).includes(normalized)) return normalized as SpecSection;
  return SECTION_ALIASES[normalized];
}

const SECTION_BLURBS: Record<SpecSection, string> = {
  identity: "what this model is",
  behavior: "what the model should do",
  examples: "teaches by demonstration",
  training: "how it is trained",
  evaluation: "how results are scored",
  runtime: "where it runs",
  pipeline: "custom step recipe",
};

const SECTION_KEYS: Record<SpecSection, readonly string[]> = {
  identity: ["name", "description", "engine", "base_model", "id", "dataset_prebuilt"],
  behavior: ["system_prompt", "guidelines", "constraints", "decision"],
  examples: ["examples"],
  training: ["hyperparameters", "foundation"],
  evaluation: ["evaluation"],
  runtime: ["runtime"],
  pipeline: ["pipeline"],
};

const KNOWN_KEYS = new Set(Object.values(SECTION_KEYS).flat());

export interface SpecViewInput {
  displayPath: string;
  sha256: string;
  source: string;
  document: unknown;
  validation: { valid: boolean; errors: string[]; warnings: string[] };
}

export interface SpecViewOptions {
  /** Show one section in full instead of the overview. */
  section?: SpecSection;
  /** Terminal columns; the view caps itself at a readable width. */
  columns?: number;
}

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function clean(value: string): string {
  return sanitizeTerminalText(value);
}

/** Collapse whitespace and cut to `width` cells with an ellipsis. */
export function truncateText(value: string, width: number): string {
  const flat = clean(value).replace(/\s+/g, " ").trim();
  if (terminalWidth(flat) <= width) return flat;
  let out = "";
  for (const char of flat) {
    if (terminalWidth(out + char) > width - 1) break;
    out += char;
  }
  return `${out}…`;
}

/** Dim `a · b · c` hints that break between items rather than inside them. */
export function hintLines(items: string[], width: number): string[] {
  return joinFitting(items, width).map((line) => chalk.dim(line));
}

/** Join `items` with ` · `, starting a new line instead of splitting an item. */
export function joinFitting(items: string[], width: number): string[] {
  const lines: string[] = [];
  let current = "";
  for (const item of items) {
    const next = current ? `${current} · ${item}` : item;
    if (current && terminalWidth(next) > width) {
      lines.push(current);
      current = item;
    } else {
      current = next;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function formatScalar(value: unknown): string {
  if (typeof value === "string") return stringColor(clean(value));
  if (typeof value === "number") return numberColor(String(value));
  if (typeof value === "boolean" || value === null) return literalColor(String(value));
  return highlightJson(JSON.stringify(value));
}

/** Flatten nested settings into `a.b.c` paths for compact display. */
function flattenSettings(value: unknown, prefix = ""): Array<[string, unknown]> {
  if (!isObject(value)) return [[prefix || "value", value]];
  const rows: Array<[string, unknown]> = [];
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (isObject(child) && Object.keys(child).length > 0) rows.push(...flattenSettings(child, path));
    else rows.push([path, child]);
  }
  return rows;
}

/** Syntax-highlight JSON text: keys, strings, numbers, and literals. */
export function highlightJson(source: string): string {
  const token = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b|([{}[\],])/g;
  return clean(source).replace(token, (match, string?: string, colon?: string, number?: string, literal?: string, punctuation?: string) => {
    if (string !== undefined) {
      return colon !== undefined ? `${keyColor(string)}${chalk.dim(colon)}` : stringColor(string);
    }
    if (number !== undefined) return numberColor(number);
    if (literal !== undefined) return literalColor(literal);
    if (punctuation !== undefined) return chalk.dim(punctuation);
    return match;
  });
}

/** Color unified-style spec diff lines (`@@ field @@`, `- old`, `+ new`). */
export function highlightSpecDiff(diff: string): string {
  return diff.split("\n").map((line) => {
    if (line.startsWith("@@")) return accent.bold(line);
    if (line.startsWith("+")) return chalk.green(line);
    if (line.startsWith("-")) return chalk.red(line);
    return line;
  }).join("\n");
}

class ViewBuilder {
  readonly lines: string[] = [];
  constructor(readonly width: number) {}

  push(line = ""): void {
    this.lines.push(line);
  }

  /** Wrap text under a fixed indent and optional gutter marker. */
  paragraph(text: string, indent: string, style: (value: string) => string = (value) => value, maxLines?: number): void {
    const body = this.width - terminalWidth(indent);
    const wrapped = clean(text).split(/\r?\n/).flatMap((line) =>
      line.trim() ? wrapTerminalLine(line, Math.max(20, body)) : [""]);
    const shown = maxLines && wrapped.length > maxLines ? wrapped.slice(0, maxLines) : wrapped;
    for (const line of shown) this.push(`${indent}${style(line)}`);
    if (shown.length < wrapped.length) {
      this.push(`${indent}${chalk.dim(`… ${wrapped.length - shown.length} more lines`)}`);
    }
  }

  heading(section: SpecSection, detail?: string): void {
    const title = section.toUpperCase();
    const room = this.width - title.length - 3;
    const extra = detail && room >= 8 ? `  ${chalk.dim(truncateText(detail, room))}` : "";
    this.push("");
    this.push(`${accent("▍")}${accent.bold(title)}${extra}`);
  }

  settingsGrid(entries: Array<[string, unknown]>, indent = "  "): void {
    if (entries.length === 0) return;
    // Keys come from the spec file too; strip control sequences before layout.
    const rows = entries.map(([key, value]): [string, unknown] => [clean(key), value]);
    // Short scalar settings (hyperparameters) pack into columns, read top-down.
    const compact = rows.every(([, value]) => value === null || ["number", "boolean"].includes(typeof value)
      || (typeof value === "string" && value.length <= 16));
    if (compact && rows.length > 4) {
      const keyWidth = Math.max(...rows.map(([key]) => key.length));
      const valueWidth = Math.max(...rows.map(([, value]) => String(value).length));
      const cell = keyWidth + 2 + valueWidth;
      const columns = Math.min(3, Math.floor((this.width - indent.length + 4) / (cell + 4)));
      if (columns >= 2) {
        const height = Math.ceil(rows.length / columns);
        for (let row = 0; row < height; row += 1) {
          const cells: string[] = [];
          for (let column = 0; column < columns; column += 1) {
            const entry = rows[column * height + row];
            if (!entry) continue;
            const [key, value] = entry;
            const text = String(value);
            cells.push(`${keyColor(key.padEnd(keyWidth))}  ${formatScalar(typeof value === "string" ? clean(value) : value)}${" ".repeat(valueWidth - text.length)}`);
          }
          this.push(`${indent}${cells.join("    ").trimEnd()}`);
        }
        return;
      }
    }
    const keyWidth = Math.min(28, Math.max(...rows.map(([key]) => key.length)));
    for (const [key, value] of rows) {
      const label = key.length > keyWidth ? truncateText(key, keyWidth) : key.padEnd(keyWidth);
      const rendered = typeof value === "string"
        ? stringColor(truncateText(value, Math.max(20, this.width - keyWidth - indent.length - 2)))
        : formatScalar(value);
      this.push(`${indent}${keyColor(label)}  ${rendered}`);
    }
  }
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => (typeof item === "string" ? item : JSON.stringify(item))) : [];
}

function exampleText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === undefined) return "";
  return JSON.stringify(value);
}

function renderIdentity(view: ViewBuilder, spec: JsonObject, full: boolean): void {
  if (!full) return;
  view.heading("identity", SECTION_BLURBS.identity);
  view.settingsGrid(SECTION_KEYS.identity
    .filter((key) => spec[key] !== undefined)
    .map((key) => [key, spec[key]] as [string, unknown]));
  if (spec.engine === undefined) view.push(`  ${chalk.dim("engine defaults to adapter (LoRA on base_model)")}`);
}

function renderBehavior(view: ViewBuilder, spec: JsonObject, full: boolean): void {
  const guidelines = stringList(spec.guidelines);
  const constraints = stringList(spec.constraints);
  view.heading("behavior", SECTION_BLURBS.behavior);
  view.push(`  ${chalk.bold("System prompt")}`);
  if (typeof spec.system_prompt === "string" && spec.system_prompt.trim()) {
    view.paragraph(spec.system_prompt, `  ${soft("│")} `, chalk.italic, full ? undefined : 6);
  } else {
    view.push(`  ${soft("│")} ${chalk.dim("(none)")}`);
  }
  // Each item's marker sits on its first line; wrapped lines hang under the text.
  const renderList = (label: string, items: string[], mark: string) => {
    view.push(`  ${chalk.bold(label)} ${chalk.dim(String(items.length))}`);
    if (items.length === 0) {
      view.push(`    ${chalk.dim("(none)")}`);
      return;
    }
    const limit = full ? items.length : 5;
    for (const item of items.slice(0, limit)) {
      const before = view.lines.length;
      view.paragraph(item, "      ", undefined, full ? undefined : 2);
      view.lines[before] = `    ${mark} ${view.lines[before]!.slice(6)}`;
    }
    if (items.length > limit) view.push(`    ${chalk.dim(`… ${items.length - limit} more · /spec behavior`)}`);
  };
  if (isObject(spec.decision)) {
    view.push(`  ${chalk.bold("Decision question")}`);
    view.settingsGrid(flattenSettings(spec.decision));
  }
  renderList("Guidelines", guidelines, chalk.green("✓"));
  renderList("Constraints", constraints, chalk.red("✗"));
}

function renderExamples(view: ViewBuilder, spec: JsonObject, full: boolean): void {
  const examples = Array.isArray(spec.examples) ? spec.examples : [];
  const count = examples.length;
  const size = count === 0 ? "none yet" : `${count} example${count === 1 ? "" : "s"}`;
  view.heading("examples", `${size} · ${SECTION_BLURBS.examples}`);
  if (count === 0) {
    view.push(`  ${chalk.dim("Ask TT to draft examples, or add input/output pairs to tunedtensor.json.")}`);
    return;
  }
  const limit = full ? count : 3;
  const number = String(Math.min(count, limit)).length;
  const textWidth = Math.max(20, view.width - number - 10);
  examples.slice(0, limit).forEach((example, index) => {
    const record: JsonObject = isObject(example) ? example : { input: example };
    const label = chalk.dim(String(index + 1).padStart(number));
    const pad = " ".repeat(number);
    if (full) {
      const before = view.lines.length;
      view.paragraph(exampleText(record.input), `  ${pad}  ${chalk.cyan("in ")} `);
      view.lines[before] = view.lines[before]!.replace(`  ${pad}`, `  ${label}`);
      view.paragraph(exampleText(record.output), `  ${pad}  ${chalk.green("out")} `, stringColor);
      view.push("");
    } else {
      view.push(`  ${label}  ${chalk.cyan("in ")} ${truncateText(exampleText(record.input), textWidth)}`);
      view.push(`  ${pad}  ${chalk.green("out")} ${stringColor(truncateText(exampleText(record.output), textWidth))}`);
    }
  });
  if (full && view.lines.at(-1) === "") view.lines.pop();
  if (count > limit) view.push(`  ${chalk.dim(`… ${count - limit} more · /spec examples shows all`)}`);
}

function renderTraining(view: ViewBuilder, spec: JsonObject): void {
  const foundation = spec.engine === "foundation";
  const settings = foundation ? spec.foundation : spec.hyperparameters;
  view.heading("training", foundation
    ? "foundation · tokenizer, pretrain, SFT"
    : spec.engine === "decision" ? "decision · typed classification"
    : `adapter · LoRA on ${typeof spec.base_model === "string" ? clean(spec.base_model) : "base_model"}`);
  const rows = isObject(settings) ? flattenSettings(settings) : [];
  if (rows.length === 0) {
    view.push(`  ${chalk.dim("Defaults — no overrides set.")}`);
    return;
  }
  view.settingsGrid(rows);
  const other = foundation ? spec.hyperparameters : spec.foundation;
  if (isObject(other)) view.push(`  ${chalk.yellow("!")} ${chalk.dim(`${foundation ? "hyperparameters" : "foundation"} is set but unused by this engine`)}`);
}

function renderSettingsSection(view: ViewBuilder, spec: JsonObject, section: "evaluation" | "runtime" | "pipeline", full: boolean): void {
  const value = spec[section];
  if (value === undefined) {
    if (full) {
      view.heading(section, SECTION_BLURBS[section]);
      view.push(`  ${chalk.dim(section === "runtime"
        ? "Not set — runs on this machine with default paths."
        : section === "pipeline"
          ? "Not set — TT derives the default pipeline from this spec."
          : "Not set — default scoring.")}`);
    }
    return;
  }
  if (section === "pipeline" && !full) {
    const steps = isObject(value) && Array.isArray(value.steps) ? value.steps.length : undefined;
    view.heading(section, SECTION_BLURBS[section]);
    view.push(`  ${steps === undefined ? "custom recipe" : `${steps} custom step${steps === 1 ? "" : "s"}`} ${chalk.dim("· /spec pipeline")}`);
    return;
  }
  view.heading(section, SECTION_BLURBS[section]);
  if (section === "pipeline") {
    for (const line of JSON.stringify(value, null, 2).split("\n")) view.push(`  ${highlightJson(line)}`);
    return;
  }
  view.settingsGrid(flattenSettings(value));
}

function renderOther(view: ViewBuilder, spec: JsonObject): void {
  const rows = Object.entries(spec).filter(([key]) => !KNOWN_KEYS.has(key));
  if (rows.length === 0) return;
  view.push("");
  view.push(`${accent("▍")}${accent.bold("OTHER")}`);
  view.settingsGrid(rows);
}

function renderHeader(view: ViewBuilder, input: SpecViewInput, spec: JsonObject): void {
  const name = typeof spec.name === "string" && spec.name.trim() ? clean(spec.name) : "Unnamed spec";
  const status = input.validation.valid
    ? chalk.green("✓ valid")
    : chalk.red(`✗ ${input.validation.errors.length} issue${input.validation.errors.length === 1 ? "" : "s"}`);
  const gap = Math.max(2, view.width - terminalWidth(name) - terminalWidth(status));
  view.push(`${chalk.bold(name)}${" ".repeat(gap)}${status}`);
  if (typeof spec.description === "string" && spec.description.trim()) {
    view.paragraph(spec.description, "", chalk.dim, 3);
  }
  const examples = Array.isArray(spec.examples) ? spec.examples.length : 0;
  const facts = [
    spec.engine === "foundation" ? "foundation" : spec.engine === "decision" ? "decision" : "adapter",
    typeof spec.base_model === "string" ? truncateText(spec.base_model, view.width) : undefined,
    `${examples} example${examples === 1 ? "" : "s"}`,
    isObject(spec.runtime) && isObject(spec.runtime.gpu) ? "remote GPU" : undefined,
  ].filter((fact): fact is string => Boolean(fact));
  view.lines.push(...joinFitting(facts, view.width));
  view.lines.push(...hintLines([truncateText(input.displayPath, view.width), `sha256 ${input.sha256.slice(0, 12)}`], view.width));
}

function renderDiagnostics(view: ViewBuilder, input: SpecViewInput): void {
  const { errors, warnings, valid } = input.validation;
  if (valid && warnings.length === 0) return;
  view.push("");
  const marked = (text: string, mark: string, style: (value: string) => string) => {
    const before = view.lines.length;
    view.paragraph(text, "  ", style);
    view.lines[before] = `${mark} ${view.lines[before]!.slice(2)}`;
  };
  for (const error of errors) marked(error, chalk.red("✗"), chalk.red);
  for (const warning of warnings) marked(warning, chalk.yellow("!"), chalk.yellow);
}

function renderInvalid(view: ViewBuilder, input: SpecViewInput): string {
  view.push(`${chalk.red("✗")} ${chalk.bold(clean(input.displayPath))} ${chalk.dim("could not be parsed")}`);
  for (const error of input.validation.errors) view.paragraph(error, "  ", chalk.red);
  view.push("");
  for (const line of clean(input.source).split("\n")) view.push(`  ${chalk.dim(line)}`);
  return `${view.lines.join("\n")}\n`;
}

export function renderSpecView(input: SpecViewInput, options: SpecViewOptions = {}): string {
  const width = Math.max(40, Math.min(options.columns ?? 88, 88));
  const view = new ViewBuilder(width);
  if (!isObject(input.document)) return renderInvalid(view, input);
  const spec = input.document;
  const section = options.section;

  if (!section) {
    renderHeader(view, input, spec);
    renderBehavior(view, spec, false);
    renderExamples(view, spec, false);
    renderTraining(view, spec);
    renderSettingsSection(view, spec, "evaluation", false);
    renderSettingsSection(view, spec, "runtime", false);
    renderSettingsSection(view, spec, "pipeline", false);
    renderOther(view, spec);
    renderDiagnostics(view, input);
    view.push("");
    view.lines.push(...hintLines(["/spec <section> for detail", "/spec show for JSON", "/spec diff", "/spec history"], width));
    view.lines.push(...wrapTerminalLine(`sections: ${SPEC_SECTIONS.join(", ")}`, width).map((line) => chalk.dim(line)));
    return `${view.lines.join("\n")}\n`;
  }

  switch (section) {
    case "identity": renderIdentity(view, spec, true); break;
    case "behavior": renderBehavior(view, spec, true); break;
    case "examples": renderExamples(view, spec, true); break;
    case "training": renderTraining(view, spec); break;
    default: renderSettingsSection(view, spec, section, true);
  }
  view.lines.shift();
  return `${view.lines.join("\n")}\n`;
}

/** The full JSON file, syntax-highlighted, followed by validation results. */
export function renderSpecSource(input: SpecViewInput): string {
  const status = input.validation.valid ? chalk.green("✓ valid") : chalk.red("✗ invalid");
  const lines = [
    `${chalk.bold(clean(input.displayPath))} ${chalk.dim(`· sha256 ${input.sha256.slice(0, 12)}`)}  ${status}`,
    highlightJson(input.source.replace(/\s+$/, "")),
    ...input.validation.errors.map((error) => chalk.red(`✗ ${clean(error)}`)),
    ...input.validation.warnings.map((warning) => chalk.yellow(`! ${clean(warning)}`)),
  ];
  return `${lines.join("\n")}\n`;
}
