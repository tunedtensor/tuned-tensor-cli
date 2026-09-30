import { describe, expect, it } from "vitest";
import chalk from "chalk";
import { stripVTControlCharacters } from "node:util";
import {
  highlightJson,
  highlightSpecDiff,
  hintLines,
  renderSpecSource,
  renderSpecView,
  resolveSpecSection,
  truncateText,
  type SpecViewInput,
} from "../spec-view.js";
import { terminalWidth } from "../terminal-markdown.js";

const document = {
  name: "Support triage",
  description: "Routes tickets to the right queue.",
  system_prompt: "Classify the ticket.\nReply with one queue name.",
  guidelines: ["Prefer billing for payment issues.", "Use general when unsure."],
  constraints: ["Never invent a queue."],
  base_model: "Qwen/Qwen3.5-2B",
  examples: Array.from({ length: 5 }, (_, index) => ({ input: `Ticket ${index + 1}`, output: index % 2 ? "billing" : "general" })),
  hyperparameters: { n_epochs: 3, learning_rate: 0.0002, batch_size: 2, lora_rank: 8, lora_alpha: 16, max_seq_length: 512 },
  evaluation: { scoring: { mode: "exact_match" } },
  runtime: { gpu: { provider: "aws", instanceId: "i-0123456789abcdef0", user: "ubuntu" } },
  custom_note: "kept",
};

function input(overrides: Partial<SpecViewInput> = {}): SpecViewInput {
  const source = JSON.stringify(document, null, 2);
  return {
    displayPath: "./tunedtensor.json",
    sha256: "a".repeat(64),
    source,
    document,
    validation: { valid: true, errors: [], warnings: ["Small example set."] },
    ...overrides,
  };
}

const plain = (text: string) => stripVTControlCharacters(text);

describe("renderSpecView", () => {
  it("groups the spec into readable, highlighted sections", () => {
    const view = plain(renderSpecView(input()));
    expect(view).toMatch(/^Support triage\s+✓ valid/);
    expect(view).toContain("adapter · Qwen/Qwen3.5-2B · 5 examples · remote GPU");
    expect(view).toContain("./tunedtensor.json · sha256 aaaaaaaaaaaa");
    for (const heading of ["▍BEHAVIOR", "▍EXAMPLES", "▍TRAINING", "▍EVALUATION", "▍RUNTIME", "▍OTHER"]) {
      expect(view).toContain(heading);
    }
    expect(view).toContain("│ Classify the ticket.");
    expect(view).toContain("✓ Prefer billing for payment issues.");
    expect(view).toContain("✗ Never invent a queue.");
    expect(view).toContain("in  Ticket 1");
    expect(view).toContain("out general");
    expect(view).not.toContain("Ticket 4");
    expect(view).toContain("… 2 more · /spec examples shows all");
    expect(view).toMatch(/n_epochs\s+3/);
    expect(view).toMatch(/scoring\.mode\s+exact_match/);
    expect(view).toMatch(/gpu\.instanceId\s+i-0123456789abcdef0/);
    expect(view).toMatch(/custom_note\s+kept/);
    expect(view).toContain("! Small example set.");
  });

  it("shows decision questions and training without describing them as LoRA", () => {
    const view = plain(renderSpecView(input({ document: {
      ...document, engine: "decision", base_model: "convaiinnovations/laya",
      decision: { type: "choice", criteria: { billing: "refunds", technical: "errors" } },
    } })));
    expect(view).toContain("decision · typed classification");
    expect(view).toContain("Decision question");
    expect(view).toContain("refunds");
    expect(view).not.toContain("adapter · LoRA");
  });

  it("shows one section in full", () => {
    const examples = plain(renderSpecView(input(), { section: "examples" }));
    expect(examples.startsWith("▍EXAMPLES")).toBe(true);
    expect(examples).toContain("Ticket 5");
    expect(examples).not.toContain("▍BEHAVIOR");
    expect(plain(renderSpecView(input(), { section: "pipeline" }))).toContain("derives the default pipeline");
  });

  it("packs short training settings into columns", () => {
    const training = plain(renderSpecView(input(), { section: "training", columns: 88 })).split("\n");
    expect(training.find((line) => line.includes("n_epochs"))).toMatch(/n_epochs\s+3\s+batch_size\s+2\s+lora_alpha\s+16/);
  });

  it("stays within narrow terminals", () => {
    for (const line of renderSpecView(input(), { columns: 44 }).split("\n")) {
      expect(terminalWidth(line)).toBeLessThanOrEqual(44);
    }
  });

  it("falls back to the raw source with errors for malformed JSON", () => {
    const view = plain(renderSpecView(input({
      source: '{"name":',
      document: undefined,
      validation: { valid: false, errors: ["Invalid JSON: Unexpected end"], warnings: [] },
    })));
    expect(view).toContain("could not be parsed");
    expect(view).toContain("Invalid JSON: Unexpected end");
    expect(view).toContain('{"name":');
  });

  it("strips terminal control sequences from spec content", () => {
    const hostile = {
      ...document,
      name: "Evil\u001b[2Jname",
      guidelines: ["\u001b]0;title\u0007hi"],
      hyperparameters: { "\u001b[31mred": 1 },
      "\u001b[2Kother": "x",
    };
    const view = renderSpecView(input({ document: hostile }));
    expect(view).not.toContain("\u001b[2J");
    expect(view).not.toContain("\u001b]0;");
    expect(view).not.toContain("\u001b[31m");
    expect(view).not.toContain("\u001b[2K");
  });
});

describe("spec highlighting", () => {
  it("colors JSON without changing its text", () => {
    const level = chalk.level;
    chalk.level = 3;
    try {
      const source = JSON.stringify(document, null, 2);
      const highlighted = highlightJson(source);
      expect(highlighted).not.toBe(source);
      expect(plain(highlighted)).toBe(source);
      expect(plain(highlightSpecDiff("@@ name @@\n- a\n+ b"))).toBe("@@ name @@\n- a\n+ b");
      expect(plain(renderSpecSource(input()))).toContain('"custom_note": "kept"');
    } finally {
      chalk.level = level;
    }
  });

  it("resolves section names and aliases", () => {
    expect(resolveSpecSection("Examples")).toBe("examples");
    expect(resolveSpecSection("hyperparameters")).toBe("training");
    expect(resolveSpecSection("eval")).toBe("evaluation");
    expect(resolveSpecSection("diff")).toBeUndefined();
  });

  it("truncates by terminal cells and breaks hints between items", () => {
    expect(truncateText("abcdefghij", 5)).toBe("abcd…");
    expect(truncateText("a  b\nc", 10)).toBe("a b c");
    expect(hintLines(["one", "two", "three"], 10).map(plain)).toEqual(["one · two", "three"]);
  });
});
