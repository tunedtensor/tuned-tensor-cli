import chalk from "chalk";
import { stripVTControlCharacters } from "node:util";
import { describe, expect, it } from "vitest";
import { StreamingTerminalMarkdown } from "../terminal-markdown.js";

describe("StreamingTerminalMarkdown", () => {
  it("keeps split Markdown tokens out of terminal output", () => {
    const originalLevel = chalk.level;
    chalk.level = 3;
    try {
      const renderer = new StreamingTerminalMarkdown();
      expect(renderer.push("**Behaviour")).toBe("");
      const output = renderer.push(" Specs**\n- List and inspect `specs`.\n");
      const plain = stripVTControlCharacters(output);

      expect(plain).toContain("Behaviour Specs");
      expect(plain).toContain("• List and inspect");
      expect(plain).not.toContain("**");
      expect(plain).not.toContain("`specs`");
      expect(output).toContain("\u001b[1m");
    } finally {
      chalk.level = originalLevel;
    }
  });

  it("renders headings, quotes, links, and fenced code without markers", () => {
    const renderer = new StreamingTerminalMarkdown();
    const output = renderer.push([
      "## Summary",
      "> Read the docs.",
      "[Tuned Tensor](https://tunedtensor.com)",
      "```sh",
      "tt runs list",
      "```",
      "",
    ].join("\n"));

    expect(output).toContain("Summary");
    expect(output).toContain("│ Read the docs.");
    expect(output).toContain("Tuned Tensor (https://tunedtensor.com)");
    expect(output).toContain("  tt runs list");
    expect(output).not.toContain("##");
    expect(output).not.toContain("```");
  });

  it("flushes a final partial line and strips terminal controls", () => {
    const renderer = new StreamingTerminalMarkdown();
    expect(renderer.push("Safe\u001b[")).toBe("");
    expect(renderer.push("31m answer")).toBe("");
    expect(renderer.flush()).toBe("Safe answer");
    expect(renderer.flush()).toBe("");
  });
});
