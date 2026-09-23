import type { Command } from "commander";
import { inspectLocalSpec, readSpecHistory, specDiff } from "../spec-workspace.js";
import { isJsonMode, printJson } from "../output.js";
import { sanitizeTerminalText } from "../terminal-markdown.js";

export async function reviewSpec(workspaceRoot: string, operation = "show", specPath = "tunedtensor.json") {
  if (!["show", "validate", "diff", "history"].includes(operation)) {
    throw new Error("Usage: spec [show|validate|diff|history] [path/to/tunedtensor.json]");
  }

  const spec = await inspectLocalSpec(workspaceRoot, specPath);
  const identity = { path: spec.displayPath, sha256: spec.sha256 };
  const header = `${identity.path} · sha256 ${identity.sha256}`;
  const diagnostics = [
    spec.validation.valid ? "Spec is valid." : spec.validation.errors.join("\n"),
    ...spec.validation.warnings,
  ].join("\n");

  if (operation === "show") {
    return {
      ...identity,
      source: spec.source,
      spec: spec.document,
      validation: spec.validation,
      text: `${header}\n${spec.source}\n${diagnostics}`,
    };
  }
  if (operation === "validate") {
    return { ...identity, ...spec.validation, text: `${header}\n${diagnostics}` };
  }

  const history = await readSpecHistory(workspaceRoot, specPath);
  if (operation === "history") {
    const revisions = history.map(({ before: _before, after: _after, ...entry }) => entry);
    const lines = revisions.map(entry =>
      `${entry.at}  ${entry.status}  ${entry.id}  ${entry.before_sha256.slice(0, 12)} → ${entry.after_sha256.slice(0, 12)}`,
    );
    return {
      ...identity,
      revisions,
      text: `${header}\n${lines.join("\n") || "No reviewed edits recorded yet."}`,
    };
  }

  if (spec.document === undefined) {
    throw new Error("Cannot compare malformed spec JSON. Use /spec or tt spec show to inspect it.");
  }
  const latest = history.find(entry => entry.status === "applied");
  const external = latest && latest.after_sha256 !== spec.sha256;
  const diff = latest
    ? specDiff(external ? latest.after : latest.before, spec.document)
    : "No reviewed edit to compare yet.";
  const label = external ? "Changes since the last approved revision:" : "Latest approved edit:";
  return { ...identity, diff, text: `${header}\n${label}\n${diff}` };
}

export function registerSpecCommands(program: Command, cwd: string) {
  program.command("spec").description("Review the local behavior spec, validation, changes and history")
    .argument("[operation]", "show, validate, diff, or history", "show")
    .argument("[path]", "Workspace-relative tunedtensor.json", "tunedtensor.json")
    .action(async (operation: string, path: string) => {
      const result = await reviewSpec(cwd, operation, path);
      if (isJsonMode()) {
        const { text: _text, ...data } = result;
        printJson(data);
      } else {
        console.log(sanitizeTerminalText(result.text));
      }
      if (operation === "validate" && "valid" in result && !result.valid) {
        process.exitCode = 1;
      }
    });
}
