import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const readme = readFileSync(new URL("../../README.md", import.meta.url), "utf8");
const trainingOverview = readme
  .split("The agent can describe the built-in adapter and foundation pipeline stages,")[1]!
  .split("To reuse a download,")[0]!
  .replace(/\s+/g, " ");

describe("README training approval guidance", () => {
  it("distinguishes dry-run approval from separately approved real training", () => {
    expect(trainingOverview).not.toContain("Real training is intentionally not model-mediated");
    expect(trainingOverview).toContain("Dry-run approval never starts real training");
    expect(trainingOverview).toContain("same spec bytes completes");
    expect(trainingOverview).toContain("separate approval starts real training in the background");
    expect(trainingOverview).toContain("tt pipeline run");
    expect(trainingOverview).toContain("[background runs](docs/local-runtime/background-runs.md)");
  });
});
