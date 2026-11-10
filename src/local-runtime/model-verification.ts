import { readFile, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import { assertArtifactManifest } from "./artifacts.js";
import { assertUsableModelArtifact } from "./model-registry.js";
import { fingerprintLocalBaseModel } from "./orchestrator.js";
import type { LocalModelRecord } from "./store.js";
import type { LocalRunnerConfig } from "./contracts.js";

export async function verifyStoredModel(model: LocalModelRecord, config: LocalRunnerConfig): Promise<{
  manifest_path: string;
  integrity: Awaited<ReturnType<typeof assertArtifactManifest>>;
  artifact: Awaited<ReturnType<typeof assertUsableModelArtifact>>;
  contract: unknown;
}> {
  const manifestPath = join(model.artifact_dir, "artifact-manifest.json");
  const integrity = await assertArtifactManifest(manifestPath, {
    requiredPaths: ["stage-metadata.json", "training-report.json"],
    scopeToRequired: true,
    verifyModel: true,
  });
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as {
    model?: {
      artifact_root?: unknown;
      base_model?: unknown;
      format?: unknown;
      framework?: unknown;
      servable?: unknown;
      base_model_artifact_uri?: unknown;
      base_model_fingerprint?: unknown;
    };
  };
  if (!manifest.model) {
    throw new Error(`Artifact manifest does not contain a model contract: ${manifestPath}`);
  }
  const artifact = await assertUsableModelArtifact(model.artifact_uri);
  if (
    typeof manifest.model.base_model_artifact_uri === "string"
    && typeof manifest.model.base_model_fingerprint === "string"
  ) {
    const actualBaseFingerprint = await fingerprintLocalBaseModel(
      manifest.model.base_model_artifact_uri,
      typeof manifest.model.base_model === "string"
        ? manifest.model.base_model
        : undefined,
      config.paths.modelCache,
    );
    if (actualBaseFingerprint !== manifest.model.base_model_fingerprint) {
      throw new Error("Recorded local base-model content no longer matches the model artifact contract.");
    }
  }
  if (
    typeof manifest.model.artifact_root !== "string"
    || resolve(manifest.model.artifact_root) !== resolve(artifact.path)
  ) {
    throw new Error("Stored model record does not match the artifact covered by its manifest.");
  }
  if (manifest.model.base_model !== model.base_model) {
    throw new Error("Stored model base model does not match its artifact manifest.");
  }
  return {
    manifest_path: manifestPath,
    integrity,
    artifact,
    contract: manifest.model,
  };
}

export async function verifyActivationEvidence(model: LocalModelRecord): Promise<void> {
  const generalBaseline = join(model.artifact_dir, "general-baseline-eval.json");
  const generalCandidate = join(model.artifact_dir, "general-candidate-eval.json");
  const [hasBaseline, hasCandidate] = await Promise.all([
    stat(generalBaseline).then((metadata) => metadata.isFile(), () => false),
    stat(generalCandidate).then((metadata) => metadata.isFile(), () => false),
  ]);
  if (!hasBaseline || !hasCandidate) {
    throw new Error(
      "This model cannot be activated because the run did not include a general-regression suite. "
      + "Add evaluation.generalRegression to tunedtensor.json with a held-out dataset, re-run training, then activate. "
      + "Until then, serve the adapter with `tt serve local-<run-id>`.",
    );
  }
  await assertArtifactManifest(join(model.artifact_dir, "artifact-manifest.json"), {
    requiredPaths: [
      "run-report.json",
      "general-baseline-eval.json",
      "general-candidate-eval.json",
    ],
    scopeToRequired: true,
    verifyModel: true,
  });
}
