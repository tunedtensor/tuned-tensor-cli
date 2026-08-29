import { evaluateCapabilities, formatCapabilitySummary } from "./capability.js";
import type { LocalRunnerConfig } from "./contracts.js";
import { collectHostInventory } from "./host-inventory.js";
import {
  readHardwareSnapshot,
  writeHardwareSnapshot,
  type HardwareSnapshot,
} from "./hardware-snapshot.js";

export interface AssessHardwareOptions {
  config?: LocalRunnerConfig;
  /** Skip the bundled torch/uv probe. */
  quick?: boolean;
  env?: NodeJS.ProcessEnv;
  cwd?: string;
}

export type HardwareReport = HardwareSnapshot;

export async function assessHardware(
  options: AssessHardwareOptions = {},
): Promise<HardwareReport> {
  const env = options.env ?? process.env;
  // Doctor and `examine_hardware --quick` must not replace a fresh full
  // probe with nvidia-smi-only verdicts (for example flipping CUDA after
  // torch reported it unavailable).
  if (options.quick === true) {
    const existing = await readHardwareSnapshot(env);
    if (existing && !existing.quick && !existing.stale) {
      return {
        version: existing.version,
        collected_at: existing.collected_at,
        quick: existing.quick,
        gpu_fingerprint: existing.gpu_fingerprint,
        inventory: existing.inventory,
        capabilities: existing.capabilities,
        summary: existing.summary,
      };
    }
  }
  const inventory = await collectHostInventory({
    config: options.config,
    quick: options.quick === true,
    env,
    cwd: options.cwd,
  });
  const capabilities = evaluateCapabilities(inventory);
  return await writeHardwareSnapshot({
    collected_at: inventory.collected_at,
    quick: inventory.quick,
    inventory,
    capabilities,
    summary: formatCapabilitySummary(capabilities),
  }, env);
}
