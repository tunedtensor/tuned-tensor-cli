import { evaluateCapabilities, formatCapabilitySummary } from "./capability.js";
import type { LocalRunnerConfig } from "./contracts.js";
import { collectHostInventory } from "./host-inventory.js";
import {
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
  const inventory = await collectHostInventory({
    config: options.config,
    quick: options.quick === true,
    env: options.env,
    cwd: options.cwd,
  });
  const capabilities = evaluateCapabilities(inventory);
  return await writeHardwareSnapshot({
    collected_at: inventory.collected_at,
    quick: inventory.quick,
    inventory,
    capabilities,
    summary: formatCapabilitySummary(capabilities),
  }, options.env ?? process.env);
}
