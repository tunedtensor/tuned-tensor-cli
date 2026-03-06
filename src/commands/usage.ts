import { Command } from "commander";
import chalk from "chalk";
import { get, type ClientOpts } from "../client.js";
import {
  printDetail,
  printJson,
  isJsonMode,
  formatDate,
} from "../output.js";

interface UsageSummary {
  tier: string;
  tierName: string;
  runsUsed: number;
  runsLimit: number;
  periodStart: string;
  periodEnd: string;
  canStartRun: boolean;
}

interface TierDefinition {
  slug: string;
  name: string;
  runsPerMonth: number;
  monthlyPriceUsd: number;
  features: string[];
}

interface UsageResponse {
  usage: UsageSummary;
  tiers: TierDefinition[];
}

export function registerUsageCommands(parent: Command) {
  parent
    .command("usage")
    .description("Show usage summary and tier info")
    .action(async () => {
      const opts = parent.opts() as ClientOpts;
      const { data } = await get<UsageResponse>("/usage", undefined, opts);

      if (isJsonMode()) return printJson(data);

      const u = data.usage;
      const pct = u.runsLimit > 0 ? ((u.runsUsed / u.runsLimit) * 100).toFixed(0) : "∞";
      const bar = u.runsLimit > 0
        ? progressBar(u.runsUsed, u.runsLimit, 20)
        : chalk.green("unlimited");

      printDetail([
        ["Tier", `${u.tierName} (${u.tier})`],
        ["Runs", `${u.runsUsed} / ${u.runsLimit} (${pct}%)`],
        ["Usage", bar],
        ["Can Start Run", u.canStartRun ? chalk.green("Yes") : chalk.red("No")],
        ["Period Start", formatDate(u.periodStart)],
        ["Period End", formatDate(u.periodEnd)],
      ]);

      if (data.tiers?.length) {
        console.log("\nAvailable Tiers:");
        for (const t of data.tiers) {
          const price = t.monthlyPriceUsd === 0 ? "Free" : `$${t.monthlyPriceUsd}/mo`;
          const current = t.slug === u.tier ? chalk.green(" (current)") : "";
          console.log(`  ${chalk.bold(t.name)}${current} — ${price}, ${t.runsPerMonth} runs/mo`);
          for (const f of t.features) {
            console.log(`    • ${f}`);
          }
        }
      }
    });
}

function progressBar(used: number, total: number, width: number): string {
  const ratio = Math.min(used / total, 1);
  const filled = Math.round(ratio * width);
  const empty = width - filled;
  const color = ratio > 0.9 ? chalk.red : ratio > 0.7 ? chalk.yellow : chalk.green;
  return color("█".repeat(filled)) + chalk.dim("░".repeat(empty));
}
