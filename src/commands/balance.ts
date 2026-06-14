import { Command } from "commander";
import chalk from "chalk";
import { get, type ClientOpts } from "../client.js";
import {
  printDetail,
  printJson,
  printTable,
  isJsonMode,
  formatDate,
  formatStatus,
  shortId,
} from "../output.js";

interface BalanceResponse {
  balance_cents: number;
  reserved_cents: number;
  available_cents: number;
  lifetime_topup_cents: number;
}

type CreditKind =
  | "signup_bonus"
  | "topup"
  | "debit_run"
  | "debit_autotune"
  | "refund"
  | "adjustment";

interface LedgerRow {
  id: string;
  kind: CreditKind;
  amount_cents: number;
  balance_after_cents: number;
  reference_id: string | null;
  created_at: string;
}

const KIND_LABELS: Record<CreditKind, string> = {
  signup_bonus: "Signup bonus",
  topup: "Top-up",
  debit_run: "Run",
  debit_autotune: "Auto-tune",
  refund: "Refund",
  adjustment: "Adjustment",
};

function formatCents(cents: number): string {
  const sign = cents < 0 ? "-" : "";
  const abs = Math.abs(cents);
  return `${sign}$${(abs / 100).toFixed(2)}`;
}

function formatSigned(cents: number): string {
  if (cents > 0) return chalk.green(`+${formatCents(cents)}`);
  if (cents < 0) return chalk.red(formatCents(cents));
  return formatCents(cents);
}

export function registerBalanceCommands(parent: Command) {
  parent
    .command("balance")
    .description("Show credit balance and recent transactions")
    .option("-n, --limit <n>", "Number of transactions to show (default 10)", "10")
    .action(async (options) => {
      const opts = parent.opts() as ClientOpts;
      const limit = Number(options.limit) || 10;

      const [{ data: balance }, { data: transactions }] = await Promise.all([
        get<BalanceResponse>("/billing/balance", undefined, opts),
        get<LedgerRow[]>("/billing/transactions", { per_page: limit }, opts),
      ]);

      if (isJsonMode()) {
        return printJson({ balance, transactions });
      }

      const lowBalance = balance.balance_cents < 100;
      const balanceLine = lowBalance
        ? chalk.red(formatCents(balance.balance_cents)) + chalk.dim(" (low)")
        : chalk.bold.green(formatCents(balance.balance_cents));

      printDetail([
        ["Credits", balanceLine],
        [
          "Reserved",
          balance.reserved_cents > 0 ? formatCents(balance.reserved_cents) : undefined,
        ],
        [
          "Available",
          balance.reserved_cents > 0 ? formatCents(balance.available_cents) : undefined,
        ],
        ["Lifetime top-ups", formatCents(balance.lifetime_topup_cents)],
      ]);

      if (transactions && transactions.length > 0) {
        console.log(`\n${chalk.bold("Recent transactions")}`);
        printTable(
          ["Date", "Type", "Amount", "Balance", "Reference"],
          transactions.map((row) => [
            formatDate(row.created_at),
            KIND_LABELS[row.kind] ?? row.kind,
            formatSigned(row.amount_cents),
            formatCents(row.balance_after_cents),
            row.reference_id ? shortId(row.reference_id) : chalk.dim("—"),
          ])
        );
      } else {
        console.log(
          `\n${chalk.dim("No transactions yet — run `tt topup` to add credits.")}`
        );
      }

      if (lowBalance) {
        console.log(
          `\n${formatStatus("preparing")} ${chalk.yellow(
            "Run `tt topup` to add more credits before starting a run."
          )}`
        );
      }
    });
}
