import { Command } from "commander";
import chalk from "chalk";
import open from "open";
import { post, type ClientOpts } from "../client.js";
import { printJson, isJsonMode, printSuccess } from "../output.js";

interface TopupResponse {
  checkout_url: string;
  session_id: string;
}

const PRESETS_USD = [10, 25, 50, 100];
const MIN_USD = 5;
const MAX_USD = 10000;

function parseAmount(input: string | undefined): number | null {
  if (!input) return null;
  const cleaned = input.replace(/^\$/, "").trim();
  const num = Number(cleaned);
  if (!Number.isFinite(num) || num <= 0) return null;
  return num;
}

async function readLine(prompt: string): Promise<string> {
  const readline = await import("node:readline/promises");
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  try {
    return (await rl.question(prompt)).trim();
  } finally {
    rl.close();
  }
}

export function registerTopupCommands(parent: Command) {
  parent
    .command("topup")
    .description("Add prepaid credits via Stripe Checkout")
    .option("-a, --amount <usd>", "Amount in USD (e.g. 25)")
    .option("--no-open", "Print checkout URL instead of opening the browser")
    .action(async (options) => {
      const opts = parent.opts() as ClientOpts;

      let amountUsd: number | null;

      if (options.amount !== undefined) {
        amountUsd = parseAmount(options.amount);
        if (amountUsd === null) {
          throw new Error("Invalid amount");
        }
      } else {
        if (isJsonMode()) {
          throw new Error(
            "amount is required in --json mode (use --amount <usd>)"
          );
        }
        console.log(chalk.bold("Add credits via Stripe Checkout"));
        console.log(
          chalk.dim(
            `Quick picks: ${PRESETS_USD.map((n) => `$${n}`).join(", ")}`
          )
        );
        const input = await readLine(`Amount in USD (min $${MIN_USD}): `);
        amountUsd = parseAmount(input);
        if (amountUsd === null) {
          throw new Error("Invalid amount");
        }
      }

      if (amountUsd < MIN_USD) {
        throw new Error(`Minimum top-up is $${MIN_USD}`);
      }
      if (amountUsd > MAX_USD) {
        throw new Error(`Maximum top-up is $${MAX_USD}`);
      }

      const amountCents = Math.round(amountUsd * 100);

      const { data } = await post<TopupResponse>(
        "/billing/topup",
        { amount_cents: amountCents },
        opts
      );

      if (isJsonMode()) {
        return printJson(data);
      }

      const url = data.checkout_url;
      console.log(
        `${chalk.bold("Checkout URL:")} ${chalk.cyan(url)}\n` +
          chalk.dim("Complete the payment to credit your account.")
      );

      if (options.open !== false) {
        try {
          await open(url);
          printSuccess("Opened Stripe Checkout in your browser.");
        } catch {
          console.log(
            chalk.yellow(
              "Could not open browser automatically — copy the URL above."
            )
          );
        }
      }
    });
}
