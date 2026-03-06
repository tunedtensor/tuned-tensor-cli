import chalk from "chalk";
import Table from "cli-table3";

let jsonMode = false;

export function setJsonMode(enabled: boolean) {
  jsonMode = enabled;
}

export function isJsonMode(): boolean {
  return jsonMode;
}

export function printJson(data: unknown) {
  console.log(JSON.stringify(data, null, 2));
}

export function printTable(
  headers: string[],
  rows: string[][],
  meta?: { page: number; per_page: number; total: number },
) {
  const table = new Table({
    head: headers.map((h) => chalk.bold.cyan(h)),
    style: { head: [], border: [] },
  });
  for (const row of rows) {
    table.push(row);
  }
  console.log(table.toString());

  if (meta) {
    const totalPages = Math.ceil(meta.total / meta.per_page);
    console.log(
      chalk.dim(
        `\nPage ${meta.page}/${totalPages} (${meta.total} total)`,
      ),
    );
  }
}

export function printDetail(fields: [string, string | undefined][]) {
  const maxLabel = Math.max(...fields.map(([l]) => l.length));
  for (const [label, value] of fields) {
    const paddedLabel = label.padEnd(maxLabel);
    console.log(`${chalk.bold.cyan(paddedLabel)}  ${value ?? chalk.dim("—")}`);
  }
}

export function printSuccess(message: string) {
  console.log(chalk.green("✓") + " " + message);
}

export function printWarning(message: string) {
  console.log(chalk.yellow("!") + " " + message);
}

export function printError(message: string) {
  console.error(chalk.red("✗") + " " + message);
}

export function formatDate(iso?: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString();
}

export function formatStatus(status: string): string {
  const map: Record<string, (s: string) => string> = {
    completed: chalk.green,
    running: chalk.blue,
    training: chalk.blue,
    evaluating: chalk.blue,
    preparing: chalk.yellow,
    pending: chalk.yellow,
    uploading: chalk.yellow,
    failed: chalk.red,
    cancelled: chalk.dim,
    validated: chalk.green,
    invalid: chalk.red,
  };
  const colorFn = map[status] || chalk.white;
  return colorFn(status);
}

export function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max - 1) + "…";
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}
