import chalk from "chalk";

/**
 * Tess, the TT mark: the Tuned Tensor logo's 3×3 tensor grid, with a purple
 * diagonal (light to dark) flanked by slate tiles. Each tile is 2×2
 * half-block pixels with a one-pixel gap, so the mark stays square in a
 * terminal. Terminals without color get a plain-character version.
 */
export const MASCOT_NAME = "Tess";
export const MASCOT_WIDTH = 11;

const PALETTE: Record<string, string> = {
  a: "#B39DFB", // diagonal, top-left
  b: "#9B72F8", // diagonal, center
  c: "#7C3AED", // diagonal, bottom-right
  d: "#2B323D", // slate tiles
};

/** Pixel grid; "." is empty. Two pixel rows make one character row. */
const MARK = [
  "aa.dd...",
  "aa.dd...",
  "........",
  "dd.bb.dd",
  "dd.bb.dd",
  "........",
  "...dd.cc",
  "...dd.cc",
];

function pixelRow(upper: string, lower: string): string {
  let line = "";
  for (let column = 0; column < upper.length; column += 1) {
    const up = upper[column]!;
    const down = lower[column]!;
    if (up === "." && down === ".") line += " ";
    else if (up === ".") line += chalk.hex(PALETTE[down]!)("▄");
    else if (down === ".") line += chalk.hex(PALETTE[up]!)("▀");
    else if (up === down) line += chalk.hex(PALETTE[up]!)("█");
    else line += chalk.hex(PALETTE[up]!).bgHex(PALETTE[down]!)("▀");
  }
  return line;
}

/** Four rows, {@link MASCOT_WIDTH} columns each. */
export function renderMascot(): string[] {
  if (chalk.level === 0) {
    return ["", " ■ □", " □ ■ □", "   □ ■"].map((row) => row.padEnd(MASCOT_WIDTH));
  }
  const rows: string[] = [];
  for (let row = 0; row < MARK.length; row += 2) {
    rows.push(` ${pixelRow(MARK[row]!, MARK[row + 1]!)}  `);
  }
  return rows;
}

/** A one-cell-high mark for inline messages: two diagonal tiles. */
export function mascotMark(): string {
  if (chalk.level === 0) return "▚";
  return `${chalk.hex(PALETTE.a!)("▀")}${chalk.hex(PALETTE.c!)("▄")}`;
}

/** Tess saying something, prefixed with the mark and name. */
export function mascotSays(message: string): string {
  return `${mascotMark()} ${chalk.hex(PALETTE.b!).bold(MASCOT_NAME)} ${chalk.dim("›")} ${message}`;
}

const FAREWELLS = [
  "See you next epoch.",
  "Bye! Your checkpoints are safe with me.",
  "Going dormant. Wake me with tt.",
];

export function mascotFarewell(seed = Date.now()): string {
  return mascotSays(FAREWELLS[Math.abs(seed) % FAREWELLS.length]!);
}
