import chalk from "chalk";

/**
 * Tess, the TT tensor cube. A 3D tensor is a cube of numbers, so Tess is a
 * small isometric purple cube drawn with half-block pixels: a light top face,
 * a violet front face that carries the expression, and a deep purple side.
 * Terminals without color get a box-drawing outline instead.
 */
export type MascotMood = "happy" | "excited" | "thinking" | "sleepy" | "worried";

const FACES: Record<MascotMood, { eyes: [string, string]; mouth: string }> = {
  happy: { eyes: ["◕", "◕"], mouth: "o" },
  excited: { eyes: ["^", "^"], mouth: "o" },
  thinking: { eyes: ["•", "•"], mouth: "." },
  sleepy: { eyes: ["-", "-"], mouth: "." },
  worried: { eyes: ["°", "°"], mouth: "~" },
};

export const MASCOT_NAME = "Tess";
export const MASCOT_WIDTH = 11;

const PALETTE: Record<string, string> = {
  T: "#DDD6FE", // top face
  F: "#8B5CF6", // front face
  f: "#7C3AED", // front face, lower edge
  S: "#5B21B6", // side face
};
const EYE = "#F5F3FF";

/**
 * Pixel grid in oblique projection: each character cell stacks two pixels, the
 * front face is square, and the top and side faces recede two pixels up and
 * to the right so the cube reads as three-dimensional.
 */
const CUBE = [
  "..TTTTTTT",
  ".TTTTTTTS",
  "FFFFFFFSS",
  "FFFFFFFSS",
  "FFFFFFFSS",
  "FFFFFFFSS",
  "FFFFFFFS.",
  "fffffff..",
];

/** Front-face text for each character row: eyes on row 1, mouth on row 2. */
function faceRows(mood: MascotMood): string[] {
  const face = FACES[mood];
  return ["", `  ${face.eyes[0]} ${face.eyes[1]}`, `   ${face.mouth}`];
}

function pixelRow(upper: string, lower: string, overlay: string | undefined): string {
  let line = "";
  for (let column = 0; column < upper.length; column += 1) {
    const up = upper[column]!;
    const down = lower[column] ?? ".";
    const mark = overlay?.[column];
    if (mark && mark !== " " && up !== "." && down !== ".") {
      line += chalk.bgHex(PALETTE[down]!).hex(EYE).bold(mark);
    } else if (up === "." && down === ".") {
      line += " ";
    } else if (up === ".") {
      line += chalk.hex(PALETTE[down]!)("▄");
    } else if (down === ".") {
      line += chalk.hex(PALETTE[up]!)("▀");
    } else if (up === down) {
      line += chalk.bgHex(PALETTE[up]!)(" ");
    } else {
      line += chalk.hex(PALETTE[up]!).bgHex(PALETTE[down]!)("▀");
    }
  }
  return line;
}

function outlineCube(mood: MascotMood): string[] {
  const face = FACES[mood];
  return [
    "  ┌──────┐",
    " ┌──────┐│",
    ` │ ${face.eyes[0]}  ${face.eyes[1]} ││`,
    " └──────┘┘",
  ].map((row) => row.padEnd(MASCOT_WIDTH));
}

/** Four rows, {@link MASCOT_WIDTH} columns each. */
export function renderMascot(mood: MascotMood = "happy"): string[] {
  if (chalk.level === 0) return outlineCube(mood);
  const face = faceRows(mood);
  const rows: string[] = [];
  for (let row = 0; row < CUBE.length; row += 2) {
    rows.push(` ${pixelRow(CUBE[row]!, CUBE[row + 1]!, face[row / 2])} `);
  }
  return rows;
}

/** A one-line cube face for inline messages. */
export function mascotFace(mood: MascotMood = "happy"): string {
  const face = FACES[mood];
  if (chalk.level === 0) return `[${face.eyes[0]}${face.eyes[1]}]`;
  return `${chalk.bgHex(PALETTE.F!).hex(EYE).bold(`${face.eyes[0]}${face.eyes[1]}`)}${chalk.hex(PALETTE.S!)("▌")}`;
}

/** Tess saying something, prefixed with its face and name. */
export function mascotSays(message: string, mood: MascotMood = "happy"): string {
  return `${mascotFace(mood)} ${chalk.hex(PALETTE.F!).bold(MASCOT_NAME)} ${chalk.dim("›")} ${message}`;
}

const FAREWELLS = [
  "See you next epoch.",
  "Bye! Your checkpoints are safe with me.",
  "Going dormant. Wake me with tt.",
];

export function mascotFarewell(seed = Date.now()): string {
  return mascotSays(FAREWELLS[Math.abs(seed) % FAREWELLS.length]!, "sleepy");
}
