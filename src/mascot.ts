import chalk from "chalk";

/**
 * Tardi, the TT tardigrade. Tardigrades survive almost anything — a fitting
 * companion for long fine-tuning runs. Tardi only uses box-drawing and
 * single-width symbols so it renders the same in every terminal font.
 */
export type MascotMood = "happy" | "excited" | "thinking" | "sleepy" | "worried";

const FACES: Record<MascotMood, { eyes: [string, string]; mouth: string }> = {
  happy: { eyes: ["◕", "◕"], mouth: "o" },
  excited: { eyes: ["^", "^"], mouth: "o" },
  thinking: { eyes: ["•", "•"], mouth: "." },
  sleepy: { eyes: ["-", "-"], mouth: "." },
  worried: { eyes: ["°", "°"], mouth: "~" },
};

export const MASCOT_NAME = "Tardi";
export const MASCOT_WIDTH = 11;

const body = chalk.hex("#8B5CF6");
const legs = chalk.hex("#A78BFA").dim;
const eye = chalk.hex("#F5F3FF").bold;

/** Three rows, {@link MASCOT_WIDTH} columns each. */
export function renderMascot(mood: MascotMood = "happy"): string[] {
  const face = FACES[mood];
  return [
    body("  ╭─────╮  "),
    `${body(" ( ")}${eye(face.eyes[0])} ${body(face.mouth)} ${eye(face.eyes[1])}${body(" ) ")}`,
    `${body("  ╰")}${legs("┬┬")}${body("─")}${legs("┬┬")}${body("╯  ")}`,
  ];
}

/** A one-line face for inline messages, e.g. `(◕o◕)`. */
export function mascotFace(mood: MascotMood = "happy"): string {
  const face = FACES[mood];
  return `${body("(")}${eye(face.eyes[0])}${body(face.mouth)}${eye(face.eyes[1])}${body(")")}`;
}

/** Tardi saying something, prefixed with its face and name. */
export function mascotSays(message: string, mood: MascotMood = "happy"): string {
  return `${mascotFace(mood)} ${body.bold(MASCOT_NAME)} ${chalk.dim("›")} ${message}`;
}

const FAREWELLS = [
  "See you next epoch.",
  "Bye! Your checkpoints are safe with me.",
  "Going dormant. Wake me with tt.",
];

export function mascotFarewell(seed = Date.now()): string {
  return mascotSays(FAREWELLS[Math.abs(seed) % FAREWELLS.length]!, "sleepy");
}
