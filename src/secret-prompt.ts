import chalk from "chalk";
import { emitKeypressEvents } from "node:readline";
import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { Writable } from "node:stream";

export async function promptVisibleInput(
  message: string,
  input: NodeJS.ReadableStream = stdin,
  output: NodeJS.WritableStream = stdout,
): Promise<string> {
  return await promptInput(message, false, input, output);
}

export async function promptHiddenInput(
  message: string,
  input: NodeJS.ReadableStream = stdin,
  output: NodeJS.WritableStream = stdout,
): Promise<string> {
  return await promptInput(message, true, input, output);
}

/**
 * Pause() does not detach stdin keypress handlers. A nested readline on the
 * same TTY otherwise receives each key twice (op → oopp).
 */
export async function withDetachedKeypress<T>(
  input: NodeJS.EventEmitter,
  run: () => Promise<T>,
): Promise<T> {
  const listeners = [...input.listeners("keypress")];
  input.removeAllListeners("keypress");
  try {
    return await run();
  } finally {
    input.removeAllListeners("keypress");
    for (const listener of listeners) {
      input.on("keypress", listener);
    }
  }
}

async function promptInput(
  message: string,
  hidden: boolean,
  input: NodeJS.ReadableStream,
  output: NodeJS.WritableStream,
): Promise<string> {
  if (
    (input as { isTTY?: boolean }).isTTY !== true
    || (output as { isTTY?: boolean }).isTTY !== true
  ) {
    throw new Error("Provider login needs an interactive tt session.");
  }

  return await withDetachedKeypress(input, async () => {
    if (!hidden) {
      const rl = createInterface({ input, output, terminal: true });
      try {
        return await rl.question(message);
      } finally {
        rl.close();
      }
    }

    let muted = false;
    const maskedOutput = new Writable({
      write(chunk, _encoding, callback) {
        if (!muted) output.write(chunk);
        callback();
      },
    });
    const rl = createInterface({
      input,
      output: maskedOutput,
      terminal: true,
    });

    try {
      const pending = rl.question(message);
      muted = true;
      return await pending;
    } finally {
      muted = false;
      output.write("\n");
      rl.close();
    }
  });
}

export type KeyChoice = "approve" | "reject" | "later";

interface Keypress {
  name?: string;
  ctrl?: boolean;
  sequence?: string;
}

/** Map one keypress to an approval decision; unknown keys return null. */
export function keyChoiceFor(key: Keypress | undefined, text?: string): KeyChoice | null {
  const name = (key?.name ?? text ?? "").toLowerCase();
  if (key?.ctrl && (name === "c" || name === "d")) return "later";
  if (name === "y") return "approve";
  if (name === "n") return "reject";
  if (name === "escape" || name === "l") return "later";
  return null;
}

const CHOICE_ECHO: Record<KeyChoice, string> = {
  approve: "yes",
  reject: "no",
  later: "later",
};

/**
 * Wait for a single keypress: y approves, n rejects, Esc (or Ctrl-C) defers.
 * Other keys are ignored so a stray Enter cannot approve a change.
 */
export async function promptKeyChoice(
  message: string,
  input: NodeJS.ReadableStream = stdin,
  output: NodeJS.WritableStream = stdout,
): Promise<KeyChoice> {
  const terminal = input as NodeJS.ReadableStream & {
    isTTY?: boolean;
    isRaw?: boolean;
    setRawMode?(mode: boolean): unknown;
  };
  if (terminal.isTTY !== true || typeof terminal.setRawMode !== "function") return "later";
  emitKeypressEvents(input);
  return await withDetachedKeypress(input, async () => {
    const wasRaw = terminal.isRaw === true;
    try {
      terminal.setRawMode!(true);
      input.resume();
      output.write(`${message} ${hint()}`);
      const choice = await new Promise<KeyChoice>((resolve) => {
        const finish = (decision: KeyChoice) => {
          input.removeListener("keypress", onKeypress);
          input.removeListener("end", onEnd);
          input.removeListener("close", onEnd);
          resolve(decision);
        };
        const onKeypress = (text: string | undefined, key: Keypress | undefined) => {
          const decided = keyChoiceFor(key, text);
          if (decided) finish(decided);
        };
        const onEnd = () => finish("later");
        input.on("keypress", onKeypress);
        input.once("end", onEnd);
        input.once("close", onEnd);
      });
      output.write(`\r\u001b[2K${message} ${CHOICE_ECHO[choice]}\n`);
      return choice;
    } finally {
      if (!wasRaw) terminal.setRawMode!(false);
    }
  });
}

function hint(): string {
  return chalk.dim("y yes · n no · esc decide later ");
}
