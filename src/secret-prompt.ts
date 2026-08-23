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
}
