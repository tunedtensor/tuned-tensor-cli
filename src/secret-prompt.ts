import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { Writable } from "node:stream";

export async function promptHiddenInput(
  message: string,
  input: NodeJS.ReadableStream = stdin,
  output: NodeJS.WritableStream = stdout,
): Promise<string> {
  if (
    (input as { isTTY?: boolean }).isTTY !== true
    || (output as { isTTY?: boolean }).isTTY !== true
  ) {
    throw new Error("Provider login needs an interactive tt session.");
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
