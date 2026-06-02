import { Command } from "commander";
import { registerModelServeCommand } from "./models.js";

export function registerServeCommand(parent: Command) {
  registerModelServeCommand(
    parent.command("serve"),
    parent,
    {
      description: "Serve a downloaded model with the Tuned Tensor local reference server",
    },
  );
}
