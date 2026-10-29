/** Windows cannot deliver catchable POSIX signals with child.kill(). */
export function listenForParentSignals(): void {
  if (process.platform !== "win32" || process.env.TT_PARENT_IPC !== "1" || !process.channel) return;
  process.on("message", message => {
    if (!message || typeof message !== "object" || !("ttSignal" in message)) return;
    if (message.ttSignal === "SIGINT" || message.ttSignal === "SIGTERM") process.emit(message.ttSignal);
  });
  process.once("disconnect", () => {
    if (!process.emit("SIGTERM")) process.exit(143);
  });
  // An idle IPC channel must not keep one-shot commands alive.
  process.channel.unref();
}
