interface PassthroughOptions {
  args: string[];
  json: boolean;
  color: boolean;
}

/**
 * Pull root presentation flags out of a passthrough namespace.
 *
 * Commander cannot know the option grammar of the local runtime, so
 * `tt --json runs list` and `tt runs list --json` both need to work.
 */
export function extractPassthroughOptions(
  args: readonly string[],
  root: { json?: boolean; color?: boolean },
): PassthroughOptions {
  let json = Boolean(root.json);
  let color = root.color !== false;
  const forwarded: string[] = [];

  for (const arg of args) {
    if (arg === "--json") {
      json = true;
    } else if (arg === "--no-color") {
      color = false;
    } else {
      forwarded.push(arg);
    }
  }

  return { args: forwarded, json, color };
}
