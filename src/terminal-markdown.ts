import chalk from "chalk";
import { stripVTControlCharacters } from "node:util";

const accent = chalk.hex("#8B5CF6");

/** Remove terminal control characters while preserving normal whitespace. */
function sanitizeTerminalText(text: string): string {
  return stripVTControlCharacters(text)
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, "");
}

function closingDelimiter(text: string, delimiter: string, from: number): number {
  return text.indexOf(delimiter, from + delimiter.length);
}

function renderInlineMarkdown(text: string): string {
  let rendered = "";
  let index = 0;

  while (index < text.length) {
    if (text[index] === "\\" && index + 1 < text.length) {
      rendered += text[index + 1];
      index += 2;
      continue;
    }

    if (text[index] === "[") {
      const link = text.slice(index).match(/^\[([^\]]+)]\(([^)]+)\)/);
      if (link) {
        rendered += `${chalk.underline(renderInlineMarkdown(link[1]!))}${chalk.dim(` (${link[2]})`)}`;
        index += link[0].length;
        continue;
      }
    }

    const strong = text.startsWith("**", index)
      ? "**"
      : text.startsWith("__", index)
        ? "__"
        : null;
    if (strong) {
      const close = closingDelimiter(text, strong, index);
      if (close > index + strong.length) {
        rendered += chalk.bold(
          renderInlineMarkdown(text.slice(index + strong.length, close)),
        );
        index = close + strong.length;
        continue;
      }
    }

    if (text[index] === "`") {
      const close = closingDelimiter(text, "`", index);
      if (close > index + 1) {
        rendered += chalk.cyan(text.slice(index + 1, close));
        index = close + 1;
        continue;
      }
    }

    const emphasis = text[index] === "*" || text[index] === "_"
      ? text[index]!
      : null;
    if (emphasis) {
      const close = closingDelimiter(text, emphasis, index);
      if (close > index + 1) {
        rendered += chalk.italic(
          renderInlineMarkdown(text.slice(index + 1, close)),
        );
        index = close + 1;
        continue;
      }
    }

    rendered += text[index];
    index += 1;
  }

  return rendered;
}

function renderMarkdownLine(line: string, inCodeBlock: boolean): string {
  if (inCodeBlock) return chalk.cyan(`  ${line}`);

  const heading = line.match(/^\s{0,3}#{1,6}\s+(.+)$/);
  if (heading) return chalk.bold(renderInlineMarkdown(heading[1]!));

  const bullet = line.match(/^(\s*)[-+*]\s+(.+)$/);
  if (bullet) {
    return `${bullet[1]}${accent("•")} ${renderInlineMarkdown(bullet[2]!)}`;
  }

  const numbered = line.match(/^(\s*)(\d+)[.)]\s+(.+)$/);
  if (numbered) {
    return `${numbered[1]}${accent(`${numbered[2]}.`)} ${renderInlineMarkdown(numbered[3]!)}`;
  }

  const quote = line.match(/^\s*>\s?(.*)$/);
  if (quote) return `${chalk.dim("│")} ${chalk.italic(renderInlineMarkdown(quote[1]!))}`;

  if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
    return chalk.dim("─".repeat(32));
  }

  return renderInlineMarkdown(line);
}

/**
 * Render streamed Markdown one completed line at a time.
 *
 * Holding only the unfinished line keeps formatting tokens from leaking when
 * the transport splits `**bold**` or other Markdown across event boundaries.
 */
export class StreamingTerminalMarkdown {
  private pending = "";
  private inCodeBlock = false;

  push(delta: string): string {
    this.pending += delta;
    let rendered = "";
    let newline = this.pending.indexOf("\n");

    while (newline !== -1) {
      const line = this.pending.slice(0, newline).replace(/\r$/, "");
      this.pending = this.pending.slice(newline + 1);
      rendered += `${this.renderLine(line)}\n`;
      newline = this.pending.indexOf("\n");
    }

    return rendered;
  }

  flush(): string {
    if (!this.pending) return "";
    const line = this.pending.replace(/\r$/, "");
    this.pending = "";
    return this.renderLine(line);
  }

  reset(): void {
    this.pending = "";
    this.inCodeBlock = false;
  }

  private renderLine(line: string): string {
    const safeLine = sanitizeTerminalText(line);
    if (/^\s*```/.test(safeLine)) {
      this.inCodeBlock = !this.inCodeBlock;
      return "";
    }
    return renderMarkdownLine(safeLine, this.inCodeBlock);
  }
}
