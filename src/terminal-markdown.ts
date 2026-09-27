import chalk from "chalk";
import { stripVTControlCharacters } from "node:util";

const accent = chalk.hex("#8B5CF6");
/* Code spans use the lighter brand violet for readability on dark terminals. */
const code = chalk.hex("#A78BFA");

/** Remove terminal control characters while preserving normal whitespace. */
export function sanitizeTerminalText(text: string): string {
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
        rendered += code(text.slice(index + 1, close));
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
  if (inCodeBlock) return code(`  ${line}`);

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

function isWideCodePoint(code: number): boolean {
  return code >= 0x1100 && (
    code <= 0x115f
    || (code >= 0x2e80 && code <= 0xa4cf)
    || (code >= 0xac00 && code <= 0xd7a3)
    || (code >= 0xf900 && code <= 0xfaff)
    || (code >= 0xfe30 && code <= 0xfe4f)
    || (code >= 0xff00 && code <= 0xff60)
    || (code >= 0xffe0 && code <= 0xffe6)
    || (code >= 0x1f300 && code <= 0x1faff)
    || (code >= 0x20000 && code <= 0x3fffd)
  );
}

/** Approximate terminal cell width of text, ignoring ANSI styling. */
export function terminalWidth(text: string): number {
  let width = 0;
  for (const char of stripVTControlCharacters(text)) {
    width += isWideCodePoint(char.codePointAt(0)!) ? 2 : 1;
  }
  return width;
}

/**
 * Word-wrap one styled line to `width` cells. List items keep a hanging
 * indent so wrapped text lines up under the item rather than its marker.
 * Words wider than the line are left for the terminal to break.
 */
export function wrapTerminalLine(line: string, width: number): string[] {
  if (width < 20 || terminalWidth(line) <= width) return [line];
  const plain = stripVTControlCharacters(line);
  const hang = plain.match(/^(\s*(?:[•│]|\d+\.)?\s*)/)?.[1]?.length ?? 0;
  const continuation = " ".repeat(hang < width / 2 ? hang : 0);
  const lines: string[] = [];
  let current = "";
  let currentWidth = 0;
  let started = false;
  for (const word of line.split(" ")) {
    const wordWidth = terminalWidth(word);
    const hasContent = stripVTControlCharacters(current).trim().length > 0;
    if (started && hasContent && wordWidth > 0 && currentWidth + 1 + wordWidth > width) {
      lines.push(current);
      current = continuation + word;
      currentWidth = continuation.length + wordWidth;
      continue;
    }
    current += (started ? " " : "") + word;
    currentWidth += (started ? 1 : 0) + wordWidth;
    started = true;
  }
  lines.push(current);
  return lines;
}
