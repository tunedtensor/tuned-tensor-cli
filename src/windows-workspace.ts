import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { join } from "node:path";

// Node cannot open Windows directory handles with FILE_SHARE_DELETE disabled.
// Keep those handles and every mutation in one native helper: a stopped helper
// cannot leave a later Node write following an unpinned workspace path.
const HELPER = String.raw`
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = New-Object Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using Microsoft.Win32.SafeHandles;
public static class TTWorkspace {
  [StructLayout(LayoutKind.Sequential)] struct Info {
    public uint Attributes, CreationLow, CreationHigh, AccessLow, AccessHigh,
      WriteLow, WriteHigh, Volume, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
  }
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern SafeFileHandle CreateFileW(string path, uint access, uint share,
    IntPtr security, uint disposition, uint flags, IntPtr template);
  [DllImport("kernel32.dll", SetLastError=true)]
  static extern bool GetFileInformationByHandle(SafeFileHandle handle, out Info info);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern bool MoveFileExW(string source, string target, uint flags);
  static string NativePath(string path) {
    path = Path.GetFullPath(path);
    return path.StartsWith(@"\\") ? @"\\?\UNC\" + path.Substring(2) : @"\\?\" + path;
  }
  static SafeFileHandle Open(string path, bool directory) {
    // No FILE_SHARE_DELETE: the directory cannot be renamed or replaced.
    var handle = CreateFileW(NativePath(path), 0x80000000u,
      directory ? 3u : 1u, IntPtr.Zero, 3, 0x02200000, IntPtr.Zero);
    if (handle.IsInvalid) { int error = Marshal.GetLastWin32Error(); handle.Dispose(); throw new Win32Exception(error); }
    Info info;
    if (!GetFileInformationByHandle(handle, out info)) {
      int error = Marshal.GetLastWin32Error(); handle.Dispose(); throw new Win32Exception(error);
    }
    if ((info.Attributes & 0x400) != 0 || (((info.Attributes & 0x10) != 0) != directory)) {
      handle.Dispose(); throw new IOException("Workspace entries must be real directories or regular files, not symlinks/reparse points.");
    }
    return handle;
  }
  public static SafeFileHandle DirectoryHandle(string path) { return Open(path, true); }
  public static string Child(string directory, string name) {
    if (String.IsNullOrEmpty(name) || name == "." || name == ".." || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
      throw new IOException("Invalid workspace entry name.");
    return Path.Combine(directory, name);
  }
  public static string Read(string path, int limit) {
    using (var handle = Open(path, false))
    using (var stream = new FileStream(handle, FileAccess.Read)) {
      if (stream.Length > limit) throw new IOException("Spec/revision exceeds the byte limit.");
      using (var bytes = new MemoryStream()) {
        var buffer = new byte[16384]; int count;
        while ((count = stream.Read(buffer, 0, buffer.Length)) > 0) {
          if (bytes.Length + count > limit) throw new IOException("Spec/revision exceeds the byte limit.");
          bytes.Write(buffer, 0, count);
        }
        return new UTF8Encoding(false, true).GetString(bytes.ToArray());
      }
    }
  }
  public static void WriteNew(string path, string source) {
    byte[] bytes = new UTF8Encoding(false).GetBytes(source);
    var security = new FileSecurity();
    security.SetAccessRuleProtection(true, false);
    security.AddAccessRule(new FileSystemAccessRule(WindowsIdentity.GetCurrent().User, FileSystemRights.FullControl, AccessControlType.Allow));
    using (var stream = new FileStream(NativePath(path), FileMode.CreateNew, FileSystemRights.Write, FileShare.Read, 4096, FileOptions.None, security)) {
      stream.Write(bytes, 0, bytes.Length); stream.Flush(true);
    }
  }
  public static FileStream LockFile(string path) {
    return new FileStream(NativePath(path), FileMode.CreateNew, FileAccess.ReadWrite,
      FileShare.None, 1, FileOptions.DeleteOnClose);
  }
  public static void Atomic(string directory, string name, string source) {
    string temporary = Child(directory, ".tt-" + Guid.NewGuid().ToString() + ".tmp");
    try {
      WriteNew(temporary, source);
      if (!MoveFileExW(NativePath(temporary), NativePath(Child(directory, name)), 9))
        throw new Win32Exception(Marshal.GetLastWin32Error());
    } finally { File.Delete(NativePath(temporary)); }
  }
}
'@
$handles = New-Object 'Collections.Generic.List[System.IDisposable]'
$directories = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
function Lock-Directory([string]$path) {
  $path = [IO.Path]::GetFullPath($path)
  $current = [IO.Path]::GetPathRoot($path)
  $parts = @('') + @($path.Substring($current.Length).Split('\') | Where-Object { $_ })
  foreach ($part in $parts) {
    if ($part) { $current = [IO.Path]::Combine($current, $part) }
    if (-not $directories.Contains($current)) {
      $handle = [TTWorkspace]::DirectoryHandle($current)
      $handles.Add($handle)
      [void]$directories.Add($current)
    }
  }
}
try {
  while ($null -ne ($line = [Console]::ReadLine())) {
    try {
      $request = $line | ConvertFrom-Json
      if ($request.op -eq 'directory') {
        Lock-Directory $request.directory
        $result = $null
      } else {
        if (-not $directories.Contains([IO.Path]::GetFullPath($request.directory))) { throw 'Directory is not pinned.' }
        $path = [TTWorkspace]::Child($request.directory, $request.name)
        switch ($request.op) {
          'read' { $result = [TTWorkspace]::Read($path, $request.limit) }
          'create' { [TTWorkspace]::WriteNew($path, $request.source); $result = $null }
          'atomic' { [TTWorkspace]::Atomic($request.directory, $request.name, $request.source); $result = $null }
          'lock' { $handles.Add([TTWorkspace]::LockFile($path)); $result = $null }
          'child' {
            [void][IO.Directory]::CreateDirectory($path)
            Lock-Directory $path
            $result = $path
          }
          default { throw 'Unknown workspace operation.' }
        }
      }
      [Console]::WriteLine((@{ ok = $true; result = $result } | ConvertTo-Json -Compress -Depth 3))
    } catch {
      $native = $_.Exception.GetBaseException()
      $number = if ($native -is [ComponentModel.Win32Exception]) { $native.NativeErrorCode } else { $native.HResult -band 65535 }
      $code = switch ($number) { 2 { 'ENOENT' } 3 { 'ENOENT' } 80 { 'EEXIST' } 183 { 'EEXIST' } default { 'EIO' } }
      [Console]::WriteLine((@{ ok = $false; error = $native.Message; code = $code } | ConvertTo-Json -Compress))
    }
  }
} finally {
  for ($i = $handles.Count - 1; $i -ge 0; $i--) { $handles[$i].Dispose() }
}
`;

export class WindowsWorkspace {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly exited: Promise<void>;
  private pending?: { resolve(value: unknown): void; reject(error: Error): void };
  private failure?: Error;
  private stderr = "";

  private constructor() {
    const executable = join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
    this.child = spawn(executable, ["-NoLogo", "-NoProfile", "-NonInteractive", "-OutputFormat", "Text", "-ExecutionPolicy", "Bypass",
      "-EncodedCommand", Buffer.from(HELPER, "utf16le").toString("base64")], { windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    const fail = (error: Error) => { this.failure = error; this.pending?.reject(error); this.pending = undefined; };
    this.child.on("error", fail);
    this.child.stdin.on("error", fail);
    this.child.stderr.on("data", data => { this.stderr = (this.stderr + String(data)).slice(-4000); });
    createInterface({ input: this.child.stdout }).on("line", line => {
      const pending = this.pending;
      if (!pending) return;
      this.pending = undefined;
      try {
        const response = JSON.parse(line) as { ok: boolean; result?: unknown; error?: string; code?: string };
        if (response.ok) pending.resolve(response.result);
        else pending.reject(Object.assign(new Error(response.error ?? "Windows workspace operation failed."), { code: response.code }));
      } catch { pending.reject(new Error("Invalid Windows workspace helper response.")); }
    });
    this.exited = new Promise(resolve => this.child.once("close", () => {
      fail(new Error(`Windows workspace helper stopped. ${this.stderr}`));
      resolve();
    }));
  }

  static async open(directory: string): Promise<WindowsWorkspace> {
    const workspace = new WindowsWorkspace();
    try { await workspace.request({ op: "directory", directory }); return workspace; }
    catch (error) { await workspace.close(); throw error; }
  }

  private async request(command: Record<string, unknown>): Promise<unknown> {
    if (this.failure) throw this.failure;
    if (this.pending) throw new Error("Concurrent operations on one Windows workspace transaction are not supported.");
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.child.kill(); reject(new Error("Windows workspace operation timed out.")); }, 30_000);
      this.pending = { resolve: value => { clearTimeout(timer); resolve(value); }, reject: error => { clearTimeout(timer); reject(error); } };
      this.child.stdin.write(JSON.stringify(command) + "\n");
    });
  }

  async read(directory: string, name: string, limit: number): Promise<string> {
    return await this.request({ op: "read", directory, name, limit }) as string;
  }
  async create(directory: string, name: string, source: string): Promise<void> {
    await this.request({ op: "create", directory, name, source });
  }
  async atomic(directory: string, name: string, source: string): Promise<void> {
    await this.request({ op: "atomic", directory, name, source });
  }
  async childDirectory(directory: string, name: string): Promise<string> {
    return await this.request({ op: "child", directory, name }) as string;
  }
  async lock(directory: string, name: string): Promise<void> {
    await this.request({ op: "lock", directory, name });
  }
  async close(): Promise<void> {
    this.child.stdin.end();
    const timer = setTimeout(() => this.child.kill(), 3000);
    try { await this.exited; } finally { clearTimeout(timer); }
  }
}
