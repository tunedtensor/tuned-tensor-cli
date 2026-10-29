import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const [installer, output] = process.argv.slice(2);
if (!installer || !output) throw new Error('Usage: node scripts/generate-winget.mjs <installer.exe> <output-dir>');
const { version } = JSON.parse(readFileSync(fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8'));
if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) throw new Error('Invalid package version');
// Prereleases remain downloadable, but are never offered on WinGet's stable channel.
if (version.includes('-')) process.exit(0);
const filename = `tuned-tensor-cli-${version}-windows-x64.exe`;
if (basename(installer) !== filename) throw new Error(`Expected installer ${filename}`);
const sha256 = createHash('sha256').update(readFileSync(installer)).digest('hex').toUpperCase();
const directory = join(output, 'manifests', 't', 'TunedTensor', 'CLI', version);
mkdirSync(directory, { recursive: true });
const common = `PackageIdentifier: TunedTensor.CLI\nPackageVersion: ${version}\n`;
const schema = 'ManifestVersion: 1.6.0\n';
const write = (suffix, body) => writeFileSync(join(directory, `TunedTensor.CLI${suffix}.yaml`), common + body + schema);
write('', 'DefaultLocale: en-US\nManifestType: version\n');
write('.locale.en-US', `PackageLocale: en-US
Publisher: Tuned Tensor
PublisherUrl: https://tunedtensor.com
PublisherSupportUrl: https://github.com/tunedtensor/tuned-tensor-cli/issues
PackageName: Tuned Tensor CLI
PackageUrl: https://github.com/tunedtensor/tuned-tensor-cli
License: Apache-2.0
LicenseUrl: https://github.com/tunedtensor/tuned-tensor-cli/blob/v${version}/LICENSE
ShortDescription: Local agent and CLI for Tuned Tensor workflows
Moniker: tt
ManifestType: defaultLocale
`);
write('.installer', `InstallerType: inno
Scope: user
MinimumOSVersion: 10.0.0.0
UpgradeBehavior: install
Commands:
  - tt
Installers:
  - Architecture: x64
    InstallerUrl: https://github.com/tunedtensor/tuned-tensor-cli/releases/download/v${version}/${filename}
    InstallerSha256: ${sha256}
    ProductCode: TunedTensor.CLI_is1
ManifestType: installer
`);
