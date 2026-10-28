; Build through scripts/build-windows.ps1. Keep AppId stable across versions.
#if Ver < EncodeVer(7, 0, 0)
  #error Inno Setup 7 is required for long dependency paths.
#endif

[Setup]
SetupArchitecture=x64
AppId=TunedTensor.CLI
AppName=Tuned Tensor CLI
AppVersion={#Version}
AppPublisher=Tuned Tensor
AppPublisherURL=https://tunedtensor.com
AppSupportURL=https://github.com/tunedtensor/tuned-tensor-cli/issues
DefaultDirName={localappdata}\Programs\TunedTensor
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
MinVersion=10.0
OutputDir={#Output}
OutputBaseFilename=tuned-tensor-cli-{#Version}-windows-x64
Compression=lzma2
SolidCompression=yes
ChangesEnvironment=yes
UninstallDisplayName=Tuned Tensor CLI
CloseApplications=no
LicenseFile=..\..\LICENSE

[Files]
Source: "{#Payload}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; This directory contains only managed application files. User state lives in
; ~/.tuned-tensor and is never removed by setup or uninstall.
Type: filesandordirs; Name: "{app}\node_modules"
Type: filesandordirs; Name: "{app}\runtime"

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not FileExists(ExpandConstant('{app}\tt-install.json')) and
     (DirExists(ExpandConstant('{app}\node_modules')) or
      DirExists(ExpandConstant('{app}\runtime')) or
      FileExists(ExpandConstant('{app}\tt.cmd'))) then
    Result := 'This directory contains files not managed by Tuned Tensor setup. Choose an empty install directory.';
end;

function PathContains(Value, Entry: String): Boolean;
begin
  Result := Pos(';' + Lowercase(Entry) + ';', ';' + Lowercase(Value) + ';') > 0;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Value, Entry: String;
begin
  if CurStep = ssPostInstall then begin
    Entry := ExpandConstant('{app}');
    RegQueryStringValue(HKCU, 'Environment', 'Path', Value);
    if not PathContains(Value, Entry) then begin
      if (Value <> '') and (Copy(Value, Length(Value), 1) <> ';') then
        Value := Value + ';';
      RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Value + Entry);
      RegWriteDWordValue(HKCU, 'Software\TunedTensor\CLI', 'AddedToPath', 1);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Value, Entry, Part, Remaining: String;
  Added, Separator: Cardinal;
begin
  if (CurUninstallStep = usPostUninstall) and
     RegQueryDWordValue(HKCU, 'Software\TunedTensor\CLI', 'AddedToPath', Added) and
     (Added = 1) then begin
    Entry := ExpandConstant('{app}');
    RegQueryStringValue(HKCU, 'Environment', 'Path', Value);
    Remaining := '';
    while Value <> '' do begin
      Separator := Pos(';', Value);
      if Separator = 0 then Separator := Length(Value) + 1;
      Part := Copy(Value, 1, Separator - 1);
      Delete(Value, 1, Separator);
      if (CompareText(Part, Entry) <> 0) and (Part <> '') then begin
        if Remaining <> '' then Remaining := Remaining + ';';
        Remaining := Remaining + Part;
      end;
    end;
    RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Remaining);
    RegDeleteValue(HKCU, 'Software\TunedTensor\CLI', 'AddedToPath');
  end;
end;
