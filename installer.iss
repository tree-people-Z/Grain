; Inno Setup script for Grain — builds a per-user installer (no admin needed).
;
;   "C:\Users\<you>\AppData\Local\Programs\InnoSetup6\ISCC.exe" installer.iss
;
; Expects the PyInstaller output to already exist in dist\Grain\ (run
;   python -m PyInstaller --noconfirm --clean Grain.spec
; first). Installer lands in dist\installer\.

#define AppName "Grain"
#define AppVersion "1.0.0"
#define AppPublisher "Grain"

[Setup]
AppId={{8E2B7C64-3B1A-4F2E-9D6C-5A4F1E2D7B90}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install: writable target dir, no elevation. data\ and engines\ are
; created next to the exe by the app, which is why we stay out of Program Files.
PrivilegesRequired=lowest
OutputDir=dist\installer
OutputBaseFilename=Grain-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\Grain.exe
VersionInfoVersion=1.0.0.0
VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} 安装程序

[Languages]
Name: "chinesesimplified"; MessagesFile: "installer\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\Grain\Grain.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\Grain\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\Grain.exe"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\Grain.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Grain.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
