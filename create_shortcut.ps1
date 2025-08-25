$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "MedDRA Auto-Coding.lnk"

$shortcut = $ws.CreateShortcut($lnkPath)
$shortcut.TargetPath = Join-Path $PSScriptRoot "launch_app.bat"
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.WindowStyle = 1
$shortcut.Description = "Lancer l'interface MedDRA (Streamlit)"
# Generic app icon
$shortcut.IconLocation = "shell32.dll,220"
$shortcut.Save()

Write-Host "Raccourci cree: $lnkPath"
