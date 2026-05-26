$path = "d:\Projects\DLEnhance_ImageQuality\results\conference_101719.tex"
$content = Get-Content -Path $path -Raw
$content = $content -replace '\\section\{Introduction\}', "`n`n\section{Introduction}`n`n"
Set-Content -Path $path -Value $content -NoNewline
