# One live snapshot of each open term. Schedule every 4 hours during registration.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
& "$root\.venv\Scripts\python.exe" -m spartanseats.fetch live spring-2027 2>&1 |
    Out-File -Append -Encoding utf8 "$root\data\collect.log"
