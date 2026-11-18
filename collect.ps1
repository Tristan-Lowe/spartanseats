# One live snapshot of each open term. Schedule every 4 hours during registration.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
& "$root\.venv\Scripts\python.exe" -m spartanseats.fetch live spring-2027 *>> "$root\data\collect.log"
