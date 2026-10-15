Get-CimInstance Win32_Process -Filter "ProcessId=16524 or ProcessId=11180" | Select-Object ProcessId,ParentProcessId,Name,CommandLine,CreationDate | Format-List
Write-Output "---children of 16524---"
Get-CimInstance Win32_Process -Filter "ParentProcessId=16524" | Select-Object ProcessId,Name,CommandLine
