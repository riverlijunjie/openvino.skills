Get-ChildItem 'D:\river\ovmx_dev\roofline\results\seedvr2_2560x1536' -Filter '*.log' | Where-Object { $_.Length -eq 0 } | Select-Object Name,Length
Write-Output "---zero-byte check done---"
