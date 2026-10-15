$c = Get-Content D:\river\ovmx_dev\roofline\results\seedvr2\_index.txt
$idx = ($c | Select-String 'RETILE START' | Select-Object -Last 1).LineNumber
$c[($idx-1)..($c.Count-1)]
