$logsDir = "D:\river\ovmx_dev\roofline\results\seedvr2_2560x1536"
$files = Get-ChildItem -Path $logsDir -Filter "*.log"
Write-Output ("Total log files: {0}" -f $files.Count)
$files | Sort-Object LastWriteTime | ForEach-Object {
  Write-Output ("{0,-40} {1} {2,10} bytes" -f $_.BaseName, $_.LastWriteTime, $_.Length)
}
