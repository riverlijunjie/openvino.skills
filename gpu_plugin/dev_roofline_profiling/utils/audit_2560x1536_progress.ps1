$logsDir = "D:\river\ovmx_dev\roofline\results\seedvr2_2560x1536"
if (Test-Path $logsDir) {
  $files = Get-ChildItem -Path $logsDir -Filter "*.log"
  Write-Output ("Log files present: {0}" -f $files.Count)
  $files | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | ForEach-Object {
    Write-Output ("{0,-45} {1} {2,10} bytes" -f $_.BaseName, $_.LastWriteTime, $_.Length)
  }
} else {
  Write-Output "LOGS DIR DOES NOT EXIST YET"
}
