Get-Process -Id 10800,12352 -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,TotalProcessorTime
