# Project Chimera - Split ZIP Creator
# Creates multiple ZIP files, each max 500MB

$projectPath = "."
$maxSizeMB = 500
$maxSizeBytes = $maxSizeMB * 1MB
$outputPrefix = "Project_Chimera_Part"
$currentPart = 1
$currentSize = 0
$currentFiles = @()

# Get all files to archive (excluding certain types)
$filesToArchive = Get-ChildItem -Path $projectPath -Recurse -File | Where-Object {
    $_.Name -notmatch "\.(zip|log|mp4|avi|mov|wav|mp3|srt|json)$" -and
    $_.Name -notmatch "^(Project_Chimera_Part|__pycache__)" -and
    $_.FullName -notmatch "\\data\\cache\\" -and
    $_.FullName -notmatch "\\assets\\videos\\" -and
    $_.FullName -notmatch "\\assets\\videos\\downloads\\"
} | Sort-Object Length -Descending

Write-Host "Starting split ZIP creation..." -ForegroundColor Green
Write-Host "Total files to archive: $($filesToArchive.Count)" -ForegroundColor Cyan
Write-Host "Max size per part: $maxSizeMB MB" -ForegroundColor Yellow
Write-Host ""

foreach ($file in $filesToArchive) {
    $fileSize = $file.Length

    # Check if adding this file would exceed the limit
    if (($currentSize + $fileSize) -gt $maxSizeBytes -and $currentFiles.Count -gt 0) {
        # Create current part
        $zipName = "$outputPrefix$($currentPart.ToString('00')).zip"
        Write-Host "Creating $zipName (Size: $([math]::Round($currentSize/1MB,2)) MB, Files: $($currentFiles.Count))" -ForegroundColor Green

        try {
            Compress-Archive -Path $currentFiles -DestinationPath $zipName -Force
            Write-Host "SUCCESS: $zipName created!" -ForegroundColor Green
        }
        catch {
            Write-Host "ERROR creating $zipName : $_" -ForegroundColor Red
        }

        # Reset for next part
        $currentPart++
        $currentSize = 0
        $currentFiles = @()
    }

    # Add file to current part
    $currentFiles += $file.FullName
    $currentSize += $fileSize
}

# Create final part if there are remaining files
if ($currentFiles.Count -gt 0) {
    $zipName = "$outputPrefix$($currentPart.ToString('00')).zip"
    Write-Host "Creating final part: $zipName (Size: $([math]::Round($currentSize/1MB,2)) MB, Files: $($currentFiles.Count))" -ForegroundColor Green

    try {
        Compress-Archive -Path $currentFiles -DestinationPath $zipName -Force
        Write-Host "SUCCESS: $zipName created!" -ForegroundColor Green
    }
    catch {
        Write-Host "ERROR creating $zipName : $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Split ZIP creation completed!" -ForegroundColor Green
Write-Host "Total parts created: $currentPart" -ForegroundColor Cyan

# Show summary of created files
Write-Host ""
Write-Host "Created ZIP files:" -ForegroundColor Yellow
Get-ChildItem "$outputPrefix*.zip" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length/1MB,2)
    Write-Host "   $($_.Name) - $sizeMB MB" -ForegroundColor White
}
