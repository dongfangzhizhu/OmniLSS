# Run all R consistency tests
$env:PYTHONPATH = "src"
$env:JAX_PLATFORMS = "cpu"

Write-Host "=== Running R Consistency Tests ===" -ForegroundColor Cyan

$tests = @(
    "tests.test_r_consistency_batch1_bc",
    "tests.test_r_consistency_batch3_4_remaining",
    "tests.test_r_consistency_batch5",
    "tests.test_r_consistency_batch6",
    "tests.test_r_consistency_batch7_8"
)

$passed = 0
$failed = 0
$skipped = 0

foreach ($test in $tests) {
    Write-Host "`n--- $test ---" -ForegroundColor Yellow
    $result = python -m unittest $test -v 2>&1
    Write-Host $result
    
    if ($result -match "OK") { $passed++ }
    elseif ($result -match "FAILED") { $failed++ }
    if ($result -match "skipped=(\d+)") { $skipped += [int]$matches[1] }
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor Red
Write-Host "Skipped: $skipped" -ForegroundColor Yellow
