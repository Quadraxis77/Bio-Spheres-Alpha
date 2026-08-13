param(
    [ValidateSet("smoke", "full")]
    [string]$Preset = "smoke",
    [int]$Warmup = 5,
    [int]$Samples = 50,
    [string]$OutputDirectory = "",
    [ValidateSet("specialized", "general", "integrated")]
    [string]$Strategy = "integrated",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$shapes = @("chain", "star", "balanced", "many-pairs", "gameplay-mixed", "dense-mechanical")
$workloads = if ($Preset -eq "full") {
    @("silent", "one-source", "vec4", "all-sparse", "every-cell", "cognocytes", "memorocytes", "saturated", "cancellation", "oscillators", "heat")
} else {
    @("one-source", "every-cell", "heat")
}
$cellCounts = @(20000, 100000, 200000)

Push-Location $repositoryRoot
try {
    if (-not $SkipBuild) {
        cargo build --release --features signal-backbone-bench --bin signal-backbone-bench
        if ($LASTEXITCODE -ne 0) { throw "release benchmark build failed" }
    }
    $benchmarkBinary = Get-Item -LiteralPath (Join-Path $repositoryRoot "target\release\signal-backbone-bench.exe") -ErrorAction SilentlyContinue
    if (-not $benchmarkBinary) { throw "release benchmark executable was not found" }

    if ($OutputDirectory) {
        New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    }

    foreach ($cellCount in $cellCounts) {
        foreach ($shape in $shapes) {
            foreach ($workload in $workloads) {
                $arguments = @(
                    "--strategy", $Strategy,
                    "--cells", $cellCount,
                    "--warmup", $Warmup,
                    "--samples", $Samples,
                    "--block-size", 64,
                    "--shape", $shape,
                    "--workload", $workload
                )
                if ($OutputDirectory) {
                    $log = Join-Path $OutputDirectory "$cellCount-$shape-$workload.txt"
                    & $benchmarkBinary.FullName @arguments 2>&1 | Tee-Object -FilePath $log
                } else {
                    & $benchmarkBinary.FullName @arguments
                }
                if ($LASTEXITCODE -ne 0) {
                    throw "benchmark failed: cells=$cellCount shape=$shape workload=$workload"
                }
            }
        }
    }
} finally {
    Pop-Location
}
