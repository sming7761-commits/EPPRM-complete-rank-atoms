$ErrorActionPreference='Stop'
Set-Location (Join-Path $PSScriptRoot '..')
foreach($d in @('gaussian','t5','exp_centered','skew_mixture')) { python 12_reproduction/reproduce_main.py calibrate $d 0 5000; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}}
python 12_reproduction/reproduce_main.py thresholds
foreach($d in @('gaussian','t5','exp_centered','skew_mixture')) {
 python 12_reproduction/reproduce_main.py raw $d health 0 1000
 python 12_reproduction/reproduce_main.py raw $d fault 0 500
 python 12_reproduction/reproduce_main.py run $d health 0 1000
 python 12_reproduction/reproduce_main.py run $d fault 0 500
 if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}
python 12_reproduction/analyze_main.py
python 12_reproduction/build_manifest.py
python 12_reproduction/verify_package.py --deep
