# Install all dependencies except packages unavailable on Windows
$reqs = Get-Content requirements.txt | Where-Object { $_ -notmatch '^mxnet' -and $_ -notmatch '^tensorflow-io-gcs-filesystem' }
$tmpReqs = "$env:TEMP\requirements_win.txt"
$reqs | Out-File -FilePath $tmpReqs -Encoding utf8
pip install -r $tmpReqs

# mxnet 1.9.1 has no Windows wheel — latest available version is 1.7.0.post2.
# --no-deps bypasses the numpy<1.17 conflict (fixed at runtime by the patch below).
pip install mxnet==1.7.0.post2 --no-deps

# darts installed without deps due to numpy version conflict
pip install darts==0.42.1 --no-deps

# On Windows, tensorflow is split into three packages: tensorflow (stub) -> tensorflow-cpu (stub) -> tensorflow-intel (actual implementation)
pip install tensorflow==2.12.0 --no-deps
pip install tensorflow-cpu==2.12.0 --no-deps
pip install tensorflow-intel==2.12.0 --no-deps

# Patch mxnet for compatibility with numpy 1.26.4 — mxnet uses np.bool which was removed in numpy 1.24
$sitePackages = python -c "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"
$utilsPath = Join-Path $sitePackages "mxnet\numpy\utils.py"

$content = Get-Content $utilsPath -Raw
$content = $content -replace 'onp\.bool\b', 'bool'
$content = $content -replace '(?m)^bool_ = bool_$', 'bool_ = np.bool_'
$content = $content -replace 'bool_ = np\.bool_', 'bool_ = onp.bool_'
Set-Content $utilsPath $content -NoNewline

python -c "import mxnet; print('mxnet ok')"