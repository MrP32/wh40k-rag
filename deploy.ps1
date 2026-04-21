param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("app", "ingest", "test", "commit")]
    [string]$Action,

    [Parameter(Mandatory=$true)]
    [string]$Message
)

if ([string]::IsNullOrWhiteSpace($Message)) {
    Write-Host "`n  ERROR: A commit message is required." -ForegroundColor Red
    Write-Host "  Usage: .\deploy.ps1 -Action app -Message `"your message here`"`n" -ForegroundColor Yellow
    exit 1
}

if ($Message.Length -lt 10) {
    Write-Host "`n  ERROR: Commit message too short. Please be descriptive (min 10 characters)." -ForegroundColor Red
    Write-Host "  Example: .\deploy.ps1 -Action app -Message `"Fix Librarius Conclave retrieval`"`n" -ForegroundColor Yellow
    exit 1
}

$ProjectDir = "C:\Projects\wh40k-app"
$VenvActivate = "$ProjectDir\.venv\Scripts\Activate.ps1"

function Write-Step($text) {
    Write-Host "`n>> $text" -ForegroundColor Cyan
}
function Write-Success($text) {
    Write-Host "   OK: $text" -ForegroundColor Green
}
function Write-Fail($text) {
    Write-Host "   FAILED: $text" -ForegroundColor Red
    exit 1
}

Write-Step "Navigating to project directory"
Set-Location $ProjectDir
Write-Success $ProjectDir

Write-Step "Staging all changes"
git add .
if ($LASTEXITCODE -ne 0) { Write-Fail "git add failed" }
Write-Success "All changes staged"

Write-Step "Committing: $Message"
$status = git status --porcelain
if ($status) {
    git commit -m $Message
    if ($LASTEXITCODE -ne 0) { Write-Fail "git commit failed" }
    Write-Success "Committed"
} else {
    Write-Host "   Nothing to commit — working tree clean" -ForegroundColor Yellow
}

Write-Step "Pushing to GitHub"
git push
if ($LASTEXITCODE -ne 0) { Write-Fail "git push failed" }
Write-Success "Pushed to origin/main"

Write-Step "Pulling latest from GitHub"
git pull
if ($LASTEXITCODE -ne 0) { Write-Fail "git pull failed" }
Write-Success "Up to date with origin/main"

Write-Step "Activating virtual environment"
& $VenvActivate
Write-Success "Virtual environment active"

switch ($Action) {
    "app" {
        Write-Step "Starting web app"
        Write-Host "   App will be available at: http://localhost:8000" -ForegroundColor Yellow
        Write-Host "   Press Ctrl+C to stop`n" -ForegroundColor Yellow
        uvicorn main:app --reload
    }
    "ingest" {
        Write-Step "Running ingestion pipeline"
        Write-Host "   This will take 30-90 minutes...`n" -ForegroundColor Yellow
        python ingest.py
        if ($LASTEXITCODE -ne 0) { Write-Fail "Ingest failed" }
        Write-Success "Ingestion complete"
    }
    "test" {
        Write-Step "Running retrieval test suite"
        python test_retrieval.py
        if ($LASTEXITCODE -ne 0) { Write-Fail "Tests failed" }
        Write-Success "Test suite complete"
    }
    "commit" {
        Write-Success "Commit and push complete — no action to run"
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Done." -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan
