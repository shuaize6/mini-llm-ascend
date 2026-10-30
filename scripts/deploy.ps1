# 将已提交源码部署到 board 远端；认证由本机 SSH 配置负责。
# 用法：
#   .\scripts\deploy.ps1
#   .\scripts\deploy.ps1 -Push                 # 部署后同时推送 GitHub origin
#   .\scripts\deploy.ps1 -Rsync -Board user@host # 同步当前工作区，需要 rsync
param(
    [switch]$Rsync,
    [switch]$Push,
    [string]$Board,
    [string]$RemotePath = '/root/vllm_shengteng/'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

if ($Rsync) {
    if ([string]::IsNullOrWhiteSpace($Board)) {
        throw '使用 -Rsync 时需要显式指定 -Board user@host。'
    }
    # 不删除远端文件；忽略文件和凭据不进入同步集合。
    Write-Host "同步源码到 ${Board}:${RemotePath}"
    & rsync -az --filter=':- .gitignore' `
        --exclude='.git/' --exclude='.deployed_commit' `
        "$repoRoot/" "${Board}:${RemotePath}"
    if ($LASTEXITCODE -ne 0) { throw "rsync 失败 (exit $LASTEXITCODE)" }
}
else {
    Push-Location $repoRoot
    try {
        & git push board main
        if ($LASTEXITCODE -ne 0) { throw "板端推送失败 (exit $LASTEXITCODE)" }
    }
    finally { Pop-Location }
}

if ($Push) {
    Push-Location $repoRoot
    try {
        & git push origin main
        if ($LASTEXITCODE -ne 0) { throw "GitHub 推送失败 (exit $LASTEXITCODE)" }
    }
    finally { Pop-Location }
}

Write-Host '源码同步完成。'
