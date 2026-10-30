# 从笔记本部署到香橙派板子
# 用法：
#   .\scripts\deploy.ps1           # git push 到板上裸库（推荐，版本化）
#   .\scripts\deploy.ps1 -Rsync    # 直接 rsync 当前工作区（含未提交改动）
#   .\scripts\deploy.ps1 -Push     # 同时推送到 origin（如已配置 GitHub）
param(
    [switch]$Rsync,
    [switch]$Push,
    [string]$Board = "root@10.201.166.209",
    [string]$AskpassDir = "E:\xiangcehngpai_shenteng\.ssh_probe"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

# 复用已配置好的 askpass（含密码），避免交互输入
$env:SSH_ASKPASS = Join-Path $AskpassDir 'askpass.cmd'
$env:SSH_ASKPASS_REQUIRE = 'force'
$env:DISPLAY = 'localhost:0'
# 注意：必须用正斜杠。反斜杠会被 git/ssh 的命令行解析吃掉
# 导致 ssh 把 known_hosts 写到奇怪的文件名下
$kh = (Join-Path $AskpassDir 'known_hosts') -replace '\\', '/'

$sshCommon = "ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=$kh -o ConnectTimeout=8 -o LogLevel=ERROR"

if ($Rsync) {
    $env:RSYNC_RSH = "$sshCommon -l root"
    Write-Host "==> rsync 工作区 -> ${Board}:/root/vllm_shengteng/" -ForegroundColor Cyan
    & rsync -az --delete `
        --exclude '.git/' --exclude '__pycache__/' --exclude '.model/' `
        --exclude '*.safetensors' --exclude 'build/' --exclude 'build_out/' `
        "$repoRoot/" "${Board}:/root/vllm_shengteng/"
    if ($LASTEXITCODE -ne 0) { throw "rsync 失败 (exit $LASTEXITCODE)" }
}
else {
    $env:GIT_SSH_COMMAND = $sshCommon
    Push-Location $repoRoot
    try {
        Write-Host "==> git push board main" -ForegroundColor Cyan
        & git push board main
        if ($LASTEXITCODE -ne 0) { throw "git push 失败 (exit $LASTEXITCODE)" }
        if ($Push) {
            Write-Host "==> git push origin main" -ForegroundColor Cyan
            & git push origin main
        }
    }
    finally { Pop-Location }
}

Write-Host "==> 部署完成：板上工作区 /root/vllm_shengteng" -ForegroundColor Green