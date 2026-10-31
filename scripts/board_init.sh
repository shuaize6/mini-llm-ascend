#!/bin/bash
# 初始化标准项目目录：源码、.git、.venv、models、results 均位于同一根目录。
# 用法：bash scripts/board_init.sh [/root/mini-llm-ascend]
set -euo pipefail

WORK="${1:-/root/mini-llm-ascend}"
case "$WORK" in
    /*) ;;
    *) echo '项目路径必须为绝对路径' >&2; exit 1 ;;
esac
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd -P)"

if [ ! -e "$WORK/.git" ]; then
    git init -b main "$WORK"
fi
if [ ! -d "$WORK/.git" ] || [ "$(git -C "$WORK" rev-parse --is-bare-repository)" != false ]; then
    echo '需要 .git 位于项目内部的普通 Git 仓库' >&2
    exit 1
fi
# 接收推送时由 Git 更新当前分支与工作副本；已跟踪文件有未提交改动时拒绝更新。
git -C "$WORK" config receive.denyCurrentBranch updateInstead
mkdir -p "$WORK/models" "$WORK/results"

if [ ! -e "$WORK/.venv" ]; then
    /usr/local/miniconda3/bin/python -m venv --system-site-packages --prompt "$(basename "$WORK")" "$WORK/.venv"
fi

echo "项目目录：$WORK"
echo "Git 数据：$WORK/.git"
echo "Python：$WORK/.venv/bin/python"
echo "模型：$WORK/models"
echo '初始化完成；模型加载依赖按 requirements-m0.txt 安装。'
