#!/bin/bash
# 在板子上初始化"部署仓库"：裸库 + post-receive 钩子自动 checkout 到工作目录
# 用法（板子上执行一次）：bash scripts/board_init.sh
set -e

BARE=/root/nanoserve.git
WORK=/root/nanoserve
BRANCH=main

echo "==> 创建裸库 $BARE"
git init --bare "$BARE" >/dev/null

echo "==> 设置默认分支为 $BRANCH"
git --git-dir="$BARE" symbolic-ref HEAD "refs/heads/$BRANCH"

echo "==> 安装 post-receive 钩子"
cat > "$BARE/hooks/post-receive" <<'HOOK'
#!/bin/bash
set -e
GIT_DIR=/root/nanoserve.git
WORK_TREE=/root/nanoserve
BRANCH=main

mkdir -p "$WORK_TREE"
git --git-dir="$GIT_DIR" --work-tree="$WORK_TREE" checkout -f "$BRANCH"
COMMIT=$(git --git-dir="$GIT_DIR" rev-parse --short HEAD)
echo "[deploy] 已部署 $COMMIT -> $WORK_TREE"

# 记录最近一次部署，便于出问题时回滚
git --git-dir="$GIT_DIR" rev-parse HEAD > "$WORK_TREE/.deployed_commit"
HOOK

chmod +x "$BARE/hooks/post-receive"

echo "==> 准备模型目录"
mkdir -p /root/models

echo "==> 完成"
echo "   裸库:   $BARE"
echo "   工作区: $WORK （push 后自动更新）"
echo "   模型:   /root/models/ （权重不进 git）"