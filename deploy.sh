#!/usr/bin/env zsh
# deploy.sh — ローカルプレビュー起動 + GitHub push を同時実行
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$REPO_DIR/docs"
PORT=8765

# ── コミットメッセージ ──────────────────────────────────
MSG="${1:-update: index.html}"

# ── 1. ローカルサーバー起動（既存プロセスを差し替え） ──
echo "\n🌐  ローカルサーバーを起動中 (port $PORT)..."
pkill -f "http-server.*$PORT" 2>/dev/null || true
sleep 0.3
npx --yes http-server "$DOCS_DIR" -p $PORT --cors -c-1 --silent &
SERVER_PID=$!

# サーバーが立ち上がるまで待機（最大3秒）
for i in {1..6}; do
  curl -s -o /dev/null "http://localhost:$PORT/" && break
  sleep 0.5
done

echo "   → http://localhost:$PORT/ を開きます"
open "http://localhost:$PORT/"

# ── 2. GitHub push ─────────────────────────────────────
echo "\n🚀  GitHub へ push 中..."
cd "$REPO_DIR"
# docs/ と主要設定ファイルのみ対象（巨大なデータファイルを除外）
git add docs/ .gitignore deploy.sh README.md 2>/dev/null || true
if git diff --cached --quiet; then
  echo "   変更なし — push をスキップしました"
else
  git commit -m "$MSG"
  git push
  echo "   ✅  push 完了！"
fi

echo "\nローカルサーバーは動き続けています (PID: $SERVER_PID)"
echo "停止するには: kill $SERVER_PID  または pkill -f 'http-server.*$PORT'"
