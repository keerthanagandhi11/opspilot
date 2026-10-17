#!/usr/bin/env bash
# Deploy the current commit to a Hugging Face Docker Space.
#
#   ./scripts/deploy_to_hf.sh <hf-username>/<space-name>
#
# Hugging Face needs two things GitHub doesn't:
#   1. a YAML header at the top of README.md (deploy/hf_space_header.md), and
#   2. no binary files outside Git LFS, so the sample PDFs are left out here;
#      the Dockerfile regenerates them during the image build.
# When git asks for a password, paste a Hugging Face *write* token.
# In CI, set HF_TOKEN and the token is used automatically.
set -euo pipefail

SPACE="${1:?usage: $0 <hf-username>/<space-name>}"
ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$TMP"
rm -f "$TMP"/backend/app/sample_data/*.pdf
cat "$ROOT/deploy/hf_space_header.md" "$ROOT/README.md" > "$TMP/README.md"

cd "$TMP"
git init -q -b main
git add -A
git -c user.name="opspilot-deploy" -c user.email="deploy@users.noreply.github.com" \
  commit -q -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"

if [ -n "${HF_TOKEN:-}" ]; then
  URL="https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}"
else
  URL="https://huggingface.co/spaces/${SPACE}"
fi
git push --force "$URL" main
echo "Deployed. Your app: https://$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]_' '[:lower:]-').hf.space"
