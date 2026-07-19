#!/usr/bin/env bash
# One-shot deploy of the discovery-engine repo (research + web/) to GitHub Pages.
# Run this from the repo root:  bash scripts/deploy.sh
#
# It pushes to GitHub and the included Actions workflow (.github/workflows/
# deploy-pages.yml) builds + publishes web/ to Pages automatically.
set -euo pipefail

# ------------------------------------------------------------------ settings
# EDIT these two, or pass as env vars:  GH_USER=you REPO=discovery-engine bash scripts/deploy.sh
GH_USER="${GH_USER:-YOUR_GITHUB_USERNAME}"
REPO="${REPO:-discovery-engine}"
BRANCH=main
# ---------------------------------------------------------------------------

if [ "$GH_USER" = "YOUR_GITHUB_USERNAME" ]; then
  echo "Set GH_USER (and optionally REPO) first, e.g.:"
  echo "  GH_USER=levimonte REPO=discovery-engine bash scripts/deploy.sh"
  exit 1
fi

cd "$(dirname "$0")/.."

# 1) create the GitHub repo if you have the gh CLI (skip if you made it on the site)
if command -v gh >/dev/null 2>&1; then
  gh repo view "$GH_USER/$REPO" >/dev/null 2>&1 || \
    gh repo create "$GH_USER/$REPO" --public --disable-issues --description \
      "Uncertainty-driven materials discovery benchmark + interactive demo"
  # turn on Pages with GitHub Actions as the source
  gh api -X POST "repos/$GH_USER/$REPO/pages" -f build_type=workflow >/dev/null 2>&1 || true
else
  echo "No gh CLI found — make sure you've created an EMPTY repo at:"
  echo "  https://github.com/new  ->  name: $REPO  (public, no README)"
  echo "Then, after this push: repo Settings -> Pages -> Source: GitHub Actions."
fi

# 2) init/commit/push
git init -q 2>/dev/null || true
git checkout -q -B "$BRANCH"
git add -A
git commit -q -m "Discovery engine + interactive web demo" || echo "(nothing new to commit)"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GH_USER/$REPO.git"

echo ">> pushing to https://github.com/$GH_USER/$REPO (you'll be prompted for a Personal Access Token as the password)"
git push -u origin "$BRANCH"

echo ""
echo "Done. Watch the deploy at:  https://github.com/$GH_USER/$REPO/actions"
echo "Live site (after the green check): https://$GH_USER.github.io/$REPO/"
echo ""
echo "If Pages wasn't auto-enabled: repo Settings -> Pages -> Source: 'GitHub Actions', then re-run the workflow."
