#!/bin/sh
# save as find-deps.sh and run inside your container (or docker run ...)

BIN=${1:-/app/DagCholeskyWorker}

if [ ! -x "$BIN" ]; then
  echo "Binaire $BIN introuvable ou non exécutable"
  exit 1
fi

echo "📦 Dépendances dynamiques pour $BIN"
echo "===================================="

# 1) Liste statique via ldd
echo "\n[ldd]"
ldd "$BIN" | awk '{print $3}' | grep '^/' | sort -u

# 2) Liste dynamique au runtime avec LD_DEBUG
echo "\n[LD_DEBUG=libs]"
LD_DEBUG=libs "$BIN" 2>&1 | grep '=>' | awk '{print $3}' | grep '^/' | sort -u

# 3) Liste runtime avec strace (ouvre tous les .so via openat)
echo "\n[strace]"
strace -e openat "$BIN" 2>&1 | grep '\.so' | sed -E 's/.*"([^"]+)".*/\1/' | sort -u

echo "\n✅ Combine ces trois listes = ton set minimal de libs à copier dans l'image runtime."
