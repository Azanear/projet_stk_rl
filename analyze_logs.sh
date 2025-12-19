#!/bin/bash

# Dossier par défaut "logs/", ou celui passé en premier argument
LOG_DIR="${1:-logs/}"

if [ ! -d "$LOG_DIR" ]; then
    echo "Erreur : Le dossier '$LOG_DIR' n'existe pas."
    exit 1
fi

echo "=========================================="
echo " 🏆 TOP PERFORMANCES (Podium : Pos <= 3)"
echo "=========================================="

{
    echo "EXP STEP TRACK REWARD POS"
    grep -r '"reward":' "$LOG_DIR" \
    | sed -E 's|'"$LOG_DIR"'/?([^/]+)/([^/]+)/.*"track": "([^"]+)".*"reward": ([0-9.-]+), "position": ([0-9]+).*|\1 \2 \3 \4 \5|' \
    | awk '$5 <= 3' \
    | sort -k5,5n -k4,4nr
} | column -t

echo ""
echo "=========================================="
echo " 📊 STATISTIQUES MOYENNES (Tri: Pos ↗, Reward ↘)"
echo "=========================================="

{
    echo "EXP STEP AVG_REWARD AVG_POSITION"
    grep -r '"reward":' "$LOG_DIR" \
    | sed -E 's|'"$LOG_DIR"'/?([^/]+)/([^/]+)/.*"reward": ([0-9.-]+), "position": ([0-9]+).*|\1 \2 \3 \4|' \
    | awk '{
        key = $1 " " $2;
        sum_reward[key] += $3;
        sum_pos[key] += $4;
        count[key]++;
    }
    END {
        # Note : On ne print plus le header ici pour ne pas le trier
        for (k in count) {
            printf "%s %.2f %.2f\n", k, sum_reward[k]/count[k], sum_pos[k]/count[k]
        }
    }' \
    | sort -k4,4n -k3,3nr
} | column -t