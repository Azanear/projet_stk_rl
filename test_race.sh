#!/bin/bash

# ===================================================
# CONFIGURATION (Change le checkpoint ici)
# ===================================================
CHECKPOINT=100000
EXP_NAME="exp12"
NB_COURSES=5
# ===================================================

# 1. Création du dossier spécifique (ex: logs/exp16/800000)
OUTPUT_DIR="logs/$EXP_NAME/$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"

echo "=== Lancement des tests pour le Checkpoint : $CHECKPOINT ==="
echo "Dossier de sortie : $OUTPUT_DIR"

# 2. Boucle de courses
for ((i=1; i<=NB_COURSES; i++))
do
   echo " -> Course $i / $NB_COURSES"
   PYTHONPATH=. master-mind rl stk-race --num-karts 4 stk_actor --hide > "$OUTPUT_DIR/race_$i.log" 2>&1
done

echo "Terminé."