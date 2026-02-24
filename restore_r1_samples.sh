#!/bin/bash

# Restore deleted r1-onevision samples from backup files

echo "=== Restoring r1-onevision JSONL files from backups ==="

nodes=("compute-st-kait-gpu-1" "compute-st-kait-gpu-2" "compute-st-kait-gpu-3" "compute-st-kait-gpu-4" \
       "compute-st-kait-gpu-5" "compute-st-kait-gpu-6" "compute-st-kait-gpu-7" "compute-st-kait-gpu-8")

for node in "${nodes[@]}"; do
    echo "Processing node: $node"
    
    # Restore geo170k
    ssh $node "cd /opt/dlami/nvme/data/r1-onevision/geo170k\(qa\)/ && \
               if [ -f train-00001-of-00004.jsonl.backup ]; then \
                   cp train-00001-of-00004.jsonl.backup train-00001-of-00004.jsonl && \
                   echo '  ✅ Restored geo170k(qa)/train-00001-of-00004.jsonl (10000 samples)'; \
               else \
                   echo '  ⚠️  Backup not found for geo170k'; \
               fi"
    
    # Restore geomverse
    ssh $node "cd /opt/dlami/nvme/data/r1-onevision/geomverse/ && \
               if [ -f train-00000-of-00001.jsonl.backup ]; then \
                   cp train-00000-of-00001.jsonl.backup train-00000-of-00001.jsonl && \
                   echo '  ✅ Restored geomverse/train-00000-of-00001.jsonl (4838 samples)'; \
               else \
                   echo '  ⚠️  Backup not found for geomverse'; \
               fi"
    
    echo ""
done

echo "=== Restoration complete ==="
