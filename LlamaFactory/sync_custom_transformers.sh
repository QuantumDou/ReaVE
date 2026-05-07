#!/bin/bash

echo "🔄 批量同步 src/transformers/ 下自定义文件到系统包..."

SRC_DIR="../src/transformers"
TARGET_DIR="/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/envs/inter2.5/lib/python3.11/site-packages/transformers"
# 检查源目录是否存在
if [ ! -d "$SRC_DIR" ]; then
    echo "❌ 源目录不存在: $SRC_DIR"
    exit 1
fi

cp "$SRC_DIR/utils.py" "$TARGET_DIR/generation/utils.py"
cp "$SRC_DIR/modeling_qwen2_5_vl.py" "$TARGET_DIR/models/qwen2_5_vl/modeling_qwen2_5_vl.py"

# 验证
if [ $? -eq 0 ]; then
    echo "✅ 全部同步完成！"
    echo ""
    echo "📊 已同步文件："
    ls -lh "$TARGET_DIR/generation/utils.py" 2>/dev/null | awk '{print "  utils.py →", $5}'
    ls -lh "$TARGET_DIR/models/qwen2_5_vl/modeling_qwen2_5_vl.py" 2>/dev/null | awk '{print "  modeling_qwen2_5_vl.py →", $5}'
    echo ""
    
    # 对比文件内容
    echo "🔍 验证文件内容是否完全相同..."
    echo ""
    
    if cmp -s "$SRC_DIR/utils.py" "$TARGET_DIR/generation/utils.py"; then
        echo "  ✅ utils.py 内容完全相同"
    else
        echo "  ❌ utils.py 内容不同！"
    fi
    
    if cmp -s "$SRC_DIR/modeling_qwen2_5_vl.py" "$TARGET_DIR/models/qwen2_5_vl/modeling_qwen2_5_vl.py"; then
        echo "  ✅ modeling_qwen2_5_vl.py 内容完全相同"
    else
        echo "  ❌ modeling_qwen2_5_vl.py 内容不同！"
    fi
    echo ""
else
    echo "❌ 同步失败！"
    exit 1
fi