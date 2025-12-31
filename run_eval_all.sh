#!/bin/bash

cd /bd_byt4090i0/users/hfang/FlowPolicy

alg_name=${1}
seed=${2}

# 配置目录和数据目录路径
CONFIG_DIR="./FlowPolicy/mp1/config/task"
DATA_DIR="./FlowPolicy/data/outputs"
EVAL_SCRIPT="scripts_mp1/eval_policy.sh"  # 评估脚本路径

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误：配置目录 $CONFIG_DIR 不存在"
    exit 1
fi

# 检查评估脚本是否存在
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "错误：评估脚本 $EVAL_SCRIPT 不存在"
    exit 1
fi

# 获取所有环境名称（从.yaml配置文件中提取）
echo "正在获取所有可用环境..."
ENVIRONMENTS=$(find "$CONFIG_DIR" -name "*.yaml" | while read -r file; do
    basename "$file" .yaml
done)

# 统计总环境数量
TOTAL_ENV_COUNT=$(echo "$ENVIRONMENTS" | wc -l | tr -d ' ')
echo "发现 $TOTAL_ENV_COUNT 个可用环境"

# 记录符合评估条件和不符合的环境
EVAL_ENV_COUNT=0
SKIP_ENV_COUNT=0
EVAL_ENVS=()
SKIP_ENVS=()

# 检查每个环境是否符合评估条件
echo "正在检查符合评估条件的环境..."
for env in $ENVIRONMENTS; do
    # 1. 检查数据目录中是否存在同时包含环境名、alg_name和seed的文件夹
    # 构建匹配模式：同时包含三个关键词
    pattern="${env}-${alg_name}-${seed}_seed0"
    if ls -d "$DATA_DIR/$pattern" 2>/dev/null | grep -q .; then
        # 获取匹配的文件夹（取第一个）
        env_dir=$(find "$DATA_DIR" -maxdepth 1 -type d -name "$pattern" | head -n 1)
        # 2. 检查该文件夹下的checkpoint文件夹是否存在latest.ckpt文件
        if [ -f "$env_dir/checkpoints/latest.ckpt" ]; then
            echo "环境 $env 符合评估条件，将加入执行队列"
            ((EVAL_ENV_COUNT++))
            EVAL_ENVS+=("$env")
        else
            echo "环境 $env 数据文件夹存在，但缺少 checkpoints/latest.ckpt，将跳过"
            ((SKIP_ENV_COUNT++))
            SKIP_ENVS+=("$env")
        fi
    else
        echo "环境 $env 不存在同时包含环境名、alg_name和seed的数据文件夹，将跳过"
        ((SKIP_ENV_COUNT++))
        SKIP_ENVS+=("$env")
    fi
done

# 显示统计信息
echo "-------------------------"
echo "符合评估条件的环境: $EVAL_ENV_COUNT"
echo "跳过的环境: $SKIP_ENV_COUNT"
echo "-------------------------"

# 如果有符合条件的环境，执行评估
if [ $EVAL_ENV_COUNT -gt 0 ]; then
    echo "开始评估符合条件的环境..."
    for env in "${EVAL_ENVS[@]}"; do
        echo "-------------------------"
        echo "开始评估环境: $env"
        echo "执行命令: bash $EVAL_SCRIPT $alg_name $env $seed 0 0"

        # 执行评估命令
        bash "$EVAL_SCRIPT" "$alg_name" "$env" "$seed" 0 0

        # 检查命令执行结果
        if [ $? -eq 0 ]; then
            echo "环境 $env 评估成功"
        else
            echo "环境 $env 评估失败"
        fi
    done

    echo "-------------------------"
    echo "所有符合条件的环境评估完毕"
else
    echo "没有符合评估条件的环境，无需执行"
fi
