#!/bin/bash

cd /bd_byt4090i0/users/hfang/FlowPolicy

alg_name=${1}
seed=${2}

# 配置目录和数据目录路径
CONFIG_DIR="./FlowPolicy/mp1/config/task"
DATA_DIR="./FlowPolicy/data/outputs"
TRAIN_SCRIPT="scripts_mp1/train_policy.sh"

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误：配置目录 $CONFIG_DIR 不存在"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误：训练脚本 $TRAIN_SCRIPT 不存在"
    exit 1
fi

# 创建数据目录（如果不存在）
mkdir -p "$DATA_DIR"

# 获取所有环境名称（从.yaml配置文件中提取）
echo "正在获取所有可用环境..."
ENVIRONMENTS=$(find "$CONFIG_DIR" -name "*.yaml" | while read -r file; do
    basename "$file" .yaml
done)

# 统计总环境数量
TOTAL_ENV_COUNT=$(echo "$ENVIRONMENTS" | wc -l | tr -d ' ')
echo "发现 $TOTAL_ENV_COUNT 个可用环境"

# 记录已运行和未运行的环境
RUNNING_ENV_COUNT=0
UNRUN_ENV_COUNT=0
UNRUN_ENVS=()

# 检查每个环境是否已运行
echo "正在检查已运行的环境..."
for env in $ENVIRONMENTS; do
    # 检查数据目录中是否存在包含当前环境名的文件夹
    if ls "$DATA_DIR" | grep -q "$env"; then
        echo "环境 $env 已运行，将跳过"
        ((RUNNING_ENV_COUNT++))
    else
        echo "环境 $env 未运行，将加入执行队列"
        ((UNRUN_ENV_COUNT++))
        UNRUN_ENVS+=("$env")
    fi
done

# 显示统计信息
echo "-------------------------"
echo "已运行环境: $RUNNING_ENV_COUNT"
echo "未运行环境: $UNRUN_ENV_COUNT"
echo "-------------------------"

# 如果有未运行的环境，执行它们
if [ $UNRUN_ENV_COUNT -gt 0 ]; then
    echo "开始运行未执行的环境..."
    for env in "${UNRUN_ENVS[@]}"; do
        echo "-------------------------"
        echo "开始运行环境: $env"
        echo "执行命令: bash $TRAIN_SCRIPT $alg_name $env $seed 0 0"

        # 执行训练命令
        bash "$TRAIN_SCRIPT" "$alg_name" "$env" "$seed" 0 0

        # 检查命令执行结果
        if [ $? -eq 0 ]; then
            echo "环境 $env 运行成功"
        else
            echo "环境 $env 运行失败"
        fi
    done

    echo "-------------------------"
    echo "所有未运行环境处理完毕"
else
    echo "没有未运行的环境，无需执行"
fi



                                
