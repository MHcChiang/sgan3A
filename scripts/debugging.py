import logging
import sys
import numpy as np
from argparse import Namespace # 用來模擬 args
import os

# 讓 Python 找得到同層級的模組
sys.path.append(".")
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.getcwd())
# Import Data Generator (The Iterator based one)
from model.data.dataloader import data_generator
from model.sgan3A import AgentFormerGenerator, AgentFormerDiscriminator
from model.losses import gan_g_loss, gan_d_loss, l2_loss
from utils.logger import Logger

try:
    # 從你的主程式引入 data_generator 和 SmartBatcher
    # 如果你的檔名不是 train_sgan3A.py，請修改這裡
    from scripts.train_sgan3A import data_generator, SmartBatcher
except ImportError as e:
    print(f"Import Error: {e}")
    print("請確認 train_sgan3A.py 位於同一目錄，且裡面包含 SmartBatcher Class")
    sys.exit(1)

# 設定 Logging
logger = Logger("./log.txt")

def main():
    # 1. 模擬參數 (Args)
    # 請確認這裡的 data_root 對應到你實際的資料夾路徑
    args = Namespace(
        dataset='eth',
        data_root_ethucy='/Users/dannychiang/Desktop/TAMU/CSCE689_MRS/project/sgan3A/datasets/eth_ucy', 
        dataset_name='eth', # 某些版本的 loader 可能看這個
        past_frames=8,
        future_frames=12,
        min_past_frames=8,
        min_future_frames=12,
        frame_skip=1,
        phase='training',
        batch_size=8,      # 設定目標 Batch Size
        augment=1,         # 開啟 Augmentation 測試加倍後的限制
        max_agents=50,      # 設定我們測試的硬上限
        traj_scale=1,
    )
    
    args.data_root_nuscenes_pred = 'datasets/nuscenes_pred' 

    print("-------------------------------------------------")
    print(f"Testing SmartBatcher logic")
    print(f"Target Batch Size: {args.batch_size}")
    print(f"Augmentation: {bool(args.augment)}")
    print(f"Max Agent Limit: {args.max_agents}")
    print("-------------------------------------------------")

    args.get = lambda key, default=None: getattr(args, key, default)

    # 2. 初始化 Generator
    train_gen = data_generator(args, logger, split='train', phase='training')

    # 3. 初始化 SmartBatcher
    # 我們設定 limit 為 args.max_agents (例如 50)
    batcher = SmartBatcher(
        train_gen, 
        batch_size=args.batch_size, 
        augment=bool(args.augment), 
        max_agents_limit=args.max_agents
    )

    # 4. 執行 Epoch 迴圈測試
    num_epochs = 10
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Starting Epoch {epoch} ===")
        
        # 呼叫 reset (這應該會觸發內部的 shuffle)
        batcher.reset()
        
        batch_count = 0
        total_agents_processed = 0
        violation_count = 0
        max_agents_in_batch = 0
        
        while batcher.has_data():
            # 獲取 Batch (這是 raw list，尚未轉 Tensor)
            batch = batcher.next_batch()
            
            if batch is None:
                break
                
            batch_count += 1
            
            # 檢查 Agent 數量
            # collate_scenes 回傳的字典中，'pre_motion_3D' 是一個包含所有 agent 的 list
            # 如果 augment=True，這裡的數量應該已經包含原始 + 旋轉後的 agent
            current_agents = len(batch['pre_motion_3D'])
            total_agents_processed += current_agents
            
            # 檢查是否超過限制
            status = "PASS"
            if current_agents > args.max_agents:
                status = "WARNING (Over Limit)"
                violation_count += 1
                max_agents_in_batch = max(max_agents_in_batch, current_agents)
                #  print(f"Batch {batch_count:03d} | Agents: {current_agents:03d} | Limit: {args.max_agents} | {status}")
                

        print(f"--- Epoch {epoch} Summary ---")
        print(f"Total Batches: {batch_count}")
        print(f"Total Agents: {total_agents_processed}")
        print(f"Max Agents in Batch: {max_agents_in_batch}")
        print(f"Found {violation_count} batches exceeding limit")

if __name__ == "__main__":
    main()