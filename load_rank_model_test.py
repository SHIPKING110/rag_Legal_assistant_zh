# test_fixed_qwen_reranker.pyreturn embed
import os
from pathlib import Path

# 设置环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def test_fixed_qwen_reranker():
    model_path = "./model/rank/Qwen/Qwen3-Reranker-0.6B"
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_path}")
        return False
    
    try:
        from sentence_transformers import CrossEncoder
        
        print("🔄 正在加载修复后的 Qwen3-Reranker...")
        # 加载模型
        model = CrossEncoder(model_path, trust_remote_code=True, local_files_only=True)
        
        # 修复：设置填充令牌
        if hasattr(model, 'tokenizer') and model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        
        print("✅ 模型加载成功")
        
        # 测试预测功能 - 逐个处理
        print("🔄 正在测试模型预测（逐个处理）...")
        test_pairs = [
            ["劳动法规定的工作时间是多少？", "根据劳动法，每日工作时间不超过8小时，平均每周工作时间不超过44小时。"],
            ["劳动法规定的工作时间是多少？", "苹果是一种水果，富含维生素和纤维。"]
        ]
        
        scores = []
        for pair in test_pairs:
            score = model.predict([pair])  # 注意：传入列表，但只包含一个元素
            scores.append(float(score[0]))
        
        print(f"✅ 模型预测测试成功")
        print(f"相关文档得分: {scores[0]:.4f}")
        print(f"不相关文档得分: {scores[1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载或测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_fixed_qwen_reranker()