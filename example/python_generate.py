"""普通解码和 dSpark 共用 generate()，返回完整结果及实际参数。"""
from spark_mlx_llm import GenerationConfig, generate, load

MODEL_PATH = "/path/to/Spark-X2.6-4B-INT4"
DRAFT_PATH = None  # 设置为匹配草稿的路径即可启用 dSpark。
MESSAGES = [{"role": "user", "content": "请用一句话介绍你自己。"}]


def main():
    model, tokenizer = load(MODEL_PATH, draft_model=DRAFT_PATH)
    config = GenerationConfig()
    result = generate(model, tokenizer, MESSAGES, config=config)
    print(result.text)
    print(result.metrics())


if __name__ == "__main__":
    main()
