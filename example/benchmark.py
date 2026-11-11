"""通过公共 benchmark() 记录每次结果、实际参数及合并吞吐。"""
import json
from spark_mlx_llm import GenerationConfig, load
from spark_mlx_llm.benchmark import benchmark

MODEL_PATH = "/path/to/Spark-X2.6-4B-INT4"
DRAFT_PATH = None


def main():
    model, tokenizer = load(MODEL_PATH, draft_model=DRAFT_PATH)
    config = GenerationConfig()
    report = benchmark(model, tokenizer, "请介绍推测解码。", config=config, trials=5)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
