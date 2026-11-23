"""同一个 generate()，通过 stream=True 消费增量。"""
from contextlib import closing
from spark_mlx_llm import GenerationConfig, generate, load

MODEL_PATH = "/path/to/Spark-X2.6-4B-INT4"
DRAFT_PATH = None
MESSAGES = [{"role": "user", "content": "请用一句话介绍你自己。"}]


def main():
    model, tokenizer = load(MODEL_PATH, draft_model=DRAFT_PATH)
    config = GenerationConfig()
    with closing(generate(model, tokenizer, MESSAGES, config=config, stream=True)) as chunks:
        for chunk in chunks:
            print(chunk.text, end="", flush=True)
            if chunk.result is not None:
                print("\n", chunk.result.metrics())


if __name__ == "__main__":
    main()
