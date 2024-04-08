**ABSTRACT:** Large Language Models (LLMs) have proven to be remarkably accurate and effective for several tasks such as summarisation, language translation, question answering and many others. However, to expand their capabilities and performance, these models have progressively increased in size. This growth has prompted research in two key areas: model compression and fine-tuning. Through compression techniques like pruning, redundant parameters and connections are trimmed, decreasing both memory usage and inference time. Fine-tuning then tailors the model's parameters to excel in designated domains or tasks, leveraging pre-trained natural language knowledge. This synergy optimises efficiency minimising impact on performance, addressing challenges of computational demands and task-specific proficiency. We seek to find the optimal ordering of this synergy, reporting our results on well-known LLM benchmarks. This study discusses a methodology for model compression and performance regeneration via Wanda pruning and LoRA fine-tuning. We investigate and quantify the impact on performance based on the ordering of pruning and fine-tuning for a compressed model on task-specific metrics, showing that _'Order Matters'_.
