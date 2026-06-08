# CoT Diversity@k Results — Step 400 (4B, all arms)

Expected distinct correct CoT clusters in a random k-subset of 64 rollouts,
averaged over all prompts. Clusters assigned by LLM judge (Qwen3-4B-Instruct)
based on reasoning strategy (macro + micro approach), not final answer.

>  unavailable: eval JSON truncated (parse error).

## math500

| arm | n_correct_prompts | div@1 | div@2 | div@4 | div@8 | div@16 | div@32 | div@64 |
|-----|:-----------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| base | 464/500 | 0.3153 | 0.4909 | 0.6649 | 0.7960 | 0.8951 | 0.9895 | 1.0938 |
| grpo | 408/500 | 0.2784 | 0.4054 | 0.5268 | 0.6354 | 0.7355 | 0.8333 | 0.9398 |
| minority | 402/500 | 0.2380 | 0.3523 | 0.4638 | 0.5613 | 0.6463 | 0.7244 | 0.7959 |
| polyepo | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## beyondaime

| arm | n_correct_prompts | div@1 | div@2 | div@4 | div@8 | div@16 | div@32 | div@64 |
|-----|:-----------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| base | 29/100 | 0.0102 | 0.0185 | 0.0317 | 0.0504 | 0.0763 | 0.1135 | 0.1698 |
| grpo | 12/100 | 0.0048 | 0.0093 | 0.0170 | 0.0292 | 0.0455 | 0.0635 | 0.0800 |
| minority | 9/100 | 0.0033 | 0.0059 | 0.0096 | 0.0140 | 0.0188 | 0.0251 | 0.0300 |
| polyepo | 13/100 | 0.0025 | 0.0048 | 0.0088 | 0.0151 | 0.0228 | 0.0285 | 0.0300 |

