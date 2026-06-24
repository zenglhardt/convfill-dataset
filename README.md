# ConvFill Dataset: Inference-Time Knowledge Transfer for Responsive and Intelligent Conversational Voice Agents

<p align="center">
  <a href="https://huggingface.co/datasets/zenglhardt/convfill-dataset"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow?style=flat" alt="HuggingFace Dataset" /></a>&nbsp;
  <a href="https://arxiv.org/abs/2511.07397"><img src="https://img.shields.io/badge/arXiv-2511.07397-b31b1b?style=flat&logo=arxiv" alt="arXiv" /></a>&nbsp;
  <a href="https://github.com/vysri/conversational-infill"><img src="https://img.shields.io/badge/GitHub-Code & Models-blue?style=flat&logo=github" alt="GitHub Code and Models" /></a>&nbsp;
  <a href="https://huggingface.co/collections/vysri/convfill-inference-time-knowledge-transfer"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Collection-yellow?style=flat" alt="HuggingFace Collection" /></a>
</p>
This repository contains the training and inference code for ConvFill a dual model collaboration system pairing a small, lightweight `Talker` model with a powerful cloud `Reasoner` model. During inference, the `Talker` has two roles. It consumes raw, inference-time information from the `Reasoner` _when available_ and transforms it into fluent, contingent conversation and it produces fast, conversationally contingent filler phrases to hide `Reasoner` latency _when necessary_.


<p align="center">
  <img src="assets/teaser.png" width="300" alt="Teaser" />
</p>

This repository contains the released ConvFill conversational infill dataset
and the scripts used to generate it. ConvFill targets real-time conversational
voice agents where a small, low-latency Talker model begins responding while a
larger Reasoner model is still reasoning, retrieving information, or using
tools.

**Paper:** *Thinking While Speaking: Inference-Time Knowledge Transfer for
Responsive and Intelligent Conversational Voice Agents*<br>
[https://arxiv.org/abs/2511.07397](https://arxiv.org/abs/2511.07397)

**Authors:** Vidya Srinivas*, Zachary Englhardt*, Vikram Iyer, Shwetak Patel<br>
Paul G. Allen School of Computer Science & Engineering<br>
`{vysri, zacharye}@cs.washington.edu`<br>
*Equal contribution.

**License:** code and supporting release files are MIT; data, topic seeds, and examples are CC BY-SA 4.0.

If you use this dataset, generation code, or find the repository helpful,
please cite the paper. The README gives a practical overview of the data format
and reproduction workflow, but the paper should be read for the formal task
definition, training setup, and evaluation details.

## Dataset Overview

ConvFill targets real-time conversational voice agents where a small,
low-latency frontend model must start speaking while a larger backend model is
still reasoning, retrieving information, or using tools.

The paper frames this as a **Talker/Reasoner** architecture:

- The **Reasoner** is a high-capability backend model. It produces a stream of
  concise knowledge chunks for the current user turn. If it has not produced a
  chunk yet, the stream contains a `<sil>` token.
- The **Talker** is a lightweight frontend model. It emits one response phrase
  for each Reasoner stream item. For `<sil>`, it should produce grounded,
  context-aware filler. For a knowledge chunk, it should turn that information
  into fluent and contextually coherent conversational speech.

In the dataset, each aligned `thoughts[i]` and `response[i]` pair represents
one phrase-level training example for this task.

## Included Data

The released data is in `generated_data/`. Each `.jsonl` file contains one
conversation per line.

| File | Conversations | Turns | Phrase examples | `<sil>` filler examples | Knowledge examples |
| --- | ---: | ---: | ---: | ---: | ---: |
| `conversations_v3_advice_full.jsonl` | 1,000 | 8,308 | 36,710 | 12,342 | 24,368 |
| `conversations_v3_assistant_full.jsonl` | 1,000 | 8,408 | 37,495 | 12,676 | 24,819 |
| `conversations_v3_cs_full.jsonl` | 1,000 | 8,313 | 35,862 | 12,147 | 23,715 |
| `conversations_v3_education_full.jsonl` | 1,000 | 8,421 | 37,153 | 12,481 | 24,672 |
| `conversations_v3_med_full.jsonl` | 1,000 | 8,400 | 37,246 | 12,540 | 24,706 |
| `conversations_v3_planning_full.jsonl` | 1,005 | 8,474 | 37,488 | 12,655 | 24,833 |
| `conversations_v3_scaffold_dstc8_full.jsonl` | 2,438 | 24,184 | 68,617 | 36,711 | 31,906 |
| **Total** | **8,443** | **74,508** | **290,571** | **111,552** | **179,019** |

`generated_data/dataset_stats.csv` contains the same statistics in CSV form.

The dataset is fully in English. The six freeform files cover advice, general
assistant queries, customer service, education, medicine, and planning. The
`scaffold_dstc8` file is generated from Schema-Guided Dialogue (SGD / DSTC8)
conversation scaffolds and includes top-level `scaffold_metadata` with the SGD
service and dialogue id.

## Data Format

Each JSONL record has a top-level `conversation` array:

```json
{
  "conversation": [
    {
      "user": "I've been having a really hard time falling asleep lately. I just lie in bed for what feels like hours before I drift off, and it's wrecking my mornings.",
      "thoughts": [
        "<sil>",
        "How long has this been going on?",
        "Does this happen at the same time every night or does it vary?"
      ],
      "response": [
        "Oh man, that sounds rough.",
        "How long would you say this has been going on?",
        "And is it the same time every night, or does it vary a lot?"
      ]
    }
  ]
}
```

For every turn:

- `user` is the user utterance for that turn.
- `thoughts` is the Reasoner stream. Entries are either knowledge chunks or
  the special `<sil>` token.
- `response` is the Talker output stream.
- `len(thoughts) == len(response)`.
- `response[i]` is aligned to `thoughts[i]`.

When `thoughts[i] == "<sil>"`, `response[i]` is a filler phrase that should
continue the conversation without inventing unavailable backend information.
Otherwise, `response[i]` should conversationally convey the information in
`thoughts[i]`.

## Loading the Dataset

The data can be loaded with the Python standard library:

```python
import json
from pathlib import Path

for path in sorted(Path("generated_data").glob("*.jsonl")):
    if path.name == "dataset_stats.csv":
        continue

    with path.open() as f:
        for line in f:
            record = json.loads(line)
            conversation = record["conversation"]

            for turn_index, turn in enumerate(conversation):
                assert len(turn["thoughts"]) == len(turn["response"])

                for phrase_index, (thought, response) in enumerate(
                    zip(turn["thoughts"], turn["response"])
                ):
                    is_filler = thought == "<sil>"
                    # Use conversation history, turn["user"], prior response
                    # phrases, and thought as inputs for a Talker example.
```

For most training setups, each `(thoughts[i], response[i])` pair is converted
into one phrase-level example, with the previous conversation turns and prior
phrases in the current turn supplied as context. See the paper for the exact
task formulation and formatting used in the ConvFill experiments.

## Repository Layout

- `generated_data/`: released JSONL dataset and dataset statistics.
- `configs/`: generation configs for each dataset subset.
- `examples/`: few-shot examples used by generation prompts.
- `topics/`: topic seeds for the freeform generation subsets.
- `prompt_template.txt` and `prompt_template_scaffold.txt`: generation prompt
  templates.
- `src/pipeline/`: dataset generation and LLM client code.
- `src/validators/`: structural, NLI, BERTScore, and scaffold validation.
- `src/evals/generated_data_stats.py`: regenerates `dataset_stats.csv`.

## Recreating or Extending the Dataset

If you are only planning on using the dataset, you only need the files located
in `generated_data/`. However, we have included the generation pipeline to aid
in reproducing or extending the dataset.

### Environment

A conda environment is recommended:

```bash
conda create -n convfill python=3.11
conda activate convfill
pip install -r requirements.txt
```

The validation models are downloaded through Hugging Face on first use.

### API Keys

API keys are supplied through environment variables, not config files.

All included configs currently use Anthropic:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

If you change a config to use OpenAI, set:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Do not commit API keys to your repository!

### Generate a Small Test Run

`configs/default_config.json` is set up as a small test run:

```bash
python src/pipeline/conv_dataset_gen.py --config configs/default_config.json
```

The output path is derived from `EXPERIMENT_NAME` and `OUTPUT_DIR` in the
config. A fresh run refuses to overwrite an existing output `.jsonl`; change
`EXPERIMENT_NAME` or move the existing file before rerunning.

To continue an interrupted run:

```bash
python src/pipeline/conv_dataset_gen.py --config configs/default_config.json --resume
```

To retry permanently failed topics from a previous run:

```bash
python src/pipeline/conv_dataset_gen.py --config configs/default_config.json --retry
```

### Generate One of the Released Freeform Subsets

Use the corresponding config:

```bash
python src/pipeline/conv_dataset_gen.py --config configs/advice_config.json
python src/pipeline/conv_dataset_gen.py --config configs/assistant_config.json
python src/pipeline/conv_dataset_gen.py --config configs/cs_config.json
python src/pipeline/conv_dataset_gen.py --config configs/education_config.json
python src/pipeline/conv_dataset_gen.py --config configs/med_config.json
```

For planning, use or create a config with `TOPICS_FILENAME` set to
`topics/topics_planning.txt` and a distinct `EXPERIMENT_NAME`.

### Generate the SGD Scaffold Subset

Clone the Schema-Guided Dialogue dataset into the repository root:

```bash
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
```

Then run:

```bash
python src/pipeline/conv_dataset_gen.py --config configs/scaffold_config.json
```

If the SGD dataset lives elsewhere, update `SGD_DATA_PATH` in
`configs/scaffold_config.json`.

### Important Config Fields

- `PROVIDER`, `MODEL_NAME`, `TEMPERATURE`, `MAX_TOKENS`,
  `CALLS_PER_MINUTE`: backend model settings.
- `EXPERIMENT_NAME`, `OUTPUT_DIR`, `CACHE_DIR`: output and cache locations.
- `TOPICS_FILENAME`, `EXAMPLE_PATH`: freeform generation inputs.
- `GENERATION_MODE`: `freeform` or `scaffold`.
- `NUM_REQUESTS`, `NUM_WORKERS`, `MAX_RETRIES`: run size and parallelism.
- `SIL_MIN`, `SIL_MAX`: number of `<sil>` placeholders per turn.
- `SUBSTANCE_MIN`, `SUBSTANCE_MAX`: number of non-silence knowledge chunks per
  turn.
- `INCLUDE_METADATA`: include validation-score metadata in generated records.
  The released dataset has these metadata blocks stripped.

Full-scale generation can be expensive because failed candidates are retried
until they pass validation. The API cost of generating the entire released
dataset was ~$2,400 using Claude Opus 4.6 in early 2026.

### Regenerate Dataset Statistics

```bash
python src/evals/generated_data_stats.py
```

This reads `generated_data/*.jsonl` and writes
`generated_data/dataset_stats.csv`.

## License

This repository uses separate licenses for code and data.

- **Code and supporting release files** in `src/`, `configs/`,
  `prompt_template.txt`, `prompt_template_scaffold.txt`, `requirements.txt`,
  and this README are released under the MIT License. See
  `LICENSE-CODE-MIT.txt`.
- **Data** in `generated_data/`, topic seeds in `topics/`, and few-shot
  examples in `examples/` are released under Creative Commons
  Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). See
  `LICENSE-DATA-CC-BY-SA-4.0.txt`.
- The DSTC8/SGD scaffold subset is derived from the Schema-Guided Dialogue
  dataset, which is also released under CC BY-SA 4.0.

## Citation

If you use the ConvFill dataset, generation pipeline, or find this repository
helpful, please cite the ConvFill paper:

```bibtex
@misc{srinivas2026thinking,
  title = {Thinking While Speaking: Inference-Time Knowledge Transfer for Responsive and Intelligent Conversational Voice Agents},
  author = {Srinivas, Vidya and Englhardt, Zachary and Iyer, Vikram and Patel, Shwetak},
  year = {2026},
  note = {arXiv:2511.07397},
  url = {https://arxiv.org/abs/2511.07397}
}
```

If you use or regenerate the SGD scaffold subset, please also consider citing the Schema-Guided Dialogue dataset:

```bibtex
@inproceedings{rastogi2020schema,
  title = {Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset},
  author = {Rastogi, Abhinav and Zang, Xiaoxue and Sunkara, Srinivas and Gupta, Raghav and Khaitan, Pranav},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {34},
  pages = {8689--8696},
  year = {2020}
}
```
