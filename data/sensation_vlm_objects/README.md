# BVIP Priority Captioning (Mandatory Sidewalk/Road + Near Objects) — SFT (+ optional DPO)

This project fine-tunes small Vision-Language Models (≈1B–3B) so that captions are **safety-precise for Blind/Visually Impaired Pedestrians (BVIP)**:

1. **Sidewalk position is mandatory** (veering/drift risk)
2. **Road position is mandatory** (left/right or *front/crossing*)
3. **Close objects are mandatory** (near obstacles/persons/vehicles must be mentioned first)

We convert your two CSVs into:

* **SFT JSONL** for supervised fine-tuning
* **DPO JSONL (optional)** for preference tuning (chosen vs rejected captions)
* **Evaluation** script that measures mandatory-field presence/accuracy + near-object mention recall

---

## 1) Input CSVs

### A) Captions CSV (`complete_captions_sensation.csv`)

Required columns:

* `image_path` — path to image file
* `target` — original caption (may omit close objects)
* `split` — `train`, `val`, `test` (your existing split is respected)

### B) Annotations CSV (`annotations_with_obj_pos.csv`)

Required columns:

* `image` — filename (used to match by basename)
* `sidewalk_pos` — `left|center|right` (or empty → treated as `unknown`)
* `road_pos` — `left|center|right|front` (or empty → `unknown`)
* close object position columns: `m_person, m_car, m_bicycle, m_obstacle, m_traffic_sign, traffic_light`
* far object position columns: same names with `_outside`

---

## 2) Mandatory output schema (non-optional)

Every training target **starts with these three fields**:

```
Sidewalk: <left|center|right|unknown>.
Road: <left|center|right|front|unknown>.
RoadVsSidewalk: <left_of|right_of|crossing_front|unknown>.
```

Then:

* `Nearby:` close objects (always first)
* `Far:` optional
* `Context:` original caption (optional, but usually helpful)

Example `response`:

```
Sidewalk: right. Road: left. RoadVsSidewalk: left_of.
Nearby: a person close on the center; an obstacle close on the right.
Far: a traffic light farther away on the left.
Context: A sidewalk is shown with a pedestrian ahead.
```

---

## 3) Generate the dataset exports

Script: `create_bvip_dataset.py`

```bash
python create_bvip_dataset.py \
  --captions_csv complete_captions_sensation.csv \
  --annotations_csv annotations_with_obj_pos.csv \
  --out_dir bvpi_out
```

Outputs:

* `bvpi_out/bvpi_sft_train.jsonl`
* `bvpi_out/bvpi_sft_val.jsonl`
* `bvpi_out/bvpi_sft_test.jsonl`
* `bvpi_out/bvpi_dpo_train.jsonl` (optional pairs, derived automatically)
* `bvpi_out/bvpi_dpo_val.jsonl`
* `bvpi_out/merged_captions_with_positions.csv` (debug)

### SFT JSONL format

Each line contains both the training text and explicit safety fields:

```json
{
  "image": "path/to/img.jpg",
  "prompt": "...",
  "response": "Sidewalk: ... Road: ... RoadVsSidewalk: ... Nearby: ...",
  "sidewalk_pos": "right",
  "road_pos": "left",
  "road_vs_sidewalk": "left_of"
}
```

### DPO JSONL format (optional)

```json
{
  "image": "path/to/img.jpg",
  "prompt": "...",
  "chosen":   "priority caption (includes mandatory fields + near objects)",
  "rejected": "original target caption",
  "sidewalk_pos": "right",
  "road_pos": "left",
  "road_vs_sidewalk": "left_of"
}
```

---

## 4) Evaluate mandatory fields + near-object recall

Script: `near_object_and_position_eval.py`

Your prediction JSONL must contain:

* `image` (same basename as dataset)
* `text` **or** `response` (the generated caption)

Example line:

```json
{"image":".../xxx.jpg","text":"Sidewalk: center. Road: front. RoadVsSidewalk: crossing_front. Nearby: ..."}
```

Run:

```bash
python near_object_and_position_eval.py \
  --pred_jsonl preds.jsonl \
  --captions_csv complete_captions_sensation.csv \
  --annotations_csv annotations_with_obj_pos.csv \
  --split test
```

Metrics reported:

* Mandatory field presence rate: Sidewalk/Road/RoadVsSidewalk
* Field accuracy vs GT
* Near-object mention recall/precision (keyword proxy)

---

## 5) Best way to fine-tune: Hugging Face or original repo?

### Rule of thumb (practical)

* **Use Hugging Face (Transformers + TRL/PEFT)** if the model has **solid Transformers support** for training/inference and you want to **compare many models fast** with one pipeline.
* **Use the original repo** if the model needs **custom image tiling / dynamic resolution / special collators** or if HF fine-tuning is poorly documented or incomplete.

### Why Hugging Face is usually best for your project

You want to train/evaluate **many 1B–3B models** with the **same dataset** and the same safety-critical schema. HF gives you:

* one dataset format (your JSONL)
* one training loop style (SFT + optional preference tuning)
* easy LoRA/QLoRA switching and reproducible runs

Hugging Face even has an explicit TRL recipe for fine-tuning Qwen2-VL in their ecosystem. ([Hugging Face][1])

### When original repos win

* **InternVL**: their docs define a specific **conversation JSONL + meta file** structure and include dedicated finetune scripts; this is often the “smoothest path” for InternVL models. ([internvl.readthedocs.io][2])
* **OpenFlamingo**: training commonly expects **WebDataset shards** (tar-based) rather than simple JSONL, so using the official training repo (and converting to WebDataset) is typical. ([GitHub][3])
* **MobileVLM-V2**: the official repo explicitly provides training code/data instructions; for full fidelity training, the original repo is usually best. ([GitHub][4])

---

## 6) Model-by-model recommendation (what I’d do)

### Best HF-first candidates (easiest unified pipeline)

Use **Transformers + PEFT/TRL** with `bvpi_sft_*.jsonl`:

* **Qwen2-VL (2B–3B)**: HF ecosystem has an end-to-end TRL fine-tuning recipe pattern you can adapt. ([Hugging Face][1])
* **DeepSeek-VL-1.3B**: has Transformers documentation for the model family. ([Hugging Face][5])
* **SmolVLM2-2.2B**: model card states you can load/infer/**fine-tune with Transformers**. ([Hugging Face][6])
* **PaliGemma 2 (3B)**: Transformers docs explain **pt checkpoints are intended for fine-tuning** (mix are “ready out of box”). ([Hugging Face][7])
* **Florence-2-base**: HF blog shows how to fine-tune Florence-2 on custom datasets (adapt prompt/target). ([Hugging Face][8])

**If your goal is “same training recipe across models” → start here.**

### Best original-repo-first candidates (custom training stack)

* **InternVL2_1B / InternVL2.x**: follow InternVL’s conversation JSONL format (with `<image>\n...`) and meta config file. ([internvl.readthedocs.io][2])
* **OpenFlamingo-3B**: plan to convert to WebDataset shards or use a Flamingo fine-tune stack that supports JSONL-to-webdataset conversion. ([GitHub][9])
* **MobileVLM-V2**: prefer official training code if you want the intended training scheme. ([GitHub][4])
* **Janus-1.3B**: Janus fine-tuning is often done via alignment/fine-tune frameworks referenced by the maintainers (e.g., Align-Anything). ([Hugging Face][10])

### “Depends / check your tooling comfort”

* **Moondream2**: easy to use via HF for inference (and there are community fine-tune guides), but fine-tuning workflows are less standardized than the big HF-first models above. ([Hugging Face][11])

---

## 7) Recommended training strategy for BVIP safety

### Stage 1 — SFT (mandatory schema)

Train with SFT JSONL so the model **always emits**:

* Sidewalk / Road / RoadVsSidewalk
* Near objects first

### Stage 2 — Optional preference tuning (DPO/MPO)

Use DPO pairs where:

* `chosen` = priority caption (must include safety fields + close objects)
* `rejected` = original caption (often missing safety items)

Hugging Face also documents preference optimization for VLMs in their TRL ecosystem (e.g., MPO). ([Hugging Face][12])

---

## 8) Links (URLs only in this block)

```text
HF TRL VLM fine-tuning (Qwen2-VL example):
https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl

InternVL chat data format (conversation JSONL, <image> token, meta file):
https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html
InternVL fine-tuning page:
https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html

DeepSeek-VL Transformers docs:
https://huggingface.co/docs/transformers/en/model_doc/deepseek_vl

SmolVLM2 model card:
https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct

PaliGemma Transformers docs:
https://huggingface.co/docs/transformers/main/model_doc/paligemma

Florence-2 fine-tuning blog:
https://huggingface.co/blog/finetune-florence2

OpenFlamingo repo:
https://github.com/mlfoundations/open_flamingo
WebDataset format:
https://github.com/webdataset/webdataset

MobileVLM repo:
https://github.com/Meituan-AutoML/MobileVLM

Janus repo:
https://github.com/deepseek-ai/Janus
Align-Anything:
https://github.com/PKU-Alignment/align-anything
```

---

### If you want one “best overall” approach

For your use case (many models, same dataset, fast iteration), I’d do:

**HF-first baseline:** Qwen2-VL (2B) or SmolVLM2-2.2B → confirm your schema works + evaluate reliably
Then add: DeepSeek-VL-1.3B, PaliGemma2-3B, Florence-2-base
Use original repos only for: InternVL, OpenFlamingo, MobileVLM (and Janus if Align-Anything route)

[1]: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl?utm_source=chatgpt.com "Fine-Tuning a Vision Language Model (Qwen2-VL-7B) ..."
[2]: https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html "Chat Data Format — InternVL"
[3]: https://github.com/mlfoundations/open_flamingo?utm_source=chatgpt.com "mlfoundations/open_flamingo: An open-source framework ..."
[4]: https://github.com/Meituan-AutoML/MobileVLM?utm_source=chatgpt.com "MobileVLM: Vision Language Model for Mobile Devices"
[5]: https://huggingface.co/docs/transformers/en/model_doc/deepseek_vl?utm_source=chatgpt.com "DeepseekVL"
[6]: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct?utm_source=chatgpt.com "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
[7]: https://huggingface.co/docs/transformers/en/model_doc/paligemma?utm_source=chatgpt.com "PaliGemma"
[8]: https://huggingface.co/blog/finetune-florence2?utm_source=chatgpt.com "Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision ..."
[9]: https://github.com/webdataset/webdataset?utm_source=chatgpt.com "webdataset/webdataset"
[10]: https://huggingface.co/deepseek-ai/Janus-1.3B/discussions/8?utm_source=chatgpt.com "deepseek-ai/Janus-1.3B · Training / Fine-tuning Code, ..."
[11]: https://huggingface.co/vikhyatk/moondream2?utm_source=chatgpt.com "vikhyatk/moondream2"
[12]: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_mpo?utm_source=chatgpt.com "Fine-Tuning a Vision Language Model with TRL using MPO"
