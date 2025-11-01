# Jed’s InternVL Doc Reasoning / OCR Mini-Repo

A ready-to-run **InternVL** inference demo focused on document pages: OCR-ish extraction, layout reasoning, and table/text Q&A.
Works with HF-exported InternVL checkpoints that support `AutoProcessor` and multimodal `generate`.

## Quickstart

```bash
conda env create -f env.yml && conda activate internvl-doc
# Single image:
python infer_internvl.py   --model-id OpenGVLab/InternVL2-8B   --images ./samples/doc_page.png   --preset document_ocr   --out runs/single

# Batch CSV (image,question[,ref]):
python batch_doc_eval.py   --model-id OpenGVLab/InternVL2-8B   --csv examples/jed_internvl_docreason/sample_prompts.csv   --out runs/batch
```
Outputs live under `runs/.../answers/`, with `preds.jsonl` and a tiny `eval.csv` using lexical F1 (token overlap) when a `ref` is provided.

## Presets
- `document_ocr`: extract key text content, avoid hallucinations
- `layout_reasoning`: summarize structure (titles/sections/tables)
- `table_qa`: answer questions about a visible table
- `chart_qa`: answer questions about a chart

> If your model expects a different special token for images, adjust `PROMPTS` in `infer_internvl.py`.
