import argparse, json, sys
from pathlib import Path
from typing import List
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "document_ocr": "Extract the key text content from this document page. Keep it faithful and avoid hallucinations.\n<image>\nText:",
    "layout_reasoning": "Analyze the layout (titles, sections, lists, tables, figures) and summarize the main points.\n<image>\nSummary:",
    "table_qa": "You are a careful table understanding assistant. Read the table.\n<image>\nQuestion: {q}\nAnswer:",
    "chart_qa": "You are a precise chart analyst.\n<image>\nQuestion: {q}\nAnswer:"
}

DEFAULT_Q = {
    "table_qa": "What is the total of the last column and which row has the max value?",
    "chart_qa": "What is the main trend and the final value?"
}

def pick_device(user: str|None):
    if user: return user
    return "cuda" if torch.cuda.is_available() else "cpu"

def pick_dtype(user: str|None):
    s = (user or "bfloat16").lower()
    if s in ("bf16","bfloat16"): return torch.bfloat16
    if s in ("fp16","float16"): return torch.float16
    if s in ("fp32","float32"): return torch.float32
    return torch.bfloat16

def build_prompt(preset: str, question: str|None):
    if preset not in PROMPTS:
        raise ValueError(f"Unknown preset: {preset}")
    tmpl = PROMPTS[preset]
    if "{q}" in tmpl:
        if not question or not question.strip():
            question = DEFAULT_Q.get(preset, "Answer the question about the image.")
        return tmpl.format(q=question)
    return tmpl

def expand_paths(patterns: List[str]) -> List[Path]:
    out = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            out.extend([p for p in Path().glob(pat) if p.is_file()])
        else:
            p = Path(pat)
            if p.is_file():
                out.append(p)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--preset", default="document_ocr", choices=list(PROMPTS.keys()))
    ap.add_argument("--question", default=None)
    ap.add_argument("--out", default="runs/internvl_doc")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out); (out_dir/"answers").mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)

    print(f"Loading {args.model_id} on {device} dtype={dtype}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    paths = expand_paths(args.images)
    if not paths:
        print("No images found."); sys.exit(1)

    prompt = build_prompt(args.preset, args.question)

    results = []
    with (out_dir/"preds.jsonl").open("w", encoding="utf-8") as f:
        for p in paths:
            img = Image.open(p).convert("RGB")
            inputs = processor(text=[prompt], images=[img], return_tensors="pt")
            if device.startswith("cuda"):
                inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
            if hasattr(processor, "batch_decode"):
                ans = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            else:
                tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
                ans = tok.decode(out_ids[0], skip_special_tokens=True)

            rec = {"image": str(p), "preset": args.preset, "prompt": prompt, "answer": ans}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (out_dir/"answers"/(p.stem + ".md")).write_text(f"# Answer for {p.name}\n\n{ans}\n", encoding="utf-8")
            results.append(rec)
            print(f"{p.name}: {ans[:160]}{'...' if len(ans)>160 else ''}")

    print(f"Wrote {out_dir/'preds.jsonl'}")

if __name__ == "__main__":
    main()
