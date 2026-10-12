"""
Prompting vs. LoRA Fine-Tuning - Structured Evaluation Series
Covers: prompt library + versioning, prompt templates, A/B testing of tuning strategies,
        hallucination rate, coverage, accuracy, latency, LoRA / PEFT fine-tuning, guardrail (output validation)

Task: classify a clinical note into 1 of 6 specialties with a small LLM (Qwen2.5-0.5B-Instruct).
      Same MTSamples data as your clinical notes project.

Run in Google Colab with a T4 GPU (Runtime -> Change runtime type -> T4 GPU), ~25-35 minutes:
  Cell 1:  !pip install -q kagglehub peft accelerate
  Cell 2:  paste this file and run it.
Paste the RESULTS block back to Claude.
"""
import os, json, time, random, shutil
from datetime import date
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
assert torch.cuda.is_available(), "Turn on the T4 GPU first (Runtime -> Change runtime type)."
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_TEST, N_TRAIN, NOTE_CHARS = 150, 900, 1200

# ---------- 1. Data (same 6 specialties as the DistilBERT project) ----------
import kagglehub
if not os.path.exists("mtsamples.csv"):
    p = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
    shutil.copy(os.path.join(p, "mtsamples.csv"), "mtsamples.csv")
df = pd.read_csv("mtsamples.csv").dropna(subset=["transcription", "medical_specialty"])
df["medical_specialty"] = df["medical_specialty"].str.strip()
drop = {"Surgery", "Consult - History and Phy.", "SOAP / Chart / Progress Notes", "Discharge Summary",
        "Office Notes", "Letters", "General Medicine", "Emergency Room Reports"}
df = df[~df["medical_specialty"].isin(drop)]
LABELS = sorted(df["medical_specialty"].value_counts().head(6).index)
df = df[df["medical_specialty"].isin(LABELS)].reset_index(drop=True)
df["text"] = df["transcription"].str.replace(r"\s+", " ", regex=True).str[:NOTE_CHARS]
train, test = train_test_split(df, test_size=0.2, stratify=df["medical_specialty"], random_state=SEED)
test = test.sample(min(N_TEST, len(test)), random_state=SEED)
train = train.sample(min(N_TRAIN, len(train)), random_state=SEED)

# ---------- 2. Prompt library with versioning ----------
shots = train.groupby("medical_specialty").head(1)
few_shot_block = "\n\n".join(f"Note: {r.text[:300]}\nSpecialty: {r.medical_specialty}" for r in shots.itertuples())
PROMPT_LIBRARY = {
    "v1.0_zero_shot": {
        "created": str(date.today()), "change": "baseline instruction",
        "template": "Classify the clinical note into exactly one specialty from this list: {labels}.\n"
                    "Answer with the specialty name only.\n\nNote: {note}\nSpecialty:"},
    "v1.1_zero_shot_strict": {
        "created": str(date.today()), "change": "added strict output rule to reduce invalid answers",
        "template": "You are a medical records classifier. Choose exactly one specialty from: {labels}.\n"
                    "Rules: reply with one specialty name copied exactly from the list, no other words.\n\n"
                    "Note: {note}\nSpecialty:"},
    "v2.0_few_shot": {
        "created": str(date.today()), "change": "added one labeled example per specialty",
        "template": "Classify the clinical note into exactly one specialty from: {labels}.\n"
                    "Reply with the specialty name only.\n\nExamples:\n" + few_shot_block.replace("{", "(").replace("}", ")") +
                    "\n\nNote: {note}\nSpecialty:"},
}
json.dump(PROMPT_LIBRARY, open("prompt_library.json", "w"), indent=2)

# ---------- 3. Model + evaluation harness ----------
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

def build(template, note):
    msg = [{"role": "user", "content": template.format(labels=", ".join(LABELS), note=note)}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

def parse(output):
    """Guardrail / validation layer: accept only an exact label from the allowed list."""
    out = output.strip().lower()
    for lab in sorted(LABELS, key=len, reverse=True):
        if lab.lower() in out:
            return lab
    return None                                     # invalid / hallucinated label

@torch.no_grad()
def evaluate(m, template, name):
    rows = []
    for r in test.itertuples():
        enc = tok(build(template, r.text), return_tensors="pt").to("cuda")
        t0 = time.time()
        gen = m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        lat = time.time() - t0
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse(text)
        rows.append({"strategy": name, "true": r.medical_specialty, "raw": text, "pred": pred,
                     "correct": pred == r.medical_specialty, "valid": pred is not None, "latency_s": lat})
    return pd.DataFrame(rows)

results = [evaluate(model, v["template"], k) for k, v in PROMPT_LIBRARY.items()]

# ---------- 4. LoRA fine-tuning (PEFT) ----------
from peft import LoraConfig, get_peft_model
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda()
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
tmpl = PROMPT_LIBRARY["v1.0_zero_shot"]["template"]
tok.padding_side = "right"
examples = []
for r in train.itertuples():
    p_ids = tok(build(tmpl, r.text), add_special_tokens=False)["input_ids"]
    a_ids = tok(r.medical_specialty + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[-512:]
    labels = ([-100] * len(p_ids) + a_ids)[-512:]           # loss only on the answer tokens
    examples.append((ids, labels))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train(); BS = 4
for epoch in range(1):
    random.shuffle(examples)
    for i in range(0, len(examples), BS):
        batch = examples[i:i + BS]; L = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch]).cuda()
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch]).cuda()
        att = (torch.arange(L)[None, :] < torch.tensor([len(x) for x, _ in batch])[:, None]).long().cuda()
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        opt.zero_grad(); loss.backward(); opt.step()
model.eval(); tok.padding_side = "left"
results.append(evaluate(model, tmpl, "lora_finetuned_v1.0"))

# ---------- 5. Scorecard + A/B test (paired bootstrap) ----------
allr = pd.concat(results)
allr.to_csv("eval_results.csv", index=False)
score = allr.groupby("strategy").agg(accuracy=("correct", "mean"), coverage=("valid", "mean"),
                                     avg_latency_s=("latency_s", "mean")).reset_index()
score["hallucination_rate"] = 1 - score["coverage"]
score = score.sort_values("accuracy", ascending=False)

best_prompt = score[score["strategy"] != "lora_finetuned_v1.0"].iloc[0]["strategy"]
a = allr[allr.strategy == "lora_finetuned_v1.0"]["correct"].values.astype(float)
b = allr[allr.strategy == best_prompt]["correct"].values.astype(float)
rng = np.random.default_rng(SEED)
diffs = [ (a[idx] - b[idx]).mean() for idx in (rng.integers(0, len(a), len(a)) for _ in range(2000)) ]
lo, hi = np.percentile(diffs, [2.5, 97.5])

print("\n================ RESULTS ================")
print(f"Model: {MODEL} | Test notes: {len(test)} | LoRA train notes: {len(train)} | Labels: {len(LABELS)}")
print(f"Prompt versions in library: {len(PROMPT_LIBRARY)} (saved to prompt_library.json)")
print(f"LoRA trainable params: {trainable:,} of {total:,} ({100*trainable/total:.2f}%)")
print(score.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nA/B: LoRA vs best prompt ({best_prompt}): accuracy diff = {a.mean()-b.mean():+.3f}, "
      f"95% CI [{lo:+.3f}, {hi:+.3f}]")
print("Saved eval_results.csv (every output, for error review)")
