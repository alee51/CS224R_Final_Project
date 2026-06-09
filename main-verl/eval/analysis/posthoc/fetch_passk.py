"""Fetch pass@k from all eval_4b JSONs and save a compact summary."""
import json
from pathlib import Path
import modal

app = modal.App("cs224r-fetch-passk")
vol = modal.Volume.from_name("main-artifacts", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("ijson")

ARMS = ["base", "grpo", "minority", "polyepo"]
FILES = {
    "aime25":     "{arm}_step400_smallood_aime25.json",
    "aime26":     "{arm}_step400_smallood_aime26.json",
    "beyondaime": "{arm}_step400_smallood_beyondaime.json",
    "hmmt_feb25": "{arm}_step400_smallood_hmmt_feb25.json",
    "hmmt_nov25": "{arm}_step400_smallood_hmmt_nov25.json",
    "math500":    "{arm}_step400_math500_math500.json",
}

@app.function(image=image, volumes={"/vol": vol}, timeout=600, memory=512)
def fetch():
    import ijson
    eval_dir = Path("/vol/probes/eval_4b")
    out = {}
    for ds, pattern in FILES.items():
        for arm in ARMS:
            fname = pattern.format(arm=arm)
            path = eval_dir / fname
            if not path.exists():
                continue
            try:
                pak = {}
                n_prompts = 0
                # Stream parse events; stop as soon as we see per_prompt
                # (pass_at_k and n_prompts come before per_prompt in every eval JSON)
                with path.open("rb") as f:
                    in_pak = False
                    current_key = None
                    for prefix, event, value in ijson.parse(f, use_float=True):
                        if prefix.endswith(".per_prompt") and event == "start_array":
                            break  # everything we need is already parsed
                        if "pass_at_k" in prefix:
                            if event == "map_key":
                                current_key = value
                            elif event in ("number", "string") and current_key:
                                pak[current_key] = float(value)
                        if prefix.endswith(".n_prompts") and event == "number":
                            n_prompts = int(value)
                if pak:
                    out.setdefault(arm, {})[ds] = {
                        "pass_at_k": pak,
                        "n_prompts": n_prompts,
                    }
                    p16 = pak.get("pass@16", "?")
                    print(f"  {arm:10} {ds:15} pass@16={p16:.4f}  n={n_prompts}")
            except Exception as e:
                print(f"  {arm}/{ds}: ERROR {e}")
    Path("/vol/probes/eval_4b/passk_all.json").write_text(json.dumps(out, indent=2))
    print("wrote passk_all.json")
    return out

@app.local_entrypoint()
def main():
    results = fetch.remote()
    for arm, dsets in results.items():
        for ds, v in dsets.items():
            p16 = v["pass_at_k"].get("pass@16", "?")
            print(f"{arm:10} {ds:15} pass@16={p16:.4f}")
