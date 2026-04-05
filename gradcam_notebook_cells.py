from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.esc50_pooling_tradeoff.data import build_dataset
from experiments.esc50_pooling_tradeoff.gradcam_figure import (
    GradCAM,
    build_model,
    choose_default_target_layer,
    find_checkpoint,
    get_module_by_name,
    infer_num_classes_from_csv,
    read_rows,
)
from experiments.esc50_pooling_tradeoff.train_eval_cv import resolve_device


# Edit these
RUN_DIR = Path(
    r"H:/sound_classification_2/sound_classification_2/outputs/esc50_pooling_tradeoff_protocol/esc50_lwcnn_pool_AdaptSSRP_W4_Ks4812_h128_seed42"
)
FOLD = "fold1"
SPLIT = "test"
SAMPLE_INDEX = 0
TARGET_CLASS = None  # None -> use predicted class
TARGET_LAYER = None  # None -> auto pick final conv/relu block
DEVICE = "cuda"
ALPHA = 0.45


fold_dir = RUN_DIR / FOLD
ckpt_path = find_checkpoint(fold_dir)
state = torch.load(ckpt_path, map_location="cpu")
cfg = state["config"]
data_cfg = dict(cfg["data"])
model_cfg = dict(cfg["model"])
device = torch.device(resolve_device(DEVICE))

split_csv = fold_dir / "normalized_csv" / f"{SPLIT}.csv"
rows = read_rows(split_csv)
data_cfg["root"] = str(Path.cwd())
dataset = build_dataset(data_cfg, model_cfg, str(split_csv), training=False)
x, y = dataset[SAMPLE_INDEX]
x = x.unsqueeze(0).to(device)
y_true = int(y.item())
num_classes = int(state.get("num_classes") or data_cfg.get("num_classes") or infer_num_classes_from_csv(split_csv))
model = build_model(cfg, num_classes=num_classes).to(device)
model.load_state_dict(state["model_state"], strict=True)
model.eval()

target_layer_name = TARGET_LAYER or choose_default_target_layer(model)
target_layer = get_module_by_name(model, target_layer_name)
print("checkpoint:", ckpt_path)
print("target_layer:", target_layer_name)
print("sample_row:", rows[SAMPLE_INDEX])


cam_engine = GradCAM(model, target_layer)
try:
    with torch.enable_grad():
        logits0 = model(x)
        if isinstance(logits0, tuple):
            logits0 = logits0[0]
        pred_idx = int(logits0.argmax(dim=1).item())
        class_idx = pred_idx if TARGET_CLASS is None else int(TARGET_CLASS)
        logits, cam = cam_engine.compute(x, class_idx)
finally:
    cam_engine.remove()

probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
pred_prob = float(probs[pred_idx].item())
mel = x.detach().cpu().squeeze(0).squeeze(0).numpy()
cam_np = cam.detach().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
axes[0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
axes[0].set_title("Input log-mel")
axes[1].imshow(cam_np, origin="lower", aspect="auto", cmap="jet")
axes[1].set_title("Grad-CAM")
axes[2].imshow(mel, origin="lower", aspect="auto", cmap="gray")
axes[2].imshow(cam_np, origin="lower", aspect="auto", cmap="jet", alpha=ALPHA)
axes[2].set_title("Overlay")
for ax in axes:
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bin")
fig.suptitle(
    f"{RUN_DIR.name} | {FOLD} | pred={pred_idx} ({pred_prob:.3f}) | true={y_true} | target={class_idx}",
    fontsize=10,
)
plt.show()


out_dir = fold_dir / "gradcam"
out_dir.mkdir(parents=True, exist_ok=True)
stem = f"{SPLIT}_idx{SAMPLE_INDEX:04d}_cls{pred_idx if TARGET_CLASS is None else int(TARGET_CLASS)}"
png_path = out_dir / f"{stem}.png"
json_path = out_dir / f"{stem}.json"
npy_path = out_dir / f"{stem}_cam.npy"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
np.save(npy_path, cam_np)
payload = {
    "run_dir": str(RUN_DIR),
    "fold": FOLD,
    "checkpoint_path": str(ckpt_path),
    "split": SPLIT,
    "sample_index": int(SAMPLE_INDEX),
    "sample_row": rows[SAMPLE_INDEX],
    "true_class": y_true,
    "pred_class": pred_idx,
    "pred_prob": pred_prob,
    "target_class": pred_idx if TARGET_CLASS is None else int(TARGET_CLASS),
    "target_layer": target_layer_name,
    "png_path": str(png_path),
    "cam_npy_path": str(npy_path),
}
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("saved png :", png_path)
print("saved json:", json_path)
print("saved npy :", npy_path)
