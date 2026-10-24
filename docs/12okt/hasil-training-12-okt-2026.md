Script perlu diperbarui. Mengambil script training dari GitHub...
Script training berhasil diperbarui.
Checkpoint ditemukan. Melanjutkan dari epoch terakhir yang tersimpan.
Class weighting: balanced
Output: /content/drive/MyDrive/acne_project/output/training_runs/2026-10-04_proposal_class_weighting_balanced
🚀 Proposal hybrid training: ResNet50 + Spatial Attention + Decoder + AMFM + MA-LDS
🧾 Manifest: /content/drive/MyDrive/acne_project/output/refined_masks_output/refined_manifest.csv
🖥️ Device: cuda
✅ Split: train=1017, val=216, test=224
⚖️ Class weights: [0.7327, 0.5713, 1.9558, 2.6763]
Train class counts: [347, 445, 130, 95]
Class weighting: balanced; weighted sampler: False
🧠 Phase 1: freeze encoder, epochs=30, lr=0.001
🧠 Phase 2: full fine-tuning, epochs=90, lr=0.0001
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth

  0%|          | 0.00/97.8M [00:00<?, ?B/s]
 11%|█         | 10.6M/97.8M [00:00<00:00, 111MB/s]
 33%|███▎      | 32.0M/97.8M [00:00<00:00, 177MB/s]
 54%|█████▎    | 52.5M/97.8M [00:00<00:00, 193MB/s]
 73%|███████▎  | 71.2M/97.8M [00:00<00:00, 194MB/s]
 92%|█████████▏| 89.8M/97.8M [00:00<00:00, 174MB/s]
100%|██████████| 97.8M/97.8M [00:00<00:00, 178MB/s]
🔁 Resume dari epoch 121
🧪 Evaluasi akhir test set
✅ Training selesai.
🏁 Best model: /content/drive/MyDrive/acne_project/output/training_runs/2026-10-04_proposal_class_weighting_balanced/best_model.pt
🔁 Last checkpoint: /content/drive/MyDrive/acne_project/output/training_runs/2026-10-04_proposal_class_weighting_balanced/last_checkpoint.pt
📊 Metrics: /content/drive/MyDrive/acne_project/output/training_runs/2026-10-04_proposal_class_weighting_balanced/metrics.json
🧾 Test | loss=1.1247 acc=0.6071 kappa=0.4346 dice=0.4430 iou=0.3016

Epoch 120 | Val Dice: 0.4668 | Val Acc: 0.5694



=============================================
       TEST METRICS — HASIL AKHIR
=============================================
  Accuracy      : 0.6071
  Cohen Kappa   : 0.4346
  Dice Score    : 0.4430
  IoU           : 0.3016
---------------------------------------------
  Total Loss    : 1.1247
  Seg Loss      : 0.6538
  Cls Loss      : 0.4701
=============================================
  Best Val Loss : 1.1025
  Epochs (P1+P2): 30 + 90
  Batch size    : 16
  Image size    : 320

=== REKAP DARI metrics.json ===
                                         run   dice    iou    acc  kappa
              proposal_hybrid_resnet50_malds 0.4515 0.3082 0.5000 0.0652
2026-10-04_proposal_class_weighting_balanced 0.4430 0.3016 0.6071 0.4346