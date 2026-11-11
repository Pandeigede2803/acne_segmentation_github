Manifest ablation (tanpa test): /content/drive/MyDrive/acne_project/output/flowchart_experiments_2026-10-24/ablation/train_val_manifest.csv
Variasi: ['baseline', 'no_attention', 'no_dilation', 'no_amfm', 'lds', 'no_fusion']
✅ Varian 1/6 (baseline) sudah selesai; memakai metrics.json yang tersimpan.
✅ Varian 2/6 (no_attention) sudah selesai; memakai metrics.json yang tersimpan.
✅ Varian 3/6 (no_dilation) sudah selesai; memakai metrics.json yang tersimpan.
✅ Varian 4/6 (no_amfm) sudah selesai; memakai metrics.json yang tersimpan.
✅ Varian 5/6 (lds) sudah selesai; memakai metrics.json yang tersimpan.
✅ Varian 6/6 (no_fusion) sudah selesai; memakai metrics.json yang tersimpan.
🎉 Semua ablation selesai dalam 0:00:08.
variant	best_epoch	val_loss	train_accuracy	val_accuracy	val_kappa	val_dice	val_iou
5	no_fusion	112	0.977858	0.839725	0.638889	0.483127	0.507655	0.356523
1	no_attention	111	0.989726	0.772861	0.532407	0.353159	0.515357	0.362323
2	no_dilation	115	1.068402	0.786627	0.527778	0.344949	0.520417	0.366425
3	no_amfm	107	1.081204	0.867257	0.601852	0.423088	0.515317	0.361364
4	lds	117	1.313444	0.740413	0.500000	0.317276	0.502523	0.351313
0	baseline	96	1.414542	0.801377	0.601852	0.445790	0.509552	0.359018