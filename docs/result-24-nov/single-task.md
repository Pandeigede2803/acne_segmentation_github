model	epoch	val_accuracy	val_kappa	criterion
0	ResNet50 single-task	77	0.719101	0.495150	validation CE minimum
1	Proposed augmented Step 11	112	0.758427	0.583841	total multi-task validation loss minimum

class	precision	recall	f1	support
0	0	0.897959	0.628571	0.739496	70
1	1	0.655738	0.930233	0.769231	86
2	2	0.571429	0.210526	0.307692	19
3	3	0.000000	0.000000	0.000000	3
Confusion matrix validation: [[44, 26, 0, 0], [5, 80, 1, 0], [0, 15, 4, 0], [0, 1, 2, 0]]
Macro-F1 validation single-task: 0.4541047188106011
Dice/IoU single-task: tidak berlaku. Test tidak dievaluasi. Objective berbeda; bukan ablasi segmentasi terisolasi.



Konfigurasi sumber dipulihkan dari Drive: /content/drive/MyDrive/acne_project/output/training_runs/2026-11-11_acne04v2_official1204_proposed_offline_aug100_balanced_v4
Single-task image-only; no test. Output: /content/drive/MyDrive/acne_project/output/training_runs/2026-11-11_acne04v2_official1204_proposed_offline_aug100_balanced_v4_single_task_v1/run
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth

  0%|          | 0.00/97.8M [00:00<?, ?B/s]
  7%|▋         | 6.75M/97.8M [00:00<00:01, 70.2MB/s]
 15%|█▍        | 14.4M/97.8M [00:00<00:01, 75.9MB/s]
 32%|███▏      | 31.5M/97.8M [00:00<00:00, 123MB/s] 
 51%|█████     | 49.8M/97.8M [00:00<00:00, 150MB/s]
 67%|██████▋   | 65.9M/97.8M [00:00<00:00, 157MB/s]
 83%|████████▎ | 80.9M/97.8M [00:00<00:00, 156MB/s]
100%|██████████| 97.8M/97.8M [00:00<00:00, 148MB/s]
Single-task ResNet50 | train=1620 val=178 | no masks, no test
Parameters: 23,778,820 | epochs=120
Epoch 1/120 | phase1_freeze
train_acc=0.2543 val_acc=0.4831 val_kappa=0.0000 val_loss=1.3623 best_epoch=1
Epoch 2/120 | phase1_freeze
train_acc=0.3611 val_acc=0.5000 val_kappa=0.0599 val_loss=1.3625 best_epoch=1
Epoch 3/120 | phase1_freeze
train_acc=0.4698 val_acc=0.5112 val_kappa=0.0830 val_loss=1.3501 best_epoch=3
Epoch 4/120 | phase1_freeze
train_acc=0.5086 val_acc=0.5112 val_kappa=0.0993 val_loss=1.3305 best_epoch=4
Epoch 5/120 | phase1_freeze
train_acc=0.5309 val_acc=0.5056 val_kappa=0.0770 val_loss=1.2836 best_epoch=5
Epoch 6/120 | phase1_freeze
train_acc=0.5148 val_acc=0.4831 val_kappa=0.0227 val_loss=1.1988 best_epoch=6
Epoch 7/120 | phase1_freeze
train_acc=0.5611 val_acc=0.5506 val_kappa=0.1743 val_loss=1.1491 best_epoch=7
Epoch 8/120 | phase1_freeze
train_acc=0.5593 val_acc=0.5169 val_kappa=0.1326 val_loss=1.0840 best_epoch=8
Epoch 9/120 | phase1_freeze
train_acc=0.5759 val_acc=0.4719 val_kappa=0.1076 val_loss=1.0231 best_epoch=9
Epoch 10/120 | phase1_freeze
train_acc=0.5846 val_acc=0.4382 val_kappa=0.0777 val_loss=1.0018 best_epoch=10
Epoch 11/120 | phase1_freeze
train_acc=0.6290 val_acc=0.4270 val_kappa=0.0552 val_loss=0.9635 best_epoch=11
Epoch 12/120 | phase1_freeze
train_acc=0.6309 val_acc=0.5000 val_kappa=0.1318 val_loss=0.9493 best_epoch=12
Epoch 13/120 | phase1_freeze
train_acc=0.6383 val_acc=0.5169 val_kappa=0.0945 val_loss=0.9364 best_epoch=13
Epoch 14/120 | phase1_freeze
train_acc=0.6580 val_acc=0.5449 val_kappa=0.1999 val_loss=0.9284 best_epoch=14
Epoch 15/120 | phase1_freeze
train_acc=0.6605 val_acc=0.5449 val_kappa=0.2142 val_loss=0.9230 best_epoch=15
Epoch 16/120 | phase1_freeze
train_acc=0.6901 val_acc=0.5787 val_kappa=0.2310 val_loss=0.9108 best_epoch=16
Epoch 17/120 | phase1_freeze
train_acc=0.7006 val_acc=0.6011 val_kappa=0.2869 val_loss=0.9091 best_epoch=17
Epoch 18/120 | phase1_freeze
train_acc=0.6944 val_acc=0.5449 val_kappa=0.1504 val_loss=0.9043 best_epoch=18
Epoch 19/120 | phase1_freeze
train_acc=0.6901 val_acc=0.5393 val_kappa=0.1422 val_loss=0.9078 best_epoch=18
Epoch 20/120 | phase1_freeze
train_acc=0.7080 val_acc=0.6067 val_kappa=0.2973 val_loss=0.8921 best_epoch=20
Epoch 21/120 | phase1_freeze
train_acc=0.7204 val_acc=0.5506 val_kappa=0.1698 val_loss=0.8942 best_epoch=20
Epoch 22/120 | phase1_freeze
train_acc=0.7284 val_acc=0.6067 val_kappa=0.3043 val_loss=0.8808 best_epoch=22
Epoch 23/120 | phase1_freeze
train_acc=0.7321 val_acc=0.5955 val_kappa=0.2725 val_loss=0.8680 best_epoch=23
Epoch 24/120 | phase1_freeze
train_acc=0.7543 val_acc=0.5843 val_kappa=0.2311 val_loss=0.8640 best_epoch=24
Epoch 25/120 | phase1_freeze
train_acc=0.7636 val_acc=0.6124 val_kappa=0.2976 val_loss=0.8825 best_epoch=24
Epoch 26/120 | phase1_freeze
train_acc=0.7549 val_acc=0.5730 val_kappa=0.2010 val_loss=0.8543 best_epoch=26
Epoch 27/120 | phase1_freeze
train_acc=0.7765 val_acc=0.6236 val_kappa=0.3182 val_loss=0.8410 best_epoch=27
Epoch 28/120 | phase1_freeze
train_acc=0.7698 val_acc=0.6124 val_kappa=0.2817 val_loss=0.8680 best_epoch=27
Epoch 29/120 | phase1_freeze
train_acc=0.7722 val_acc=0.6011 val_kappa=0.2635 val_loss=0.8454 best_epoch=27
Epoch 30/120 | phase1_freeze
train_acc=0.7951 val_acc=0.5899 val_kappa=0.2345 val_loss=0.8725 best_epoch=27
Epoch 31/120 | phase2_unfreeze
train_acc=0.8000 val_acc=0.6180 val_kappa=0.2963 val_loss=0.8370 best_epoch=31
Epoch 32/120 | phase2_unfreeze
train_acc=0.8031 val_acc=0.6348 val_kappa=0.3285 val_loss=0.8422 best_epoch=31
Epoch 33/120 | phase2_unfreeze
train_acc=0.8185 val_acc=0.6348 val_kappa=0.3317 val_loss=0.8287 best_epoch=33
Epoch 34/120 | phase2_unfreeze
train_acc=0.8179 val_acc=0.6404 val_kappa=0.3481 val_loss=0.8127 best_epoch=34
Epoch 35/120 | phase2_unfreeze
train_acc=0.8093 val_acc=0.6292 val_kappa=0.3170 val_loss=0.8225 best_epoch=34
Epoch 36/120 | phase2_unfreeze
train_acc=0.8340 val_acc=0.6517 val_kappa=0.3567 val_loss=0.8019 best_epoch=36
Epoch 37/120 | phase2_unfreeze
train_acc=0.8383 val_acc=0.6573 val_kappa=0.3700 val_loss=0.8020 best_epoch=36
Epoch 38/120 | phase2_unfreeze
train_acc=0.8506 val_acc=0.6629 val_kappa=0.3821 val_loss=0.7835 best_epoch=38
Epoch 39/120 | phase2_unfreeze
train_acc=0.8537 val_acc=0.6629 val_kappa=0.3832 val_loss=0.8026 best_epoch=38
Epoch 40/120 | phase2_unfreeze
train_acc=0.8531 val_acc=0.6742 val_kappa=0.4042 val_loss=0.7845 best_epoch=38
Epoch 41/120 | phase2_unfreeze
train_acc=0.8562 val_acc=0.6798 val_kappa=0.4124 val_loss=0.7633 best_epoch=41
Epoch 42/120 | phase2_unfreeze
train_acc=0.8568 val_acc=0.6685 val_kappa=0.3934 val_loss=0.7839 best_epoch=41
Epoch 43/120 | phase2_unfreeze
train_acc=0.8481 val_acc=0.6573 val_kappa=0.3670 val_loss=0.7681 best_epoch=41
Epoch 44/120 | phase2_unfreeze
train_acc=0.8617 val_acc=0.6517 val_kappa=0.3650 val_loss=0.7618 best_epoch=44
Epoch 45/120 | phase2_unfreeze
train_acc=0.8679 val_acc=0.6798 val_kappa=0.4129 val_loss=0.7678 best_epoch=44
Epoch 46/120 | phase2_unfreeze
train_acc=0.8716 val_acc=0.6798 val_kappa=0.4118 val_loss=0.7714 best_epoch=44
Epoch 47/120 | phase2_unfreeze
train_acc=0.8728 val_acc=0.6966 val_kappa=0.4453 val_loss=0.7766 best_epoch=44
Epoch 48/120 | phase2_unfreeze
train_acc=0.8716 val_acc=0.6854 val_kappa=0.4271 val_loss=0.7281 best_epoch=48
Epoch 49/120 | phase2_unfreeze
train_acc=0.8722 val_acc=0.6910 val_kappa=0.4372 val_loss=0.7319 best_epoch=48
Epoch 50/120 | phase2_unfreeze
train_acc=0.8877 val_acc=0.6910 val_kappa=0.4341 val_loss=0.7470 best_epoch=48
Epoch 51/120 | phase2_unfreeze
train_acc=0.8895 val_acc=0.6854 val_kappa=0.4265 val_loss=0.7300 best_epoch=48
Epoch 52/120 | phase2_unfreeze
train_acc=0.8883 val_acc=0.6742 val_kappa=0.3998 val_loss=0.7543 best_epoch=48
Epoch 53/120 | phase2_unfreeze
train_acc=0.8802 val_acc=0.6910 val_kappa=0.4389 val_loss=0.7081 best_epoch=53
Epoch 54/120 | phase2_unfreeze
train_acc=0.8877 val_acc=0.6910 val_kappa=0.4341 val_loss=0.7458 best_epoch=53
Epoch 55/120 | phase2_unfreeze
train_acc=0.8944 val_acc=0.6742 val_kappa=0.4025 val_loss=0.7681 best_epoch=53
Epoch 56/120 | phase2_unfreeze
train_acc=0.8975 val_acc=0.6798 val_kappa=0.4044 val_loss=0.8014 best_epoch=53
Epoch 57/120 | phase2_unfreeze
train_acc=0.8944 val_acc=0.6798 val_kappa=0.4100 val_loss=0.7517 best_epoch=53
Epoch 58/120 | phase2_unfreeze
train_acc=0.9049 val_acc=0.6910 val_kappa=0.4351 val_loss=0.7339 best_epoch=53
Epoch 59/120 | phase2_unfreeze
train_acc=0.8969 val_acc=0.6742 val_kappa=0.4016 val_loss=0.7424 best_epoch=53
Epoch 60/120 | phase2_unfreeze
train_acc=0.9031 val_acc=0.6798 val_kappa=0.4113 val_loss=0.7620 best_epoch=53
Epoch 61/120 | phase2_unfreeze
train_acc=0.9031 val_acc=0.6348 val_kappa=0.3204 val_loss=0.7808 best_epoch=53
Epoch 62/120 | phase2_unfreeze
train_acc=0.9099 val_acc=0.6629 val_kappa=0.3812 val_loss=0.7497 best_epoch=53
Epoch 63/120 | phase2_unfreeze
train_acc=0.9000 val_acc=0.6854 val_kappa=0.4267 val_loss=0.7277 best_epoch=53
Epoch 64/120 | phase2_unfreeze
train_acc=0.9272 val_acc=0.6910 val_kappa=0.4351 val_loss=0.7140 best_epoch=53
Epoch 65/120 | phase2_unfreeze
train_acc=0.9068 val_acc=0.6910 val_kappa=0.4324 val_loss=0.7326 best_epoch=53
Epoch 66/120 | phase2_unfreeze
train_acc=0.9222 val_acc=0.7079 val_kappa=0.4659 val_loss=0.7544 best_epoch=53
Epoch 67/120 | phase2_unfreeze
train_acc=0.9179 val_acc=0.7022 val_kappa=0.4531 val_loss=0.7590 best_epoch=53
Epoch 68/120 | phase2_unfreeze
train_acc=0.9198 val_acc=0.6966 val_kappa=0.4422 val_loss=0.7413 best_epoch=53
Epoch 69/120 | phase2_unfreeze
train_acc=0.9160 val_acc=0.6517 val_kappa=0.3586 val_loss=0.7640 best_epoch=53
Epoch 70/120 | phase2_unfreeze
train_acc=0.9167 val_acc=0.7079 val_kappa=0.4665 val_loss=0.7508 best_epoch=53
Epoch 71/120 | phase2_unfreeze
train_acc=0.9253 val_acc=0.6910 val_kappa=0.4380 val_loss=0.7199 best_epoch=53
Epoch 72/120 | phase2_unfreeze
train_acc=0.9247 val_acc=0.6910 val_kappa=0.4380 val_loss=0.7483 best_epoch=53
Epoch 73/120 | phase2_unfreeze
train_acc=0.9235 val_acc=0.6854 val_kappa=0.4218 val_loss=0.7565 best_epoch=53
Epoch 74/120 | phase2_unfreeze
train_acc=0.9395 val_acc=0.7022 val_kappa=0.4605 val_loss=0.7478 best_epoch=53
Epoch 75/120 | phase2_unfreeze
train_acc=0.9167 val_acc=0.6966 val_kappa=0.4477 val_loss=0.7586 best_epoch=53
Epoch 76/120 | phase2_unfreeze
train_acc=0.9222 val_acc=0.7079 val_kappa=0.4701 val_loss=0.7108 best_epoch=53
Epoch 77/120 | phase2_unfreeze
train_acc=0.9284 val_acc=0.7191 val_kappa=0.4952 val_loss=0.6906 best_epoch=77
Epoch 78/120 | phase2_unfreeze
train_acc=0.9340 val_acc=0.7416 val_kappa=0.5364 val_loss=0.7122 best_epoch=77
Epoch 79/120 | phase2_unfreeze
train_acc=0.9340 val_acc=0.7135 val_kappa=0.4802 val_loss=0.7506 best_epoch=77
Epoch 80/120 | phase2_unfreeze
train_acc=0.9253 val_acc=0.7135 val_kappa=0.4818 val_loss=0.7252 best_epoch=77
Epoch 81/120 | phase2_unfreeze
train_acc=0.9364 val_acc=0.7247 val_kappa=0.5025 val_loss=0.7298 best_epoch=77
Epoch 82/120 | phase2_unfreeze
train_acc=0.9370 val_acc=0.6910 val_kappa=0.4374 val_loss=0.7446 best_epoch=77
Epoch 83/120 | phase2_unfreeze
train_acc=0.9426 val_acc=0.7247 val_kappa=0.5039 val_loss=0.7123 best_epoch=77
Epoch 84/120 | phase2_unfreeze
train_acc=0.9475 val_acc=0.7191 val_kappa=0.4933 val_loss=0.7186 best_epoch=77
Epoch 85/120 | phase2_unfreeze
train_acc=0.9407 val_acc=0.7022 val_kappa=0.4585 val_loss=0.7801 best_epoch=77
Epoch 86/120 | phase2_unfreeze
train_acc=0.9340 val_acc=0.7191 val_kappa=0.4909 val_loss=0.7542 best_epoch=77
Epoch 87/120 | phase2_unfreeze
train_acc=0.9481 val_acc=0.6966 val_kappa=0.4477 val_loss=0.7714 best_epoch=77
Epoch 88/120 | phase2_unfreeze
train_acc=0.9432 val_acc=0.7135 val_kappa=0.4787 val_loss=0.7633 best_epoch=77
Epoch 89/120 | phase2_unfreeze
train_acc=0.9475 val_acc=0.7247 val_kappa=0.5035 val_loss=0.7439 best_epoch=77
Epoch 90/120 | phase2_unfreeze
train_acc=0.9272 val_acc=0.7135 val_kappa=0.4793 val_loss=0.7399 best_epoch=77
Epoch 91/120 | phase2_unfreeze
train_acc=0.9401 val_acc=0.7022 val_kappa=0.4626 val_loss=0.7557 best_epoch=77
Epoch 92/120 | phase2_unfreeze
train_acc=0.9426 val_acc=0.7247 val_kappa=0.5039 val_loss=0.7347 best_epoch=77
Epoch 93/120 | phase2_unfreeze
train_acc=0.9469 val_acc=0.7135 val_kappa=0.4886 val_loss=0.7346 best_epoch=77
Epoch 94/120 | phase2_unfreeze
train_acc=0.9481 val_acc=0.7247 val_kappa=0.5041 val_loss=0.7345 best_epoch=77
Epoch 95/120 | phase2_unfreeze
train_acc=0.9549 val_acc=0.7247 val_kappa=0.5015 val_loss=0.7214 best_epoch=77
Epoch 96/120 | phase2_unfreeze
train_acc=0.9475 val_acc=0.6966 val_kappa=0.4488 val_loss=0.7536 best_epoch=77
Epoch 97/120 | phase2_unfreeze
train_acc=0.9636 val_acc=0.7360 val_kappa=0.5224 val_loss=0.7288 best_epoch=77
Epoch 98/120 | phase2_unfreeze
train_acc=0.9432 val_acc=0.7022 val_kappa=0.4605 val_loss=0.7470 best_epoch=77
Epoch 99/120 | phase2_unfreeze
train_acc=0.9494 val_acc=0.7079 val_kappa=0.4763 val_loss=0.7529 best_epoch=77
Epoch 100/120 | phase2_unfreeze
train_acc=0.9519 val_acc=0.7022 val_kappa=0.4626 val_loss=0.7612 best_epoch=77
Epoch 101/120 | phase2_unfreeze
train_acc=0.9531 val_acc=0.7247 val_kappa=0.5015 val_loss=0.7272 best_epoch=77
Epoch 102/120 | phase2_unfreeze
train_acc=0.9568 val_acc=0.7135 val_kappa=0.4812 val_loss=0.6996 best_epoch=77
Epoch 103/120 | phase2_unfreeze
train_acc=0.9525 val_acc=0.7135 val_kappa=0.4818 val_loss=0.7106 best_epoch=77
Epoch 104/120 | phase2_unfreeze
train_acc=0.9568 val_acc=0.7191 val_kappa=0.4928 val_loss=0.6944 best_epoch=77
Epoch 105/120 | phase2_unfreeze
train_acc=0.9611 val_acc=0.7191 val_kappa=0.4929 val_loss=0.7289 best_epoch=77
Epoch 106/120 | phase2_unfreeze
train_acc=0.9636 val_acc=0.6966 val_kappa=0.4492 val_loss=0.7483 best_epoch=77
Epoch 107/120 | phase2_unfreeze
train_acc=0.9593 val_acc=0.7303 val_kappa=0.5136 val_loss=0.7205 best_epoch=77
Epoch 108/120 | phase2_unfreeze
train_acc=0.9630 val_acc=0.7191 val_kappa=0.4977 val_loss=0.7287 best_epoch=77
Epoch 109/120 | phase2_unfreeze
train_acc=0.9586 val_acc=0.7191 val_kappa=0.4924 val_loss=0.7330 best_epoch=77
Epoch 110/120 | phase2_unfreeze
train_acc=0.9617 val_acc=0.7079 val_kappa=0.4810 val_loss=0.7229 best_epoch=77
Epoch 111/120 | phase2_unfreeze
train_acc=0.9580 val_acc=0.7360 val_kappa=0.5250 val_loss=0.7163 best_epoch=77
Epoch 112/120 | phase2_unfreeze
train_acc=0.9623 val_acc=0.7303 val_kappa=0.5159 val_loss=0.7134 best_epoch=77
Epoch 113/120 | phase2_unfreeze
train_acc=0.9667 val_acc=0.7247 val_kappa=0.5076 val_loss=0.7118 best_epoch=77
Epoch 114/120 | phase2_unfreeze
train_acc=0.9611 val_acc=0.7360 val_kappa=0.5274 val_loss=0.7513 best_epoch=77
Epoch 115/120 | phase2_unfreeze
train_acc=0.9593 val_acc=0.7191 val_kappa=0.4939 val_loss=0.7821 best_epoch=77
Epoch 116/120 | phase2_unfreeze
train_acc=0.9691 val_acc=0.7191 val_kappa=0.4981 val_loss=0.7589 best_epoch=77
Epoch 117/120 | phase2_unfreeze
train_acc=0.9722 val_acc=0.7247 val_kappa=0.5035 val_loss=0.7692 best_epoch=77
Epoch 118/120 | phase2_unfreeze
train_acc=0.9673 val_acc=0.7022 val_kappa=0.4616 val_loss=0.8061 best_epoch=77
Epoch 119/120 | phase2_unfreeze
train_acc=0.9660 val_acc=0.7135 val_kappa=0.4812 val_loss=0.7566 best_epoch=77
Epoch 120/120 | phase2_unfreeze
train_acc=0.9648 val_acc=0.6798 val_kappa=0.4197 val_loss=0.8092 best_epoch=77
Selesai. Hasil hanya validation; Dice/IoU tidak berlaku untuk single-task klasifikasi.