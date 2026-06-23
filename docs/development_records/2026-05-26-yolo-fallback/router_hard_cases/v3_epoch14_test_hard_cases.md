# Router hard cases: v3 epoch14 test set

Generated: 2026-05-26 UTC  
Model: `coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv3_recall_ft_b48_e15/weights/epoch14.pt`  
Dataset: `handoff_20260519/.../datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2`  
Split: `test`  
Matching rule: prediction `conf >= 0.25`, GT/pred matched by IoU >= 0.5. Low-confidence correct cases are correct class matches with `0.25 <= conf < 0.35`.

## Summary

- Images scanned: 348
- GT instances: 676
- Hard-case rows written to CSV: 221
- Misclassified matched GT: 26
- Missed GT: 91
- Low-confidence correct matches: 7
- False positives at conf >= 0.25: 97

## Confusion / error direction counts

- `壁类 -> none`: 71
- `none -> 壁类`: 65
- `none -> 天井`: 21
- `天井 -> none`: 14
- `RC柱 -> 壁类`: 13
- `none -> RC柱`: 11
- `RC柱 -> none`: 6
- `壁类 -> 天井`: 5
- `天井 -> 壁类`: 4
- `壁类 -> RC柱`: 4

## Main finding

The important training target is still the boundary between `壁类` and `RC柱`, plus missed/low-confidence wall-like regions. For Gemini synthetic-data generation, prioritize hard scenes that look like these failure cases rather than adding generic clean wall/column images.

Recommended synthetic categories:

1. `壁类 <-> RC柱` junctions: partial columns, wall-column seams, column edges cropped by the photo boundary.
2. Wall surfaces with vertical structural lines, shadows, pipes, or corners that resemble RC columns.
3. RC columns partly occluded by walls, furniture, debris, or close-up framing.
4. Ceiling-wall junctions and oblique indoor shots if `天井 <-> 壁类` appears in the table below.
5. Low-light / motion-blur / disaster-site clutter, but keep bounding boxes accurate.

## Wall / RC柱 related hard cases

| image | type | GT | pred | conf | IoU | note |
|---|---:|---:|---:|---:|---:|---|
| `RC壁_c-199_03206.jpg` | misclass | RC柱 | 壁类 | 0.746 | 0.951 |  |
| `RC壁_c-40017_03381.jpg` | misclass | RC柱 | 壁类 | 0.357 | 0.955 |  |
| `RC壁_c-40436_03251.jpg` | misclass | RC柱 | 壁类 | 0.714 | 0.955 |  |
| `RC壁_c-40477_03184.jpg` | misclass | RC柱 | 壁类 | 0.309 | 0.969 |  |
| `RC壁_c-40616_03440.jpg` | misclass | 壁类 | RC柱 | 0.422 | 0.952 |  |
| `RC壁_c-40634_03446.jpg` | misclass | 壁类 | RC柱 | 0.876 | 0.711 |  |
| `RC壁_ｃ-10117_03195.jpg` | misclass | RC柱 | 壁类 | 0.932 | 0.980 |  |
| `RC柱_4-B-00139_03309.jpg` | misclass | 壁类 | RC柱 | 0.933 | 0.753 |  |
| `RC柱_4-B-00234_03324.jpg` | misclass | RC柱 | 壁类 | 0.709 | 0.906 |  |
| `RC柱_d-40027_03307.jpg` | misclass | RC柱 | 壁类 | 0.699 | 0.987 |  |
| `RC柱_d-40136_03437.jpg` | misclass | RC柱 | 壁类 | 0.483 | 0.714 |  |
| `内壁_2-C-20098_03380.jpg` | misclass | RC柱 | 壁类 | 0.939 | 0.564 |  |
| `内壁_b-30134_03348.jpg` | misclass | RC柱 | 壁类 | 0.863 | 0.859 |  |
| `内壁_b-40006_03323.jpg` | misclass | RC柱 | 壁类 | 0.637 | 0.779 |  |
| `天井_1-C-10051_03246.jpg` | misclass | 壁类 | RC柱 | 0.886 | 0.994 |  |
| `天井_a-40217_03341.jpg` | misclass | RC柱 | 壁类 | 0.553 | 0.973 |  |
| `天井_a-40245_03190.jpg` | misclass | RC柱 | 壁类 | 0.796 | 0.886 |  |
| `RC壁_3-D-40769_03271.jpg` | missed_gt | 壁类 | RC柱 | 0.007 | 0.634 | matched only below conf threshold |
| `RC壁_c-40471_03282.jpg` | missed_gt | 壁类 | RC柱 | 0.013 | 0.718 | matched only below conf threshold |
| `RC壁_c-40544_03422.jpg` | missed_gt | RC柱 | 壁类 | 0.222 | 0.973 | matched only below conf threshold |
| `内壁_b-10065_03283.jpg` | missed_gt | 壁类 | RC柱 | 0.001 | 0.900 | matched only below conf threshold |
| `天井_a-20054_03400.jpg` | missed_gt | RC柱 | 壁类 | 0.406 | 0.087 | no pred with IoU>=0.5 at conf threshold |
| `天井_a-40307_03332.jpg` | missed_gt | RC柱 | 壁类 | 0.001 | 0.778 | matched only below conf threshold |


## Misclassified matched boxes

| image | type | GT | pred | conf | IoU | note |
|---|---:|---:|---:|---:|---:|---|
| `RC壁_c-199_03206.jpg` | misclass | RC柱 | 壁类 | 0.746 | 0.951 |  |
| `RC壁_c-40017_03381.jpg` | misclass | RC柱 | 壁类 | 0.357 | 0.955 |  |
| `RC壁_c-40436_03251.jpg` | misclass | RC柱 | 壁类 | 0.714 | 0.955 |  |
| `RC壁_c-40477_03184.jpg` | misclass | RC柱 | 壁类 | 0.309 | 0.969 |  |
| `RC壁_c-40616_03440.jpg` | misclass | 壁类 | RC柱 | 0.422 | 0.952 |  |
| `RC壁_c-40634_03446.jpg` | misclass | 壁类 | RC柱 | 0.876 | 0.711 |  |
| `RC壁_ｃ-10117_03195.jpg` | misclass | RC柱 | 壁类 | 0.932 | 0.980 |  |
| `RC柱_4-B-00139_03309.jpg` | misclass | 壁类 | RC柱 | 0.933 | 0.753 |  |
| `RC柱_4-B-00234_03324.jpg` | misclass | RC柱 | 壁类 | 0.709 | 0.906 |  |
| `RC柱_d-40027_03307.jpg` | misclass | RC柱 | 壁类 | 0.699 | 0.987 |  |
| `RC柱_d-40136_03437.jpg` | misclass | RC柱 | 壁类 | 0.483 | 0.714 |  |
| `内壁_2-C-20098_03380.jpg` | misclass | RC柱 | 壁类 | 0.939 | 0.564 |  |
| `内壁_b-30134_03348.jpg` | misclass | RC柱 | 壁类 | 0.863 | 0.859 |  |
| `内壁_b-40006_03323.jpg` | misclass | RC柱 | 壁类 | 0.637 | 0.779 |  |
| `天井_1-C-10051_03246.jpg` | misclass | 壁类 | RC柱 | 0.886 | 0.994 |  |
| `天井_a-40217_03341.jpg` | misclass | RC柱 | 壁类 | 0.553 | 0.973 |  |
| `天井_a-40245_03190.jpg` | misclass | RC柱 | 壁类 | 0.796 | 0.886 |  |
| `RC壁_c-10058_03453.jpg` | misclass | 壁类 | 天井 | 0.978 | 0.987 |  |
| `RC壁_c-40023_03254.jpg` | misclass | 天井 | 壁类 | 0.527 | 0.969 |  |
| `内壁_b-20042_03203.jpg` | misclass | 壁类 | 天井 | 0.747 | 0.801 |  |
| `内壁_b-30134_03348.jpg` | misclass | 壁类 | 天井 | 0.915 | 0.518 |  |
| `内壁_b-40544_03215.jpg` | misclass | 壁类 | 天井 | 0.509 | 0.654 |  |
| `内壁_b-40571_03288.jpg` | misclass | 天井 | 壁类 | 0.973 | 0.967 |  |
| `天井_1-C-00027_03306.jpg` | misclass | 天井 | 壁类 | 0.960 | 0.925 |  |
| `天井_a-40102_03331.jpg` | misclass | 壁类 | 天井 | 0.391 | 0.977 |  |
| `天井_a-40252_03441.jpg` | misclass | 天井 | 壁类 | 0.906 | 0.992 |  |


## Missed GT boxes

| image | type | GT | pred | conf | IoU | note |
|---|---:|---:|---:|---:|---:|---|
| `RC壁_3-D-40769_03271.jpg` | missed_gt | 壁类 | RC柱 | 0.007 | 0.634 | matched only below conf threshold |
| `RC壁_c-40471_03282.jpg` | missed_gt | 壁类 | RC柱 | 0.013 | 0.718 | matched only below conf threshold |
| `RC壁_c-40544_03422.jpg` | missed_gt | RC柱 | 壁类 | 0.222 | 0.973 | matched only below conf threshold |
| `内壁_b-10065_03283.jpg` | missed_gt | 壁类 | RC柱 | 0.001 | 0.900 | matched only below conf threshold |
| `天井_a-20054_03400.jpg` | missed_gt | RC柱 | 壁类 | 0.406 | 0.087 | no pred with IoU>=0.5 at conf threshold |
| `天井_a-40307_03332.jpg` | missed_gt | RC柱 | 壁类 | 0.001 | 0.778 | matched only below conf threshold |
| `RC壁_3-B-00051_03223.jpg` | missed_gt | 天井 | 壁类 | 0.002 | 0.850 | matched only below conf threshold |
| `RC壁_3-D-40769_03271.jpg` | missed_gt | 壁类 | 壁类 | 0.978 | 0.032 | no pred with IoU>=0.5 at conf threshold |
| `RC壁_3-D-40770_03225.jpg` | missed_gt | 壁类 | 壁类 | 0.106 | 0.829 | matched only below conf threshold |
| `RC壁_3-D-40770_03225.jpg` | missed_gt | 壁类 | 壁类 | 0.022 | 0.552 | matched only below conf threshold |
| `RC壁_c-10058_03453.jpg` | missed_gt | 壁类 | 天井 | 0.978 | 0.851 |  |
| `RC壁_c-10061_03412.jpg` | missed_gt | RC柱 | RC柱 | 0.007 | 0.912 | matched only below conf threshold |
| `RC壁_c-20023_03445.jpg` | missed_gt | 壁类 | 壁类 | 0.002 | 0.852 | matched only below conf threshold |
| `RC壁_c-40023_03254.jpg` | missed_gt | 壁类 | 壁类 | 0.585 | 0.438 | no pred with IoU>=0.5 at conf threshold |
| `RC壁_c-40023_03254.jpg` | missed_gt | 壁类 | 壁类 | 0.014 | 0.792 | matched only below conf threshold |
| `RC壁_c-40023_03254.jpg` | missed_gt | 壁类 | 壁类 | 0.002 | 0.964 | matched only below conf threshold |
| `RC壁_c-40143_03458.jpg` | missed_gt | 天井 | 天井 | 0.149 | 0.840 | matched only below conf threshold |
| `RC壁_c-40304_03168.jpg` | missed_gt | 壁类 | 壁类 | 0.002 | 0.672 | matched only below conf threshold |
| `RC壁_c-40471_03282.jpg` | missed_gt | RC柱 | RC柱 | 0.139 | 0.926 | matched only below conf threshold |
| `RC壁_c-40503_03214.jpg` | missed_gt | 壁类 | 壁类 | 0.984 | 0.443 | no pred with IoU>=0.5 at conf threshold |
| `RC壁_c-40503_03214.jpg` | missed_gt | 壁类 | 壁类 | 0.984 | 0.335 | no pred with IoU>=0.5 at conf threshold |
| `RC壁_c-40544_03422.jpg` | missed_gt | 壁类 | 壁类 | 0.003 | 0.866 | matched only below conf threshold |
| `RC壁_c-40546_03294.jpg` | missed_gt | 壁类 | 壁类 | 0.001 | 0.971 | matched only below conf threshold |
| `RC壁_c-40551_03488.jpg` | missed_gt | 天井 | 天井 | 0.012 | 0.608 | matched only below conf threshold |
| `RC壁_c-40582_03434.jpg` | missed_gt | 壁类 | 壁类 | 0.005 | 0.980 | matched only below conf threshold |
| `RC壁_c-40634_03446.jpg` | missed_gt | 天井 | 壁类 | 0.047 | 0.432 | no pred with IoU>=0.5 at conf threshold |
| `RC壁_c-67_03378.jpg` | missed_gt | 壁类 | 壁类 | 0.001 | 0.538 | matched only below conf threshold |
| `RC壁_ｃ-10117_03195.jpg` | missed_gt | 天井 | 天井 | 0.002 | 0.923 | matched only below conf threshold |
| `RC柱_4-B-00123_03404.jpg` | missed_gt | 壁类 | 壁类 | 0.100 | 0.705 | matched only below conf threshold |
| `RC柱_4-B-10043_03427.jpg` | missed_gt | 壁类 | 壁类 | 0.036 | 0.268 | no pred with IoU>=0.5 at conf threshold |
| `RC柱_d-123_03362.jpg` | missed_gt | 壁类 | 壁类 | 0.100 | 0.705 | matched only below conf threshold |
| `RC柱_d-144_03249.jpg` | missed_gt | 壁类 | 壁类 | 0.010 | 0.908 | matched only below conf threshold |
| `RC柱_d-144_03249.jpg` | missed_gt | 壁类 | 壁类 | 0.010 | 0.963 | matched only below conf threshold |
| `RC柱_d-179_03178.jpg` | missed_gt | 壁类 | 壁类 | 0.181 | 0.932 | matched only below conf threshold |
| `RC柱_d-179_03178.jpg` | missed_gt | 壁类 | 壁类 | 0.015 | 0.915 | matched only below conf threshold |
| `RC柱_d-20029_03268.jpg` | missed_gt | 壁类 | 壁类 | 0.165 | 0.906 | matched only below conf threshold |
| `RC柱_d-231_03334.jpg` | missed_gt | 壁类 | 壁类 | 0.001 | 0.949 | matched only below conf threshold |
| `RC柱_d-40078_03229.jpg` | missed_gt | 壁类 | 壁类 | 0.005 | 0.786 | matched only below conf threshold |
| `RC柱_d-40079_03304.jpg` | missed_gt | 壁类 | 壁类 | 0.190 | 0.960 | matched only below conf threshold |
| `RC柱_d-40079_03304.jpg` | missed_gt | 壁类 | 壁类 | 0.010 | 0.956 | matched only below conf threshold |
| ... | 51 more | | | | | see CSV |


## Low-confidence correct boxes

| image | type | GT | pred | conf | IoU | note |
|---|---:|---:|---:|---:|---:|---|
| `RC壁_c-40606_03224.jpg` | low_conf_correct | 壁类 | 壁类 | 0.335 | 0.895 | class correct but confidence near threshold |
| `RC柱_4-B-10043_03427.jpg` | low_conf_correct | 壁类 | 壁类 | 0.304 | 0.910 | class correct but confidence near threshold |
| `RC柱_d-189_03370.jpg` | low_conf_correct | 天井 | 天井 | 0.335 | 0.944 | class correct but confidence near threshold |
| `天井_a-20054_03400.jpg` | low_conf_correct | 天井 | 天井 | 0.346 | 0.901 | class correct but confidence near threshold |
| `天井_a-30041_03401.jpg` | low_conf_correct | 天井 | 天井 | 0.311 | 0.939 | class correct but confidence near threshold |
| `天井_a-40289_03339.jpg` | low_conf_correct | RC柱 | RC柱 | 0.349 | 0.818 | class correct but confidence near threshold |
| `天井_a-40330_03260.jpg` | low_conf_correct | 壁类 | 壁类 | 0.286 | 0.918 | class correct but confidence near threshold |


## Images with multiple major hard cases

| image | type | GT | pred | conf | IoU | note |
|---|---:|---:|---:|---:|---:|---|
| `内壁_b-40544_03215.jpg` | 4 major |  |  |  |  | missed_gt:天井->壁类, missed_gt:壁类->壁类, misclass:壁类->天井, missed_gt:壁类->壁类 |
| `RC壁_c-40023_03254.jpg` | 4 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类, misclass:天井->壁类, missed_gt:壁类->壁类 |
| `天井_a-40236_03394.jpg` | 3 major |  |  |  |  | missed_gt:RC柱->RC柱, missed_gt:天井->天井, missed_gt:壁类->天井 |
| `天井_a-40102_03331.jpg` | 3 major |  |  |  |  | misclass:壁类->天井, missed_gt:壁类->天井, missed_gt:壁类->壁类 |
| `天井_a-10061_03475.jpg` | 3 major |  |  |  |  | missed_gt:天井->天井, missed_gt:天井->壁类, missed_gt:天井->壁类 |
| `内壁_b-30134_03348.jpg` | 3 major |  |  |  |  | misclass:壁类->天井, misclass:RC柱->壁类, missed_gt:壁类->壁类 |
| `天井_a-40347_03351.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类 |
| `天井_a-40307_03332.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:RC柱->壁类 |
| `天井_a-40254_03467.jpg` | 2 major |  |  |  |  | missed_gt:天井->天井, missed_gt:壁类->壁类 |
| `天井_a-40245_03190.jpg` | 2 major |  |  |  |  | misclass:RC柱->壁类, missed_gt:壁类->壁类 |
| `天井_a-40217_03341.jpg` | 2 major |  |  |  |  | misclass:RC柱->壁类, missed_gt:壁类->壁类 |
| `天井_a-20054_03400.jpg` | 2 major |  |  |  |  | missed_gt:天井->天井, missed_gt:RC柱->壁类 |
| `内壁_b-40571_03288.jpg` | 2 major |  |  |  |  | misclass:天井->壁类, missed_gt:壁类->壁类 |
| `内壁_b-40526_03463.jpg` | 2 major |  |  |  |  | missed_gt:天井->天井, missed_gt:壁类->壁类 |
| `内壁_b-40123_03435.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类 |
| `内壁_2-C-20098_03380.jpg` | 2 major |  |  |  |  | misclass:RC柱->壁类, missed_gt:壁类->壁类 |
| `RC柱_d-40079_03304.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类 |
| `RC柱_d-179_03178.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类 |
| `RC柱_d-144_03249.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类 |
| `RC壁_ｃ-10117_03195.jpg` | 2 major |  |  |  |  | misclass:RC柱->壁类, missed_gt:天井->天井 |
| `RC壁_c-40634_03446.jpg` | 2 major |  |  |  |  | missed_gt:天井->壁类, misclass:壁类->RC柱 |
| `RC壁_c-40544_03422.jpg` | 2 major |  |  |  |  | missed_gt:RC柱->壁类, missed_gt:壁类->壁类 |
| `RC壁_c-40503_03214.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类 |
| `RC壁_c-40471_03282.jpg` | 2 major |  |  |  |  | missed_gt:壁类->RC柱, missed_gt:RC柱->RC柱 |
| `RC壁_c-10058_03453.jpg` | 2 major |  |  |  |  | misclass:壁类->天井, missed_gt:壁类->天井 |
| `RC壁_3-D-40770_03225.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->壁类 |
| `RC壁_3-D-40769_03271.jpg` | 2 major |  |  |  |  | missed_gt:壁类->壁类, missed_gt:壁类->RC柱 |
| `天井_a-40259_03468.jpg` | 1 major |  |  |  |  | missed_gt:壁类->壁类 |
| `天井_a-40252_03441.jpg` | 1 major |  |  |  |  | misclass:天井->壁类 |
| `天井_a-40238_03482.jpg` | 1 major |  |  |  |  | missed_gt:壁类->壁类 |


## Files

- Full CSV: `/workspace/Shimizu-2026/docs/development_records/2026-05-26-yolo-fallback/router_hard_cases/v3_epoch14_test_hard_cases.csv`
- Prediction output: `/workspace/Shimizu-2026/outputs/router_eval/augv3_epoch14_test_saved_preds/labels`
- Evaluation log: `/workspace/Shimizu-2026/outputs/router_eval/augv3_epoch14_test_saved_preds.log`
