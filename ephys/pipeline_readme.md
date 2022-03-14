Processing ephys data
1. Move glb_dmx to raw_ephys_data, extract pulses, ephys_pulses, wait for KS2
2. pipe_2_ks2_to_alf.py create alf folders
3. spike sort curation
4. pipe_4_relabel_and_metrics.py, corrects 3A files and homogenizes columns in metrics
5. pipe_5_patch_cluster_object.py, deals with split,merges, etc
6. pipe_6_getAlyxpenetration.py, get xyz sync_points
7. Use ephys aligment gui to align ang get channel locations
8. pipe_8_json_to_alf.py, transforms json channel locations to alf npys
9. pipe_9_full_bandit_fix.py, fixes choices
