# Physical AI OpenVLA Assignment: RaccoonBot Manipulation

## 1. Overview
This repository contains the modified client and server codes, environments, and logs for the Physical AI Assignment. The goal was to extend the basic VLA manipulation tasks (Dataset Extension) and optimize the robot's motion execution pipeline (Code Improvement) using the 7B OpenVLA model in a MuJoCo simulation.

## 2. Dataset Extension & RLDS Rebuilding
- **New Objects & Tasks:** Expanded the environment to include 6 varied objects (`red_cube`, `blue_sphere`, cylinders). Added `push` and `lift` tasks with diverse natural language instructions.
- **Dataset Generation:** Modified the `convert_raw_to_openvla_rlds_intermediate` script. The `collect_dataset` main loop was updated to dynamically balance and randomize object positions, colors, and tasks until reaching the target of 400 episodes (including 300 newly fine-tuned episodes).

## 3. Code Improvement
- **Speed Up Inference & Motion:** Addressed the bottleneck of conservative VLA action predictions by introducing the `--delta_scale` (action multiplier) and minimizing `--settle_seconds_per_action` to achieve fluid, continuous motion.
- **Fail-Fast & Auto-Termination:** Implemented real-time physical state tracking (e.g., Z-axis height for lifting, XY-displacement for pushing). The script now automatically terminates with a `[TASK SUCCESS]` log when the objective is met, preventing infinite loops.

## 4. How to Run
Ensure the server is running, then execute the optimized client command:
```bash
python src/openvla_multicolor_client.py \
  --server_url [http://127.0.0.1:8000](http://127.0.0.1:8000) \
  --xml_path src/Raccoon_colored_cylinder.xml \
  --target_color blue_sphere \
  --task_type push \
  --speed 100 \
  --settle_seconds_per_action 0.4 \
  --max_delta_xyz 0.012 \
  --delta_scale 1.5 \
  --min_object_distance 0.05 \
  --object_x_range -0.13 0.13 \
  --use_viewer