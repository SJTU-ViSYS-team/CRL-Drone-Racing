## CRL for Drone Racing with Random Obstacles

[![Conference](https://img.shields.io/badge/ICRA-2026-red.svg)](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra)
[![Simulator](https://img.shields.io/badge/Simulator-VisFly-brightgreen.svg)](https://github.com/SJTU-ViSYS-team/VisFly)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

### Demo video

- YouTube demo: [`https://youtu.be/_u7rpTWxA-I`](https://youtu.be/_u7rpTWxA-I)
[![Video Demo](https://img.youtube.com/vi/_u7rpTWxA-I/hqdefault.jpg)](https://youtu.be/_u7rpTWxA-I)

---

### Abstract

```text
Autonomous drone racing has attracted increasing interest as a research topic for exploring the limits of agile flight. However, existing studies primarily focus on obstacle-free racetracks, while the perception and dynamic challenges introduced by obstacles remain underexplored, often resulting in low success rates and limited robustness in real-world flight. To this end, we propose a novel vision-based curriculum reinforcement learning framework for training a robust controller capable of addressing unseen obstacles in drone racing. We combine multi-stage cu
rriculum learning, domain randomization, and a multi-scene updating strategy to address the conflicting challenges of obstacle avoidance and gate traversal. Our end-to-end control policy is implemented as a single network, allowing high-speed flight of quadrotors in environments with variable obstacles. Both hardware-in-the-loop and real-world experiments demonstrate that our method achieves faster lap times and higher success rates than existing approaches, effectively advancing drone racing in obstacle-rich environments. The video and code are available at: \url{https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing}.
```

---

### Repository structure

- `envs/` – quadrotor racing and waypoint environments  
  - e.g. `demo3_ellipse_onboard.RacingEnv` for onboard depth-based ellipse track racing.
- `utils/` – algorithms (PPO, SHAC, BPTT), policy networks, data handling, logging, plotting.
- `configs/` – quadrotor dynamics and controller parameters (e.g. `example_offboard.json`).
- `examples/ete_racing_sim/` – end-to-end racing training and evaluation scripts in simulation.
- `examples/ete_racing_real/` – ROS bag processing, dynamics fitting, and sim-to-real deployment tools.

Large datasets, trained weights and some auxiliary examples are intentionally kept local (see notes below) to keep the public repository compact.

---

### Installation

We recommend using **Conda** to manage dependencies.

```bash
conda env create -f environment.yml
conda activate vision-racing
```

Requirements (high-level):

- Python 3.x
- PyTorch with CUDA (for GPU training)
- Habitat-Sim / rendering dependencies (as required by your environment setup)

Please refer to `environment.yml` for the exact package versions.

---

### Quick start (English) – train a racing policy in simulation

1. **Prepare datasets and scenes**

   Place the simulator datasets and scene configs under:

   - `datasets/spy_datasets/configs/`

   You can use `examples/ete_racing_sim/racing_demo.py` as a reference and adjust the `scene_path` variable to point to your desired track configuration.

2. **Start training**

   From the repository root:

   ```bash
   python examples/ete_racing_sim/racing_demo.py --train --comment demo1_straight
   ```

   This will:

   - Instantiate a racing environment (e.g. `envs.demo1_straight.RacingEnv2`).  
   - Train a PPO policy with multi-modal observations (state, depth, gate index).  
   - Save logs and checkpoints in `examples/ete_racing_sim/saved/`.

3. **Evaluate a trained model**

   ```bash
   python examples/ete_racing_sim/racing_demo.py --train 0 --weight <saved_model_name>
   ```

   where `<saved_model_name>` is the model f### 快速上手（中文）——在仿真中训练端到端赛车策略

**1. 环境准备**

```bash
conda env create -f environment.yml
conda activate vision-racing
```

- 请确保本机有可用的 GPU + CUDA 环境，用于加速训练。
- 将仿真场景配置放在 `datasets/spy_datasets/configs/` 目录下，可参考  
  `examples/ete_racing_sim/racing_demo.py` 中的 `scene_path` 设置。ile name saved under the `saved/` folder (without extension).


### Real-world experiments and sim-to-real

Scripts in `examples/ete_racing_real/` provide utilities for:

- Converting ROS bag data to depth images and state logs.  
- Fitting dynamics models from real-flight data.  
- Deploying trained policies in HITL / real-world experiments.

These tools are **optional** and are not required if you only need simulation training, but are helpful for reproducing sim-to-real results.

---

### Notes on repository size and assets

To keep the GitHub repository lightweight and easy to clone:

- Large data files are ignored via `.gitignore`, including:
  - `.bag`, `.zip`, `.onnx`, `.pt`
  - common image and video formats (`.png`, `.jpg`, `.avi`, `.mp4`, etc.)
- Some example folders are **only used locally** and not tracked by Git:
  - `examples/diff_baseline/`
  - `examples/multi_task_IL/`
  - `examples/sim_to_sim/`
  - `examples/VLA_task/`

If you need the full datasets, trained weights or additional examples, please generate them locally or contact the authors.

---

### Citation

If you find this repository useful in your research, please cite the corresponding ICRA 2026 paper (arxiv now – update with the final entry when available):

```text
@inproceedings{visionracing_icra2026,
  title     = {Vision-based Quadrotor Racing with Depth Perception and Reinforcement Learning},
  author    = {Authors},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

---

### License

This code is released for **research purposes only**.  
For other uses, please contact the authors of the associated ICRA 2026 paper.
