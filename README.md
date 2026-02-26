## CRL for Drone Racing with Random Obstacles

[![Conference](https://img.shields.io/badge/ICRA-2026-red.svg)](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra)
[![Simulator](https://img.shields.io/badge/Simulator-VisFly-brightgreen.svg)](https://github.com/SJTU-ViSYS-team/VisFly)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official implementation of the ICRA 2026 paper  
**“Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles”**,  
built on top of the **VisFly** simulator [`SJTU-ViSYS-team/VisFly`](https://github.com/SJTU-ViSYS-team/VisFly).

---

### Video

- YouTube demo: [`https://youtu.be/_u7rpTWxA-I`](https://youtu.be/_u7rpTWxA-I)
- Clickable thumbnail:

[![Video Demo](https://img.youtube.com/vi/_u7rpTWxA-I/hqdefault.jpg)](https://youtu.be/_u7rpTWxA-I)

---

### Abstract

```text
Autonomous drone racing has attracted increasing interest as a research topic for exploring the limits of agile flight.
However, existing studies primarily focus on obstacle-free racetracks, while the perception and dynamic challenges
introduced by obstacles remain underexplored, often resulting in low success rates and limited robustness in real-world
flight. To this end, we propose a novel vision-based curriculum reinforcement learning framework for training a robust
controller capable of addressing unseen obstacles in drone racing. We combine multi-stage curriculum learning, domain
randomization, and a multi-scene updating strategy to address the conflicting challenges of obstacle avoidance and gate
traversal. Our end-to-end control policy is implemented as a single network, allowing high-speed flight of quadrotors
in environments with variable obstacles. Both hardware-in-the-loop and real-world experiments demonstrate that our
method achieves faster lap times and higher success rates than existing approaches, effectively advancing drone racing
in obstacle-rich environments.
```

---

### Built on VisFly simulator

This project is developed **on top of the VisFly simulator** [`SJTU-ViSYS-team/VisFly`](https://github.com/SJTU-ViSYS-team/VisFly),  
which is a fast and versatile quadrotor simulator specialized for **vision-based flight** using differentiable
simulation (see the VisFly paper for details).

Concretely:

- We build our **racing and waypoint environments** by extending VisFly’s environment interfaces (`envs/`).
- We reuse and adapt VisFly’s **dynamics, randomization, and rendering pipeline**, including the
  `datasets/spy_datasets` format.
- We add new **curriculum reinforcement learning pipelines** and task-specific environments for quadrotor racing
  with random obstacles.

For more information about the underlying simulator, please refer to the VisFly repository  
[`SJTU-ViSYS-team/VisFly`](https://github.com/SJTU-ViSYS-team/VisFly) and its README.

---

### Repository structure

- `envs/` – quadrotor racing and waypoint environments  
  - e.g. `demo3_ellipse_onboard.RacingEnv` for onboard depth-based ellipse track racing.
- `utils/` – algorithms (PPO, SHAC, BPTT), policy networks, data handling, logging, plotting.
- `configs/` – quadrotor dynamics and controller parameters (e.g. `example_offboard.json`).
- `examples/ete_racing_sim/` – end-to-end racing training and evaluation scripts in simulation.
- `examples/ete_racing_real/` – ROS bag processing, dynamics fitting, and sim-to-real deployment tools.

Large datasets, trained weights and some auxiliary examples are intentionally kept local (see notes below)  
to keep the public repository compact.

---

### Installation

We recommend using **Conda** to manage dependencies.

```bash
conda env create -f environment.yml
conda activate vision-racing
```

High-level requirements:

- Python 3.x  
- PyTorch with CUDA (for GPU training)  
- Habitat-Sim / rendering dependencies (same family as VisFly)  

Please refer to `environment.yml` for exact package versions.  
For more details about Habitat-Sim and dataset preparation, you may also refer to
[`SJTU-ViSYS-team/VisFly`](https://github.com/SJTU-ViSYS-team/VisFly).

---

### Quick start – train a racing policy in simulation

1. **Prepare datasets and scenes**

   Place the simulator datasets and scene configs under:

   - `datasets/spy_datasets/configs/`

   You can use `examples/ete_racing_sim/racing_demo.py` as a reference and adjust the `scene_path` variable
   to point to your desired track configuration.

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

   where `<saved_model_name>` is the model file name saved under the `saved/` folder (without extension).

---

### Real-world experiments and sim-to-real

Scripts in `examples/ete_racing_real/` provide utilities for:

- Converting ROS bag data to depth images and state logs.  
- Fitting dynamics models from real-flight data.  
- Deploying trained policies in HITL / real-world experiments.

These tools are **optional** and are not required if you only need simulation training,  
but are helpful for reproducing sim-to-real results.

---

### Citation

If you find this repository useful in your research, please cite the corresponding ICRA 2026 paper  
(update the BibTeX with the final version when available):

```text
@inproceedings{crl_drone_racing_icra2026,
  title     = {Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles},
  author    = {Authors},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

Since this project is built on **VisFly**, please also cite the VisFly simulator:

```text
@misc{li2024visfly,
  title         = {VisFly: An Efficient and Versatile Simulator for Training Vision-based Flight},
  author        = {Fanxing Li and Fangyu Sun and Tianbao Zhang and Danping Zou},
  year          = {2024},
  eprint        = {2407.14783},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2407.14783}
}
```

---

### License

This project is released under the **MIT License**. See the `LICENSE` file for details.  
Please also follow the licenses of any third-party dependencies (including VisFly and Habitat-Sim).

