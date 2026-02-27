## CRL for Drone Racing with Random Obstacles

[![Conference](https://img.shields.io/badge/ICRA-2026-red.svg)](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra)
[![Simulator](https://img.shields.io/badge/Simulator-VisFly-brightgreen.svg)](https://github.com/SJTU-ViSYS-team/VisFly)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official implementation of the ICRA 2026 paper  
**“Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles”**.

---
### Video

- YouTube demo: [`https://youtu.be/_u7rpTWxA-I`](https://youtu.be/_u7rpTWxA-I)

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
which is a fast and versatile quadrotor simulator specialized for **vision-based flight**.

Concretely:

- We build our **racing in cluttered environments** by extending VisFly’s environment interfaces (`envs/`).
- We reuse and adapt VisFly’s **dynamics, randomization, and rendering pipeline**, including the `datasets/spy_datasets` format.

---

### Quick start - train and test a racing and obstacle-avoidance policy in simulation

1. Clone this repository:

```bash
git clone https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing.git
cd CRL-Drone-Racing
```

2. Set up the Conda environment (same dependency stack as **VisFly**):

```bash
conda env create -f environment.yml
conda activate visfly
```

3. **Prepare datasets and scenes**

   Place the simulator datasets and scene configs under:

   - `datasets/spy_datasets/configs/`

   You can use `examples/ete_racing_sim/run.py` as a reference and adjust the `scene_path` variable
   to point to your desired track configuration.

4. **Start training**

   ```bash
   python examples/ete_racing_sim/run.py -t 1
   ```

5. **Evaluate a trained model**

   ```bash
   python examples/ete_racing_sim/racing_demo.py -t 0 -w <saved_model_name>
   ```

### 
---

### Advanced Guide: Customizing or Replacing Simulation Tracks

1. **Create or Acquire Scene Models**

   - You can handcraft track scenes using [Blender](https://www.blender.org) or similar 3D modeling software, or directly download publicly available `.glb` models (ensure they include proper collision geometry, i.e., “physically inflated glb”). Place these files in `datasets/spy_datasets/self_define_objects/`.

2. **Add Scene JSON Configuration**

   - Refer to the file structure and examples used by [`VisFly`](https://github.com/SJTU-ViSYS-team/VisFly`) (especially how habitat-sim scene JSONs are organized). You’ll need to provide a corresponding `.json` configuration for each new scene config.

3. **Train or Evaluate with New Scenes**

   - In your training/testing script (e.g., `examples/ete_racing_sim/run.py`), update the `scene_path` variable to point to your custom scene configuration file.

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
@INPROCEEDINGS{11128458,
  author={Li, Fanxing and Sun, Fangyu and Zhang, Tianbao and Zou, Danping},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={VisFly: An Efficient and Versatile Simulator for Training Vision-Based Flight}, 
  year={2025},
  volume={},
  number={},
  pages={11325-11332},
  doi={10.1109/ICRA55743.2025.11128458}}
```
