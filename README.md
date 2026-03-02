## CRL for Drone Racing with Random Obstacles

[![Conference](https://img.shields.io/badge/ICRA-2026-red.svg)](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra)
[![Simulator](https://img.shields.io/badge/Simulator-VisFly-brightgreen.svg)](https://github.com/SJTU-ViSYS-team/VisFly)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-E95420?logo=ubuntu)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)

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
- We reuse and adapt VisFly’s **dynamics, randomization, rendering pipeline, and the datasets format**.

For more information about the underlying simulator, please refer to the VisFly repository  
[`SJTU-ViSYS-team/VisFly`](https://github.com/SJTU-ViSYS-team/VisFly) and its README.

---


### Quick start - train and test in simulation

1. Clone this repository:

```bash
git clone https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing.git
cd CRL-Drone-Racing
```

2. Create Conda environment:

```bash
# Assuming you have already set up the visfly environment.
conda activate visfly
```

3. Start training:

```bash
python examples/ete_racing_sim/run.py -t 1
```

4. Evaluate a trained model:

```bash
python examples/ete_racing_sim/run.py -t 0 -w <saved_model_name>
```

---

### Advanced guide: customizing simulation tracks

1. Create or download `.glb` track assets and place them in `datasets/spy_datasets/self_define_objects/`.
2. Add matching scene JSON configs following the VisFly scene format.
3. Update `scene_path` in `examples/ete_racing_sim/run.py` to use your custom track.

---

### Citation

If you find this repository useful in your research, please cite the corresponding ICRA 2026 paper  
(update the BibTeX with the final version when available):

```text
@misc{sun2026curriculumreinforcementlearningquadrotor,
      title={Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles}, 
      author={Fangyu Sun and Fanxing Li and Yu Hu and Linzuo Zhang and Yueqian Liu and Wenxian Yu and Danping Zou},
      year={2026},
      eprint={2602.24030},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.24030}, 
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
