---
license: cc-by-4.0
---

# What is ReplicaCAD?

[**Vist the ReplicaCAD Homepage**](https://aihabitat.org/datasets/replica_cad/)

The ReplicaCAD dataset is an artist recreation of the scanned “FRL apartment” variations from the Replica dataset. 
This dataset is intended for use in the Habitat simulator for embodied in-home interaction tasks such as object re-arrangement.

All materials are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) Public License](https://creativecommons.org/licenses/by/4.0/).

## Dataset Contents:

We provide two dataset downloads with different properties: one suited to interactive simulation and the other for photorealistic visualization.

Note: Both downloadable datasets contain 84 of the 105 variations described in the paper and shown in the video with the remaining 21 scenes (1 macro variation and associated 20 micro variations) withheld as a test set for challenge evaluation.

### ReplicaCAD Interactive (this repository): 
[132MB]

Intended for use with a PBR shader. Contains 1 empty scene and 6 re-creations of the scanned “FRL apartment” variations staged with both large furniture and small objects and ready for dynamic simulation in Habitat-sim. Also included are 84 (of 105) artist authored re-arrangements of large furniture (fully static placements except articulations) organized into 5 macro variations (as different tenants may organize the same apartment) each with an additional 20 micro variations (with a few pieces of furniture moved/swapped).

- 90+ 3D object assets with convex collision geometry and physical properties (mass, friction, restitution) as well as receptacle metadata for use generating object clutter (e.g. for rearrangement tasks).
- 6 stage (i.e., static background) assets emptied of all but architectural features (1 each for FRL apartment and the 5 macro variations).
- 6+ URDF assets defining articulated furniture and door properties including receptacle metadata for generating object clutter.
- 1 SceneDataset configuration file which aggregates all config and asset paths for one-line import in Habitat.
- .navmesh files (in navmeshes/ directory) for every scene computed for an agent with 0.3m radius (e.g. appropriate for a Fetch robot base) and additional .navmesh files (in navmeshes_default/ directory) computed with Habitat default agent parameters for optional use.
- 84 + 6 SceneDataset configuration files defining object metadata and scene layouts for easy use in the Habitat simulator referencing the Fetch tuned NavMeshes.

### ReplicaCAD with baked lighting:

[Get ReplicaCAD with baked lighting here](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_baked_lighting) [525MB]

Contains the same 84 (of 105) artist authored re-arrangements of large furniture described in ReplicaCAD Interactive with synthetic global illumination baked into the textures for more photo-realistic visualization. All articulated furniture is included with baked lighting textures, but all other furniture is static.

---

Citing ReplicaCAD
---

Using ReplicaCAD in your research? Please cite the following paper: [arxiv](https://arxiv.org/abs/2106.14405)

```
@inproceedings{szot2021habitat,
    title     =     {Habitat 2.0: Training Home Assistants to Rearrange their Habitat},
    author    =     {Andrew Szot and Alex Clegg and Eric Undersander and Erik Wijmans and Yili Zhao and John Turner and Noah Maestre and Mustafa Mukadam and Devendra Chaplot and Oleksandr Maksymets and Aaron Gokaslan and Vladimir Vondrus and Sameer Dharur and Franziska Meier and Wojciech Galuba and Angel Chang and Zsolt Kira and Vladlen Koltun and Jitendra Malik and Manolis Savva and Dhruv Batra},
    booktitle   =     {Advances in Neural Information Processing Systems (NeurIPS)},
    year      =     {2021}
}
```