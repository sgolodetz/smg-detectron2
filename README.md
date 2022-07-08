# smg-detectron2

This Python package provides a wrapper for Detectron2.

It is a submodule of [smglib](https://github.com/sgolodetz/smglib), the open-source Python framework associated with our drone research in the [Cyber-Physical Systems](https://www.cs.ox.ac.uk/activities/cyberphysical/) group at the University of Oxford.

### Development Installation (as part of smglib)

Note: Please read the [top-level README](https://github.com/sgolodetz/smglib/blob/master/README.md) for smglib before following these instructions.

1. Open the terminal.

2. Activate the Conda environment, e.g. `conda activate smglib`.

3. If you haven't already installed PyTorch, install it now. In our case, we did this via:

   ```
   pip install https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp37-cp37m-win_amd64.whl
   pip install https://download.pytorch.org/whl/torchaudio-0.9.1-cp37-cp37m-win_amd64.whl
   pip install https://download.pytorch.org/whl/cu111/torchvision-0.10.1%2Bcu111-cp37-cp37m-win_amd64.whl
   ```

   However, you may need a different version of PyTorch for your system, so change this as needed. (In particular, the latest version will generally be ok.)

4. Install `cudatoolkit`, e.g. via `conda install cudatoolkit==11.3.1` (the version you need may be different).

5. Install Detectron2 as per [here](https://ivanpp.cc/detectron2-walkthrough-windows). If you run into any trouble:

   i. If you get the error `ImportError: cannot import name '_nt_quote_args' from 'distutils.spawn'`, install `setuptools` version `59.6` or below. (See also [here](https://github.com/pytorch/pytorch/issues/70390).)

   ii. Try applying the fixes in `fix_torch_for_detectron2.sh`. (With newer versions of PyTorch, they may no longer be needed.)

6. Change to the `<root>/smg-detectron2` directory.

7. Check out the `master` branch.

8. Run `pip install -e .` at the terminal.

### Publications

If you build on this framework for your research, please cite the following paper:
```
@inproceedings{Golodetz2022TR,
author = {Stuart Golodetz and Madhu Vankadari* and Aluna Everitt* and Sangyun Shin* and Andrew Markham and Niki Trigoni},
title = {{Real-Time Hybrid Mapping of Populated Indoor Scenes using a Low-Cost Monocular UAV}},
booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
month = {October},
year = {2022}
}
```

### Acknowledgements

This work was supported by Amazon Web Services via the [Oxford-Singapore Human-Machine Collaboration Programme](https://www.mpls.ox.ac.uk/innovation-and-business-partnerships/human-machine-collaboration/human-machine-collaboration-programme-oxford-research-pillar), and by UKRI as part of the [ACE-OPS](https://gtr.ukri.org/projects?ref=EP%2FS030832%2F1) grant. We would also like to thank [Graham Taylor](https://www.biology.ox.ac.uk/people/professor-graham-taylor) for the use of the Wytham Flight Lab, [Philip Torr](https://eng.ox.ac.uk/people/philip-torr/) for the use of an Asus ZenFone AR, and [Tommaso Cavallari](https://uk.linkedin.com/in/tcavallari) for implementing TangoCapture.
