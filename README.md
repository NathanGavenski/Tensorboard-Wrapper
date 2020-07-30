# Tensorboard

This is a wrapper class for the Tensorflow's Tensorboard application. It is still in early development, and should be used with a grain of salt.
Tensorboard.py is the Wrapper that contains all functions needed to interact, more documentation should come in the future, but as for now, the requirements for this project are:

```
numpy
tensorboard==1.15
torch
```




There are no installs for the moments to be used with any other repositories.

Todo List:
- [ ] Unit Tests for the functions available
- [ ] Documentation
- [ ] pip install module
- [ ] Implement rest of the Tensorboard functions

A list of the functions already implemented:
- [x] tebsirbiard.writer.SummaryWritter
- [x] add_scalar
- [x] add_scalar*
- [x] add_histogram*
- [x] add_image
- [x] add_images*
- [ ] add_figure
- [ ] add_video
- [ ] add_audio
- [ ] add_text
- [x] add_graph
- [ ] add_embedding
- [ ] add_pr_curve
- [ ] add_custom_scalars
- [ ] add_mesh
- [x] add_hparams
- [x] flush*
- [x] close*

*Altered in some shape or form. All changes will be in the documentation in the future.
