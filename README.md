# place-pulse-dataset
Hi, I'm Aleksandrs Koselevs and this is a PyTorch dataset class for the Place Pulse 2.0 dataset [1]

This implementation returns the TrueSkill score for a given image.

Other PlacePulse 2.0 dataset implementations: 

https://github.com/rsangers/explainable_perception

Note:

The scores in qscores.tsv come from synthetic comparisons made by the model from [1] and for most use cases would not make a good target label. Feel free to replace them with another metric, e.g. total positive votes for an image.

[1] A. Dubey, N. Naik, D. Parikh, R. Raskar, and C. A. Hidalgo, “Deep Learning the City : Quantifying Urban Perception At A Global Scale.” arXiv, Sep. 12, 2016. Accessed: Apr. 26, 2023. [Online]. Available: http://arxiv.org/abs/1608.01769
