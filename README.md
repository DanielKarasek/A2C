# A2C

This project is implementation of synchronous brother of Asynchronous algorithm A3C.
Improvement compared to policy alogirthms such as Actor critic or Reinforce is in multiple agents(instead of one) acting on copies of environment
in their own way(they have different experiences with similiar policy==better policy value estimation). This reduces variance in updates on the other hand this algorithm requires more computation power.

This is paper on A3C https://arxiv.org/pdf/1602.01783.pdf. A2C is same, but instead of doing everything asynchronously we take all
the experiences from all Agents and put them into one batch and do one update (which is shown by experimentation to have little to no effect on results compared to A3C)

This Project was done in past therefor library versions are unknown (I will add all the version requirements after some experimentation)
