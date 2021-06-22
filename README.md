# A2C

This project is implementation of synchronous brother of Asynchronous algorithm A3C.
Improvement against policy alogirthms such as Actor critic or Reinforce is in multiple agents acting on copies of environment
in their own way(can have different experiences and different exploration policy). This reduces variance in updates but requires more computation power.

Paper on A3C is here https://arxiv.org/pdf/1602.01783.pdf A2C works exactly same way, but instead of doing everything asynchronously we take, all
the experiences put them into one batch and do one update (which is shown by experimentation to have little to no effect on results)

This Project was done in past therefor library versions are unknown (I will add all the version requirements after some experimentation)
