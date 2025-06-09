import fire
from nnsi.data import utm_data_generator as utm
from nnsi.data import utms as utms_lib
from nnsi.data import utm_data_generator as utm_dg_lib
import numpy as np

import jax

def go():
    rng = jax.random.PRNGKey(0)

    program_sampler = utms_lib.FastSampler(rng=rng)

    udg = utm.UTMDataGenerator(
      batch_size=1,
      seq_length=512,
      rng=np.random.default_rng(seed=0),
      utm=utms_lib.BrainPhoqueUTM(program_sampler),
      tokenizer=utm_dg_lib.Tokenizer.ASCII,
      memory_size=16000,
      maximum_steps=200_000,
      maximum_program_length=1_000)

    for _ in range(1000):
        program = udg.sample_params(1)
        res = udg.sample_from_params(program)

        print(res[1].argmax(axis=-1)[0, :30])


if __name__ == '__main__':
    fire.Fire(go)