import fire
from nnsi.data import utm_data_generator as utm
from nnsi.data import utms as utms_lib
from nnsi.data import utm_data_generator as utm_dg_lib
import numpy as np
import random

import jax

NUM_TOKENS = 256

def go():
    rng = jax.random.PRNGKey(0)

    program_sampler = utms_lib.FastSampler(rng=rng)

    udg = utm.UTMDataGenerator(
      batch_size=32,
      seq_length=512,
      rng=np.random.default_rng(seed=0),
      utm=utms_lib.BrainPhoqueUTM(
        program_sampler,
        alphabet_size=128),
      tokenizer=utm_dg_lib.Tokenizer.ASCII,
      memory_size=10_000,
      maximum_steps=200_000,
      maximum_program_length=1000,
      rep_pad=True)

    for _ in range(10):

        # map = list(range(NUM_TOKENS))
        # random.shuffle(map)
        # map = {fr: to for fr, to in enumerate(map)}

        # program = udg.sample_params(1)
        # res = udg.sample_from_params(program)

        res = udg.sample()
        # print(res[0].shape)

        seq = res[0].argmax(axis=-1)
        # print(seq.shape)
        # print(seq.max())
        # seq = [map[i] for i in seq.tolist()]

        print(seq[0, :60])

if __name__ == '__main__':
    fire.Fire(go)