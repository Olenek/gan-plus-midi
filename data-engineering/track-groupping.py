import os
import shutil

import pandas as pd

df = pd.read_csv('vgmidi-data/vgmidi_labelled.csv')


def copy_to_new_location(row):
    val_arous_to_dir = {
        (1, 1): 'pleasant-excited/',
        (1, -1): 'pleasant-unexcited/',
        (-1, 1): 'unpleasant-excited/',
        (-1, -1): 'unpleasant-unexcited/',
    }

    general_dir = 'midi-files'
    for key, value in val_arous_to_dir.items():
        os.makedirs(os.path.join(general_dir, value), exist_ok=True)

    source = os.path.join('vgmidi', row.midi)
    dest = os.path.join(general_dir, val_arous_to_dir.get((row.valence, row.arousal))) + row.midi.split('/')[-1]

    shutil.copyfile(source, dest)
    return dest


df.apply(lambda x: copy_to_new_location(x), axis=1)
