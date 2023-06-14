"""Script for downloading the Calcium imaging data from the Allen Brain Observatory's visual coding dataset,
only the drifting grating experiments."""

import requests
import os

from absl import app, flags
from tqdm import tqdm
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info


FLAGS = flags.FLAGS
flags.DEFINE_string('root', './data', 'Root directory for saving data.')


def main(argv):
    boc = BrainObservatoryCache(manifest_file=os.path.join(FLAGS.root, 'manifest.json'))

    # get metadata for all drifting grating experiments
    exps = boc.get_ophys_experiments(stimuli=[stim_info.DRIFTING_GRATINGS])

    # download data for each experiment
    for exp in tqdm(exps):
        success = False
        while not success:
            try:
                exp_id = exp['id']
                exp = boc.get_ophys_experiment_data(exp_id)
                events = boc.get_ophys_experiment_events(exp_id)
                success = True
            except requests.ConnectionError:
                continue



if __name__ == '__main__':
    app.run(main)