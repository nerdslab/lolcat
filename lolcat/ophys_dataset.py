import os

import numpy as np
try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    import allensdk.brain_observatory.stimulus_info as stim_info
    allen_sdk_installed = True
except ImportError:
    allen_sdk_installed = False
import torch
from torch_geometric.data import Data

from . import InMemoryDataset


StimulusType = ['drifting_gratings', 'natural_movie_one', 'natural_movie_three', 'spontaneous']
OPhys4CellType = ['Pvalb', 'Vip', 'Sst', 'E']
# OPhysCellType = ['Cux2', 'Emx1', 'Fezf2', 'Nr5a1', 'Ntsr1', 'Pvalb', 'Rbp4',
#                  'Rorb', 'Scnn1a', 'Slc17a7', 'Sst', 'Tlx3', 'Vip']


class OPhysDataset(InMemoryDataset):
    r"""A dataset for the Allen Brain Observatory's visual coding data. 

    Args:
        root (string): Root directory where the dataset should be saved.
        stimuli (string, list of strings): Stimuli to include in the dataset.
        concat (bool, optional): If set to :obj:`True`, the data object will contain the concatenated trials of all 
            stimuli, otherwise each sample can only have the trials from the same stimulus condition. (default: 
            :obj:`False`)
        session_type (string, optional): The session type to include in the dataset. (default: :obj:`three_session_A`)
        transform (callable, optional): A function/transform that takes in a data sample and returns a transformed 
            version. This function will be applied each time the __getitem__ method is called. (default: None)
        force_process (bool, optional): If True, the dataset will be processed even if a processed version already
            exists. (default: False)
    """

    dt = 0.032303411814239395  # sampling period in seconds
    class_names = OPhys4CellType

    def __init__(self, root, stimuli, concat=False, session_type='three_session_A', transform=None, 
                 force_process=False):
        
        if not allen_sdk_installed:
            raise ImportError('The AllenSDK is not installed. Please install it using `pip install allensdk`.')

        # load raw data using the allensdk, and get the list of experiments
        self.boc = BrainObservatoryCache(manifest_file=os.path.join(root, 'manifest.json'))
        self.experiments = self.boc.get_ophys_experiments(session_types=[session_type])

        # make sure stimuli is a list, if not, convert it to a list of length 1
        if not isinstance(stimuli, list):
            stimuli = [stimuli]
        self.stimuli = stimuli
        self.concat = concat

        trial_length = 3. # in seconds
        self.window_size = int(trial_length / self.dt) - 1  # should be 91

        name = '_'.join(self.stimuli)
        if self.concat:
            name += '_concat'

        super().__init__(root, name, transform, force_process)

    def process(self):
        cell_id_list = []
        cell_list = []

        for experiment in self.experiments:
            experiment_id = experiment['id']
            exp = self.boc.get_ophys_experiment_data(experiment_id)
            events = self.boc.get_ophys_experiment_events(experiment_id)

            # get cell ids
            cell_ids = exp.get_cell_specimen_ids()

            # get available stimuli
            stimuli_list = exp.list_stimuli()

            # collect the data for stimuli of interest
            data_list = []
            stimulus_id_list = []
            for stimulus in self.stimuli:
                if stimulus in stimuli_list:
                    table = exp.get_stimulus_table(stimulus)
                    stimulus_id = StimulusType.index(stimulus)
                    if stimulus in [stim_info.DRIFTING_GRATINGS]:
                        data = self._extract_drifting_gratings(events, table)
                    elif stimulus in [stim_info.NATURAL_MOVIE_THREE, stim_info.NATURAL_MOVIE_ONE]:
                        data = self._extract_natural_movies(events, table)
                    elif stimulus in [stim_info.SPONTANEOUS_ACTIVITY]:
                        data = self._extract_sponatneous(events, table)
                    else:
                        raise NotImplementedError

                    # data of shape (number of cells, number of trials, time)
                    data_list.append(data)
                    stimulus_id_list.append(stimulus_id)

            if self.concat:
                stimulus_id_list = [len(StimulusType)]
                data = np.concatenate(data_list, axis=1)
                data_list = [data]

            for data, stimulus_id in zip(data_list, stimulus_id_list):
                # compute isi distribution
                isi = self._compute_isi_distribtuion(data)

                cre_line = experiment['cre_line']
                cell_type = cre_line.split('-')[0]
                cell_type = cell_type if cell_type in ['Pvalb', 'Sst', 'Vip'] else 'E'
                y = OPhys4CellType.index(cell_type)

                for i in range(data.shape[0]):
                    cell_data = Data(
                        x=torch.FloatTensor(isi[i]),
                        y=torch.tensor(y),
                        stimulus=torch.tensor(stimulus_id),
                        experiment_id=experiment_id,
                        cell_id=cell_ids[i],
                    )
                    cell_list.append(cell_data)

                cell_id_list.append(cell_ids)
        cell_ids = np.concatenate(cell_id_list)
        return dict(data_list=cell_list, cell_ids=cell_ids)


    #########################################################
    # Methods for processing data based on the stimuli type #
    #########################################################
    def _extract_drifting_gratings(self, events, table):
        r"""Extracts the data for the drifting gratings stimulus. 
        
        .. note::
        - Blank sweeps are skipped.
        - We use 3 seconds of data for each trial. The first second is the baseline, and the next 2 seconds are 
        the stimulus.
        """
        data_list = []

        for i, row in table.iterrows():
            # skip if blank sweep
            if row['blank_sweep'] == 1.:
                continue
            start, end = int(row['start']), int(row['end'])
            # add 1 sec before
            end = start + self.window_size
            data_list.append(events[:, start:end])

        data = np.stack(data_list, axis=1)
        return data

    def _extract_natural_movies(self, events, table):
        r"""Extracts the data for the natural movies stimulus.
        
        .. note::
        - The movies last longer than 3 seconds, and can be repeated multiple times, we slice the data into multiple 
            trials of 3 seconds.
        """
        data_list = []

        epoch_start = table.groupby('repeat')['start'].min()
        epoch_end = table.groupby('repeat')['end'].max()
        for i in range(len(epoch_start)):
            slices = np.arange(epoch_start[i], epoch_end[i], self.window_size).astype(int)
            for j in range(len(slices)-1):
                start, end = slices[j], slices[j+1]
                data_list.append(events[:, start:end])

        data = np.stack(data_list, axis=1)
        return data

    def _extract_sponatneous(self, events, table):
        r"""Extracts the data for the spontaneous activity stimulus.
        
        .. note::
        - There can be one or multiple blocks of spontaneous activity that last longer than 3 seconds, we slice the data 
            into multiple trials of 3 seconds."""
        data_list = []

        for i, row in table.iterrows():
            slices = np.arange(row['start'], row['end'], self.window_size).astype(int)
            for j in range(len(slices)-1):
                start, end = slices[j], slices[j+1]
                data_list.append(events[:, start:end])

        data = np.stack(data_list, axis=1)
        return data

    ##############################################
    # Methods for computing ISI and firing rates #
    ##############################################
    def _compute_isi_distribtuion(self, data):
        r"""Computes the ISI distribution for each neuron in the dataset."""
        isi_matrix = np.zeros_like(data)
        out = np.nonzero(data)
        trial_index = out[0] * data.shape[1] + out[1]
        time_index = trial_index * 2 * data.shape[2] + out[2]  # add a gap bigger than the duration of the trial

        isi = np.concatenate([np.zeros(1, dtype=np.int64), time_index[1:] - time_index[:-1]])
        isi[isi >= data.shape[2]] = 0
        np.add.at(isi_matrix, (out[0], out[1], isi), np.ones_like(isi))
        isi_matrix = isi_matrix[..., 1:]
        return isi_matrix
    
    def _compute_firing_rate(self, data):
        raise NotImplementedError

    #############
    # Utilities #
    #############
    def filter_data(self, data, thresh):
        r"""Drops cells that have less than `thresh` spikes across all the trials."""
        cond = (data.x.sum(dim=1) >= thresh).sum() > 0
        return cond

    def get_split_indices(self, split_id, thresh=5., good_cells_only=True):
        # load assignment masks from split file
        split_filename = os.path.join(self.root, 'calcium_splits.npy')
        assign_mask = np.load(split_filename, allow_pickle=True).item()

        if self.concat:
            good_cells_only = False

        if good_cells_only:
            good_cells_filename = os.path.join(self.root, 'good_cells.npy')
            good_cells = np.load(good_cells_filename, allow_pickle=True).item()

        train_val_test_indices = [[], [], []]
        for i, data in enumerate(self):
            cell_id = data.cell_id
            if cell_id in assign_mask:
                stimulus_id = data.stimulus.item()
                stimulus = StimulusType[stimulus_id]
                if good_cells_only and stimulus in good_cells:
                    keep = cell_id in good_cells[stimulus]
                elif stimulus not in good_cells:
                    keep = True
                else:
                    keep = False
                if keep and self.filter_data(data, thresh=thresh):
                    assign_id = assign_mask[cell_id][split_id]
                    train_val_test_indices[assign_id].append(i)
        return map(torch.tensor, train_val_test_indices)

    def get_labels(self, indices):
        target = []
        for i in indices:
            j = 0 if self[i].stimulus == 0 else 1
            target.append(self[i].y + 4 * j)
        return torch.tensor(target)
