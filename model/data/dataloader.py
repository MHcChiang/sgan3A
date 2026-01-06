import os, random, numpy as np, copy

from .nuscenes_pred_split import get_nuscenes_pred_split
from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split

# Try to import print_log from utils
try:
    from utils.utils import print_log
except ImportError:
    import logging
    def print_log(msg, log=None):
        if log is None:
            log = logging.getLogger(__name__)
        log.info(msg)


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.frame_skip
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred           
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy            
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()


class data_loader(data_generator):
    """
    New-added class to apply coordinate transformation (scene-centered coordinate system).
    
    This class inherits from data_generator and adds the capability to center trajectories
    around the scene center (mean position of all agents at the last observed timestep).
    This transformation is applied at the dataloader level and cached to save compute resources
    during training. Once a sample is centered, it is cached so subsequent accesses reuse the
    cached version instead of recomputing the transformation.
    """
    
    def __init__(self, parser, log, split='train', phase='training', centered=True):
        """
        Initialize the data loader with optional scene-centered coordinate transformation.
        
        Args:
            parser: Configuration parser object
            log: Logger object
            split: Data split ('train', 'val', 'test')
            phase: Phase ('training' or 'testing')
            centered: If True, apply scene-centered coordinate transformation (default: True)
        """
        super().__init__(parser, log, split=split, phase=phase)
        self.centered = centered
        # Cache for centered trajectories: key is sample_index, value is centered data dict
        self._centered_cache = {}
    
    def _apply_scene_centering(self, data):
        """
        Apply scene-centered coordinate transformation to the data.
        
        The scene center is computed as the mean position of all agents at the last
        observed timestep (t=-1 of pre_motion). This center is then subtracted from
        all trajectory points in both past and future motions.
        
        Args:
            data: Dictionary containing 'pre_motion_3D' and 'fut_motion_3D' lists
            
        Returns:
            Modified data dictionary with centered trajectories
        """
        if not self.centered:
            return data
        
        pre_motion_3D = data['pre_motion_3D']
        fut_motion_3D = data['fut_motion_3D']
        
        if len(pre_motion_3D) == 0:
            return data
        
        # Stack all agent trajectories: [Time, Agents, 2]
        # Each element in pre_motion_3D is [Time, 2], so we stack them along a new axis
        stacked_pre_motion = np.stack(pre_motion_3D, axis=1)  # [Time, Agents, 2]
        
        # Get the last timestep positions: [Agents, 2]
        current_pos_t0 = stacked_pre_motion[-1]
        
        # Compute scene center: mean across agents -> [2]
        scene_center = np.mean(current_pos_t0, axis=0, keepdims=True)  # [1, 2]
        
        # Subtract center from each agent's trajectory
        centered_pre_motion_3D = []
        for agent_motion in pre_motion_3D:
            # agent_motion is [Time, 2], scene_center is [1, 2]
            centered_motion = agent_motion - scene_center
            centered_pre_motion_3D.append(centered_motion)
        
        centered_fut_motion_3D = []
        for agent_motion in fut_motion_3D:
            centered_motion = agent_motion - scene_center
            centered_fut_motion_3D.append(centered_motion)
        
        # Update data dictionary
        data['pre_motion_3D'] = centered_pre_motion_3D
        data['fut_motion_3D'] = centered_fut_motion_3D
        
        return data
    
    def shuffle(self):
        """
        Shuffle the sample list and clear the cache since sample order changes.
        """
        super().shuffle()
        # Clear cache when shuffling to ensure consistency
        self._centered_cache.clear()
    
    def next_sample(self):
        """
        Get next sample and apply scene centering if enabled.
        Uses caching to avoid recomputing the transformation for the same sample.
        
        Returns:
            Data dictionary with optionally centered trajectories
        """
        # Get the sample index before incrementing (used as cache key)
        sample_index = self.sample_list[self.index]
        
        # Check cache first if centering is enabled
        if self.centered and sample_index in self._centered_cache:
            self.index += 1
            # Return a deep copy of cached data to avoid modifying the cache
            return copy.deepcopy(self._centered_cache[sample_index])
        
        # Get raw data from parent class
        data = super().next_sample()
        
        if data is None:
            return None
        
        # Apply scene centering transformation
        if self.centered:
            data = self._apply_scene_centering(data)
            # Cache the centered data using sample_index as key
            self._centered_cache[sample_index] = copy.deepcopy(data)
        
        return data
    
    def __call__(self):
        return self.next_sample()
