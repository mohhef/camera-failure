"""
This is the file for RAFT optical flow.
Note we aren't using it right now and isn't maintained.
Shall be updated included all bugfixes when needed.
"""

import sys
sys.path.append('RAFT/core')
from raft import RAFT

class Args:
    def __init__(self, model=None, path=None, small=False, mixed_precision=False, alternate_corr=False):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
    
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
            
# Instantiate with default values
# RAFT pre-trained on kitti dataset!
args = Args(
    model='RAFT/raft-kitti.pth',
)

# Load the model 
of_model = torch.nn.DataParallel(RAFT(args))
of_model.load_state_dict(torch.load("RAFT/raft-kitti.pth"))
of_model = of_model.module
of_model.to('cuda')
of_model.eval()