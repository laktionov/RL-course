import numpy as np

# You can use this implementation of SumTree structure by MorvanZhou
# to make sampling O(log n)
class SumTree():
    """
    Stores the priorities in sum-tree structure for effecient sampling.
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity

    def update(self, idx, p):
        """
        input: idx - int, id of leaf to update
        input: p - float, new priority value
        """
        assert idx < self.capacity, "SumTree overflow"
        
        idx += self.capacity - 1  # going to leaf №i
        
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:    # faster than the recursive loop
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        """
        input: v - float, cumulative priority of first i leafs
        output: i - int, selected index
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx] or self.tree[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        return leaf_idx - (self.capacity - 1)
        
    def __getitem__(self, indices):
        return self.tree[indices + self.capacity - 1]

    @property
    def total_p(self):
        return self.tree[0]  # the root is sum of all priorities

class PrioritizedSampler():
    def __init__(self, size, clip_priorities=1.0, rp_alpha=0.4):
        """
        Parameters
        ----------
        size: int
            Max number of prirotieis to store (same as your replay buffer size). 
        clip_priorities: float
            All priorities higher than this will be clipped. You can experiment with this...
        rp_alpha: float
            p**rp_alpha is used as sampling probability, where p is priority
        """
        self.clip_priorities = clip_priorities
        self.rp_alpha = rp_alpha     
        
        self.priorities = SumTree(size)    

        # New transitions should be added to replay buffer with max priority
        # (or you can try to compute priority proxy in "play_and_record" code)
        self.max_priority = 1.0  

    def sample_indices(self, batch_size):
        '''
        Samples indices using priorities
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        indices: np.array
            batch of indices to sample from replay buffer
        weights: np.array
            batch of importance sampling weights
        """
        '''
        # sample batch_size indices
        indices = <YOUR CODE>
        
        priorities = self.priorities[indices]
        
        # you can change this to implement bias correction
        weights = np.ones(batch_size)
        return indices, weights

    def update_priorities(self, indices, new_priorities):
        '''
        Updates priorities with new_priorities on given indices.
        Parameters
        ----------
        indices: np.array
            indices to update
        new_priorities: np.array
            new priorities to set
        '''
        new_priorities = (new_priorities**self.rp_alpha).clip(min=1e-5, max=self.clip_priorities)
        
        # update priorities
        <YOUR CODE>
        
        # update max priority for new transitions
        self.max_priority = max(self.max_priority, new_priorities.max())