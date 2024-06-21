# Code from https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces
# More details can be found in the paper: Realistic Website Fingerprinting By Augmenting Network Trace. CCS 2023.

import random
import numpy as np
import bisect

class Augmentor():
    def __init__(self, max_outgoing_burst_size, outgoing_burst_sizes, OUTGOING_BURST_SIZE_CDF):
        methods = {
            'merge downstream burst',
            'change downstream burst sizes',
            'merge downstream and upstream bursts',
            'add upstream bursts',
            'remove upstrean bursts',
            'divide bursts'
        }
    
        self.large_burst_threshold = 10
        
        # changing the content
        self.upsample_rate = 1.0
        self.downsample_rate = 0.5
        
        # merging bursts
        self.num_bursts_to_merge = 5
        self.merge_burst_rate = 0.1
        
        # add incoming bursts
        self.add_outgoing_burst_rate = 0.3
        self.outgoing_burst_sizes = list(range(max_outgoing_burst_size))
        self.outgoing_burst_sizes2 = outgoing_burst_sizes
        self.OUTGOING_BURST_SIZE_CDF = OUTGOING_BURST_SIZE_CDF
        
        # shift
        self.shift_param = 10
        
        
        
    def find_bursts(self, x):
        direction = x[0]
        bursts = []
        start = 0
        temp_burst = x[0]
        for i in range(1, len(x)):
            if x[i] == 0.0:
                break

            elif x[i] == direction:
                temp_burst += x[i]

            else:
                # if temp_burst <= -10 or temp_burst > 0:
                bursts.append((start, i, temp_burst))
                start = i
                temp_burst = x[i]
                direction *= -1

        return bursts
        
        
    # representing the change of contents of a website
    def increase_incoming_bursts(self, burst_sizes):
        out = []
        for i, size in enumerate(burst_sizes):
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random()*self.upsample_rate
                new_size = int(size * (1+up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
                
        return out
        
        
    def decrease_incoming_bursts(self, burst_sizes):
        out = []
        for i, size in enumerate(burst_sizes):
            if size <= -self.large_burst_threshold:
                up_sample_rate = random.random()*self.downsample_rate
                new_size = int(size * (1-up_sample_rate))
                out.append(new_size)
            else:
                out.append(size)
                
        return out
        
        
    def change_content(self, trace):
        bursts = self.find_bursts(trace)
        burst_sizes = [x[2] for x in bursts]
        
        if len(trace) < 1000:
            new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
            
        elif len(trace) > 4000:
            new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
            
        else:
            p = random.random()
            if p >= 0.5:
                new_burst_sizes = self.increase_incoming_bursts(burst_sizes)
                
            else:
                new_burst_sizes = self.decrease_incoming_bursts(burst_sizes)
                
                
        return new_burst_sizes
    
    
    def merge_incoming_bursts(self, burst_sizes):
        
        out = []
        
        # skipping first 20 cells
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
            
        
        while i < len(burst_sizes) - self.num_bursts_to_merge:
            prob = random.random()
            
            # ignore outgoing bursts
            if burst_sizes[i] > 0:
                out.append(burst_sizes[i])
                i+= 1
                continue
            
            if prob < self.merge_burst_rate:
                num_merges = random.randint(2, self.num_bursts_to_merge)
                merged_size = 0
                
                # merging the incoming bursts
                while i < len(burst_sizes) and num_merges > 0:
                    if burst_sizes[i] < 0:
                        merged_size += burst_sizes[i]
                        num_merges -= 1
                    i += 1     
                out.append(merged_size)
                    
            else:
                out.append(burst_sizes[i])
                i += 1
                
        return out
    
    
    def add_outgoing_burst(self, burst_sizes):
        
        out = []
        
        i = 0
        num_cells = 0
        while i < len(burst_sizes) and num_cells < 20:
            num_cells += abs(burst_sizes[i])
            out.append(burst_sizes[i])
            i += 1
            
        
        for size in burst_sizes[i:]:
            if size > -10 :
                out.append(size)
                continue
            
            prob = random.random()
            
            if prob < self.add_outgoing_burst_rate:
                
                index = len(self.outgoing_burst_sizes2)
                while index >= len(self.outgoing_burst_sizes2):
                    outgoing_burst_prob = random.random()
                    index = bisect.bisect_left(self.OUTGOING_BURST_SIZE_CDF, outgoing_burst_prob)
                    
                outgoing_burst_size = self.outgoing_burst_sizes[index]
                divide_place = random.randint(3, abs(size) - 3)
                
                out += [-divide_place, outgoing_burst_size, -(abs(size) - divide_place)]
                
            else:
                out.append(size)
                
        return out
                
        
    def create_trace_from_burst_sizes(self, burst_sizes):
        out = []
        
        for size in burst_sizes:
            val = 1 if size > 0 else -1
            
            out += [val]*(int(abs(size)))
            
        if len(out) < 5000:
            out += [0]*(5000 - len(out))
            
        return np.array(out)[:5000]
    
    def shift(self, x):
        pad = np.random.randint(0, 2, size = (self.shift_param, ))
        pad = 2*pad-1
        zpad = np.zeros_like(pad)
        
        shift_val = np.random.randint(-self.shift_param, self.shift_param+1, 1)[0]
        shifted = np.concatenate((x, zpad, pad), axis=-1)
        shifted = np.roll(shifted, shift_val, axis=-1)
        shifted = shifted[:5000]
        
        return shifted
        
    
    def augment(self, trace):
        
        mapping = {
            0: self.change_content,
            1: self.merge_incoming_bursts,
            2: self.add_outgoing_burst
        }
        
        bursts = self.find_bursts(trace)
        
        burst_sizes = [x[2] for x in bursts]
        if len(burst_sizes) == 0:
            return trace
            
        aug_method = mapping[random.randint(0, len(mapping)-1)]

        augmented_sizes = aug_method(burst_sizes)
        
        augmented_trace = self.create_trace_from_burst_sizes(augmented_sizes)
        
        return self.shift(augmented_trace)