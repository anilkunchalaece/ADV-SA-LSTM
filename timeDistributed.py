# Ref - https://www.programmersought.com/article/96395072975/
# https://github.com/pytorch/pytorch/issues/1927
# Applying time-distributed layers to normal layers
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
    
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
    
        output = self.module(reshaped_input)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.size(0), -1,output.size(-1))
        else:
            # (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output