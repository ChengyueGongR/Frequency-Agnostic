import numpy as np

import torch
from torch.autograd import Variable

def embedded_dropout(embed, sigma, words, dropout=0.1, scale=None, is_training=False):
  #sigma = torch.where(torch.exp(sigma.weight) > 0.3, torch.ones_like(sigma.weight) * 0.3, torch.exp(sigma.weight))
  #m = torch.distributions.normal.Normal(torch.zeros_like(sigma), torch.ones_like(sigma) * 1)
  #sigma = m.sample() * sigma
  #binary_mask = torch.distributions.bernoulli.Bernoulli(torch.ones(embed.weight.size(0)) * 0.75).sample().cuda()
  #sigma = sigma * binary_mask.view([-1, 1])
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
    masked_sigma_weight = mask * sigma#torch.exp(sigma.weight) #* (torch.abs(torch.detach(embed.weight)) + 0.5) #torch.exp(sigma.weight)
  else:
    masked_embed_weight = embed.weight
    masked_sigma_weight = sigma#torch.exp(sigma.weight)
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    masked_sigma_weight = scale.expand_as(masked_sigma_weight) * masked_sigma_weight
  padding_idx = embed.padding_idx
  if padding_idx is None:
    padding_idx = -1
  #m = torch.distributions.normal.Normal(torch.zeros_like(masked_sigma_weight), torch.ones_like(masked_sigma_weight) * 1)
  #noise = m.sample()
  #if is_training and not is_switch:
  #  masked_embed_weight += masked_sigma_weight * noise

  X = torch.nn.functional.embedding(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  Y = torch.nn.functional.embedding(words, masked_sigma_weight,
  #  padding_idx, embed.max_norm, embed.norm_type,
  #  embed.scale_grad_by_freq, embed.sparse
  )
  return X, Y

if __name__ == '__main__':
  V = 50
  h = 4
  bptt = 10
  batch_size = 2

  embed = torch.nn.Embedding(V, h)

  words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
  words = torch.LongTensor(words)
  words = Variable(words)

  origX = embed(words)
  X = embedded_dropout(embed, words)

  print(origX)
  print(X)
