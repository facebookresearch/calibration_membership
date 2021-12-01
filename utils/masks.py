# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch

import operator

def to_mask(n_data, indices):
    mask = torch.zeros(n_data, dtype=bool)
    mask[indices] = 1

    return mask


def multiply_round(n_data, cfg):
    s_total = sum(cfg.values())
    sizes = {name: int(s * n_data / s_total) for name, s in cfg.items()}

    max_name = max(sizes.items(), key=operator.itemgetter(1))[0]
    sizes[max_name] += n_data - sum(sizes.values())

    return sizes


def generate_masks(n_data, split_config):
    assert type(split_config) is dict
    assert "public" in split_config and "private" in split_config
    assert type(split_config["private"]) is dict

    permutation = np.random.permutation(n_data)
    if type(split_config["public"]) is dict:
        n_public=int(sum(split_config["public"].values())*n_data)
    else:
        n_public = int(split_config["public"] * n_data)
    n_private = n_data - n_public

    known_masks = {}
    known_masks["public"] = to_mask(n_data, permutation[:n_public])
    known_masks["private"] = to_mask(n_data, permutation[n_public:])

    hidden_masks = {}
    
    hidden_masks["private"] = {}

    sizes = multiply_round(n_private, split_config["private"])
    print(' Private', sizes)
    offset = n_public
    for name, size in sizes.items():
        hidden_masks["private"][name] = to_mask(n_data, permutation[offset:offset+size])
        offset += size

    assert offset == n_data

    if type(split_config["public"]) is dict:
        hidden_masks["public"] = {}
        public_sizes = multiply_round(n_public, split_config["public"])
        print('Public', public_sizes)
        public_offset = 0
        for name, size in public_sizes.items():
            hidden_masks["public"][name] = to_mask(n_data, permutation[public_offset:public_offset+size])
            public_offset += size
        assert public_offset == n_public
    else:
        hidden_masks["public"] = known_masks["public"]

    return known_masks, hidden_masks

def evaluate_masks(guessed_membership, private_masks, threshold, attack_base=None):

    if attack_base=='loss' or attack_base=='mean':
        true_positives = (guessed_membership[private_masks["train"]] <= threshold).float()
        false_negatives= (guessed_membership[private_masks["train"]] > threshold).float()
        true_negatives = (guessed_membership[private_masks["heldout"]] > threshold).float()
        false_positives = (guessed_membership[private_masks["heldout"]] <= threshold).float()
    else:
        true_positives = (guessed_membership[private_masks["train"]] >= threshold).float()
        false_negatives = (guessed_membership[private_masks["train"]] < threshold).float()
        true_negatives = (guessed_membership[private_masks["heldout"]] < threshold).float()
        false_positives = (guessed_membership[private_masks["heldout"]] >= threshold).float()

    fpr=torch.sum(false_positives) / (torch.sum(false_positives) + torch.sum(true_negatives))
    recall = torch.sum(true_positives) / torch.sum(private_masks["train"].float())
    precision = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_positives))

    accuracy = (torch.sum(true_positives) + torch.sum(true_negatives)) / (torch.sum(private_masks["heldout"].float()) + torch.sum(private_masks["train"].float()))

    return fpr, precision, recall, accuracy
