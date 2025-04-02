#!/usr/bin/env python
#
# Copyright (c) 2017 10X Genomics, Inc. All rights reserved.
#

from collections import namedtuple

FeatureDef = namedtuple('FeatureDef', ['index', 'id', 'name', 'feature_type', 'tags'])

# Required HDF5 datasets
REQUIRED_DATASETS = ['id', 'name', 'feature_type']

class FeatureDefException(Exception):
    pass

class FeatureReference(object):
    '''Store a list of features (genes, antibodies, etc).'''

    def __init__(self, feature_defs, all_tag_keys):
        '''Create a FeatureReference.

        Args:
            feature_defs (list of FeatureDef): All feature definitions.
            all_tag_keys (list of str): All optional tag keys.
        '''
        self.feature_defs = feature_defs
        self.all_tag_keys = all_tag_keys

        # Assert uniqueness of feature IDs
        id_map = {}
        for fdef in self.feature_defs:
            if fdef.id in id_map:
                this_fd_str = 'ID: %s; name: %s; type: %s' % (fdef.id, fdef.name, fdef.feature_type)
                seen_fd_str = 'ID: %s; name: %s; type: %s' % (id_map[fdef.id].id, id_map[fdef.id].name, id_map[fdef.id].feature_type)
                raise FeatureDefException('Found two feature definitions with the same ID: (%s) and (%s). All feature IDs must be distinct.' %
                                          (this_fd_str, seen_fd_str))
            id_map[fdef.id] = fdef

        self.id_map = id_map

    def __eq__(self, other):
        return self.feature_defs == other.feature_defs

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def addtags(feature_ref, new_tags, new_labels=None):
        '''Add new tags and corresponding labels to existing feature_ref
           If new labels are None, empty strings are supplied by default
        
        Args:
            feature_ref: a FeatureReference instance
            new_tags: a list of new tags
            new_labels: per feature list of label values corresponding to the new tags
        '''
        assert len(new_tags) > 0
        for tag in new_tags:
            assert tag not in feature_ref.all_tag_keys

        use_labels = []
        if new_labels is not None:
            assert len(feature_ref.feature_defs) == len(new_labels)
            for labels in new_labels:
                assert len(labels) == len(new_tags)
            use_labels = new_labels
        else:
            # initialize to empty
            for i in range(len(feature_ref.feature_defs)):
                use_labels += [[''] * len(new_tags)]
        assert len(feature_ref.feature_defs) == len(use_labels)

        augmented_features = []
        for fd, newvals in zip(feature_ref.feature_defs, use_labels):
            A = {a:b for a,b in zip(new_tags, newvals)}
            A.update(fd.tags)
            augmented_features.append(FeatureDef(index=fd.index,
                                      id=fd.id,
                                      name=fd.name,
                                      feature_type=fd.feature_type,
                                      tags=A))

        return FeatureReference(feature_defs=augmented_features,
                                all_tag_keys=feature_ref.all_tag_keys + new_tags)

    @staticmethod
    def join(feature_ref1, feature_ref2):
        '''Concatenate two feature references, requires unique ids and identical tags'''
        assert feature_ref1.all_tag_keys == feature_ref2.all_tag_keys
        feature_defs1 = feature_ref1.feature_defs
        feature_defs2 = feature_ref2.feature_defs
        return FeatureReference(feature_defs=feature_defs1 + feature_defs2, all_tag_keys=feature_ref1.all_tag_keys)

    @classmethod
    def empty(cls):
        return cls(feature_defs=[], all_tag_keys=[])

    def get_num_features(self):
        return len(self.feature_defs)
