#!/usr/bin/env python3
"""decision tree task 0"""
import numpy as np


class Node:
    """node class"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """max depth method"""
        if self.is_leaf:
            return self.depth

        left_depth = self.left_child.max_depth_below()\
            if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below()\
            if self.right_child else self.depth

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """count nodes method"""
        if self.is_leaf:
            return 1

        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves) if self.left_child else 0
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves) if self.right_child else 0

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def __str__(self):
        """str rep method"""
        if self.is_root:
            node_str = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            node_str = f"node [feature={self.feature}, threshold={self.threshold}]\n"

        if self.left_child:
            node_str += left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            node_str += right_child_add_prefix(self.right_child.__str__())
        return node_str


def left_child_add_prefix(text):
    """left child prefix method"""
    lines = text.split("\n")
    new_text = "    +--"+lines[0]+"\n"
    for x in lines[1:]:
        new_text += ("    |  "+x)+"\n"
    return (new_text)


def right_child_add_prefix(text):
    """right child prefix method"""
    lines = text.split("\n")
    new_text = "    +---> " + lines[0] + "\n"
    for x in lines[1:]:
        if x.strip():
            new_text += "       " + x + "\n"
    return new_text


class Leaf(Node):
    """leaf class"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """max depth method"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """count nodes mthod"""
        return 1

    def __str__(self):
        """str rep method"""
        return f"leaf [value={self.value}]"


class Decision_Tree():
    """decision tree class"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """depth method"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """count nodes method"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """str rep method"""
        return self.root.__str__()
