#!/usr/bin/env python3
"""decision tree task 0"""
import numpy as np


class Node:
    """Node class"""
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
        """max depth below method"""
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """count nodes below method"""
        if self.is_leaf:
            return 1

        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves) if self.left_child else 0
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves) if self.right_child else 0

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Add prefix for left child"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Add prefix for right child"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("       "+x)+"\n"
        return new_text

    def __str__(self):
        """String representation of the node and its children"""
        label = "root" if self.is_root else "-> node"
        left_str = self.left_child_add_prefix(
            str(self.left_child)) if self.left_child else ""
        right_str = self.right_child_add_prefix(
            str(self.right_child)) if self.right_child else ""
        return f"{label} [feature={self.feature}, threshold=\
{self.threshold}]\n{left_str}{right_str}".strip()

    def get_leaves_below(self):
        """Returns a list of all leaves in the subtree rooted at this node."""
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """update bounds below method"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:

            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


class Leaf(Node):
    """Leaf class"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """max depth below method"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """count nodes below method"""
        return 1

    def __str__(self):
        """String representation of a leaf node"""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """get leaves method"""
        return [self]

    def update_bounds_below(self):
        """update bounds below method"""
        pass


class Decision_Tree():
    """Decision tree class"""
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
        """String representation of the decision tree"""
        return self.root.__str__()

    def get_leaves(self):
        """get leaves method"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """update bounds method"""
        self.root.update_bounds_below()
