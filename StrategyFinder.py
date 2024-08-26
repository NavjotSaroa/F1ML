"""
Author: Navjot Saroa

Step 5 of process:
    This is the last step for all the prediction work, in this file we produce lap nodes, consisting of information
    about that laps predictions, generate a tree of possible pit strategies, and return the most optimal strategy
    that is computationally possible. I use a system similar to stockfish's, where after a certain depth of the tree,
    the tree stops growing. Except in my case, I just prune the worst branches and keep predicting more laps, keeping the 
    count of the leaf nodes at a certain limit. And then I finally return the best branch which hopefully is the optimal
    strategy
"""

import pandas as pd
import numpy as np
from FeaturePredictor import FeaturePredictor
from UDP import *
import queue

class StrategyNode():
    def __init__(self, lap_time, features):
        self.parent = None
        self.s_child = None
        self.m_child = None
        self.h_child = None
        self.lap_time = lap_time
        self.features = features
        self.total_time = None
        self.tagged = False

    def add_pit_time_penalty(self):
        """
        20.4 seconds is the average cost of a pitstop in Austria.
        """
        self.lap_time += 20.4

    def update_total_time(self):
        """
        Backtracks up the tree to find the total time of the strategy the
        branch represents.
        """
        self.total_time = self.lap_time
        temp = self.parent
        while temp is not None:
            self.total_time = temp.lap_time
            temp = temp.parent

    def all_children_tagged(self):
        """
        Checks if all the children are tagged. If so, this node will be cut off.
        """
        if self.s_child.tagged and self.m_child.tagged and self.h_child.tagged:
            return True
        else:
            return False

class EvaluateStrategy():
    
    def tagger_initial(self, node, elim_lst):
        """
        Tags the worst leaf nodes as per the elim_list.
        """
        if node in elim_lst:
            node.tagged = True

    def tagger(self, node):
        """
        Recursively tags all the parents of the nodes if needed.
        """
        if node.all_children_tagged:
            node.tagged = True
            self.tagger(node.parent)

    def prune(self, root):
        """
        Goes down the tree, and cuts off any nodes that have been tagged.
        """
        print("Pruning...")
        frontier = queue.Queue()
        frontier.put(root)
        
        while not frontier.empty():
            check_node = frontier.get()
            frontier.put(check_node.s_child)
            frontier.put(check_node.m_child)
            frontier.put(check_node.h_child)
            if check_node.tagged:
                if check_node.parent:
                    temp = check_node.parent
                    if check_node == temp.s_child:
                        temp.s_child = None
                    elif check_node == temp.m_child:
                        temp.m_child = None
                    else:
                        temp.h_child = None
                else:
                    return "Something went wrong"
        
        print("Done!")