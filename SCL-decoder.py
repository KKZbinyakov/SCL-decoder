import math
from copy import deepcopy
from numpy import random


def L(x, y):
    """
    Computes the log-likelihood ratio (LLR) for the F operation in polar codes.
    
    Formula: sign(x*y) * min(|x|, |y|)
    
    Args:
        x (float): First input LLR value
        y (float): Second input LLR value
        
    Returns:
        float: Result of the F operation
    """
    return (x * y / abs(x * y)) * min(abs(x), abs(y))


def R(x, y, b):
    """
    Computes the LLR for the G operation in polar codes using previous bit.
    
    Formula:
        b = 0: x + y
        b = 1: y - x
    
    Args:
        x (float): First input LLR value
        y (float): Second input LLR value
        b (int): Decoded bit from previous step (0 or 1)
        
    Returns:
        float: Result of the G operation
    """
    if b == 0:
        return x + y
    else:
        return y - x


def find_error(pred, true):
    """
    Computes the error between predicted LLR and true bit.
    
    Rules:
        - Correct prediction: (pred >= 0 and true=0) OR (pred < 0 and true=1) -> error=0
        - Incorrect prediction: error = |pred|
    
    Args:
        pred (float): Predicted LLR value
        true (int): True bit value (0 or 1)
        
    Returns:
        float: Error value
    """
    if (pred >= 0 and true == 0) or (pred < 0 and true == 1):
        return 0
    else:
        return abs(pred)


def concat(u, v):
    """
    Combines two vectors using XOR operation.
    
    Format: (u[0]⊕v[0], u[1]⊕v[1], ..., v[0], v[1], ...)
    
    Args:
        u (list): First bit vector
        v (list): Second bit vector
        
    Returns:
        list: Combined vector
    """
    return list((a ^ b for a, b in zip(u, v))) + v


def polar_encode(message):
    """
    Encodes a binary message into polar code using binary tree construction.
    
    Process:
        1. Builds tree bottom-up
        2. Leaves = original message bits
        3. Parent nodes = concat(left_child, right_child)
    
    Args:
        message (list): Binary bits to encode (0/1)
        
    Returns:
        list: Encoded message (root of tree)
    """
    tree = []
    for i in range(int(math.log2(len(message))) + 1):
        tree.append([])
    curnodes = 1
    for level in range(len(tree)):
        for node in range(curnodes):
            tree[level].append([])
        curnodes *= 2
    
    for i in range(len(message)):
        tree[-1][i] = [message[i]]

    for level in range(len(tree) - 2, -1, -1):
        for node in range(len(tree[level])):
            left_child = tree[level + 1][2 * node]
            right_child = tree[level + 1][2 * node + 1]
            tree[level][node] = concat(left_child, right_child)
    return tree[0][0]


def BPSK(message):
    """
    Performs BPSK modulation (0 -> +1, 1 -> -1).
    
    Args:
        message (list): Binary bits (0/1)
        
    Returns:
        list: Modulated signal (±1)
    """
    for i in range(len(message)):
        if message[i] == 0:
            message[i] = 1
        else:
            message[i] = -1
    return message


def add_noice(message, mu, sigma):
    """
    Adds Gaussian noise to signal.
    
    Args:
        message (list): BPSK modulated signal
        mu (float): Mean of Gaussian distribution
        sigma (float): Standard deviation of Gaussian distribution
        
    Returns:
        list: Noisy signal
    """
    noice = random.normal(mu, sigma, len(message))
    for i in range(len(message)):
        message[i] += noice[i]
    return message


def add_errors(message, positions, mu, sigma):
    """
    Artificially flips bits at specified positions and adds noise.
    
    Args:
        message (list): BPSK modulated signal
        positions (list): Indices to force errors
        mu (float): Mean of Gaussian distribution
        sigma (float): Standard deviation of Gaussian distribution
        
    Returns:
        list: Signal with artificial errors
    """
    noice = random.normal(mu, sigma, len(message))
    for i in range(len(message)):
        if i in positions:
            message[i] = ((message[i] / abs(message[i])) * -1) + noice[i] 
    return message


class Node:
    """
    Represents a node in polar decoding tree.
    
    Attributes:
        value (list/float): LLR values for node
        bits (list): Decoded bits for node
        level (int): Tree level (0=root)
        position (int): Position within level
    """
    def __init__(self, value=None, bits=None, level=None, position=None):
        self.value = value
        self.bits = bits
        self.level = level
        self.position = position

    def get_parent(self, tree):
        """
        Gets parent node from tree structure.
        
        Args:
            tree (list): Tree structure containing nodes
            
        Returns:
            Node: Parent node or None if root
        """
        return tree[self.level - 1][self.position // 2] if self.level > 0 else None

    def get_left_child(self, tree):
        """
        Gets left child node from tree structure.
        
        Args:
            tree (list): Tree structure containing nodes
            
        Returns:
            Node: Left child or None if leaf
        """
        if self.level < len(tree) - 1:
            return tree[self.level + 1][2 * self.position]
        return None

    def get_right_child(self, tree):
        """
        Gets right child node from tree structure.
        
        Args:
            tree (list): Tree structure containing nodes
            
        Returns:
            Node: Right child or None if leaf
        """
        if self.level < len(tree) - 1:
            return tree[self.level + 1][2 * self.position + 1]
        return None


def build_tree(nbits):
    """
    Builds empty decoding tree structure.
    
    Structure:
        - Levels: log2(nbits) + 1
        - Nodes at level i: 2^i
    
    Args:
        nbits (int): Length of encoded message
        
    Returns:
        list: Tree structure with initialized nodes
    """
    nlevels = int(math.log2(nbits)) + 1
    tree = []
    nodes_count = 1
    for level in range(nlevels):
        level_nodes = []
        for i in range(nodes_count):
            level_nodes.append(Node(level=level, position=i))
        tree.append(level_nodes)
        nodes_count *= 2
    return tree


def proceed_left_node(iterator, frozen_bits, trees):
    """
    Processes left child node during SC decoding.
    
    Operations:
        1. Computes LLRs using L function
        2. For leaf nodes:
            - Frozen bits: force bit=0
            - Non-frozen: create branches for 0 and 1
    
    Args:
        iterator (TreeIterator): Current tree iterator
        frozen_bits (list): Positions of frozen bits
        trees (list): List to store new tree branches
    """
    node = iterator.next_node()
    tree = iterator.tree
    node.value = []
    parent_value = node.get_parent(tree).value

    if node.level != len(tree) - 1:
        stacked = [
            parent_value[: len(parent_value) // 2],
            parent_value[len(parent_value) // 2 :],
        ]

        for i in range(len(stacked[0])):
            node.value.append(L(stacked[0][i], stacked[1][i]))

        if node.get_left_child(tree).value and node.get_right_child(tree).value:
            node.bits = concat(
                node.get_left_child(tree).bits, node.get_right_child(tree).bits
            )

    else:
        value = L(parent_value[0], parent_value[1])
        node.value = value

        if node.position in frozen_bits:
            node.bits = [0]

        else:
            node.bits = [0]
            trees.append(deepcopy(iterator))
            node.bits = [1]

    return node


def proceed_right_node(iterator, frozen_bits, trees):
    """
    Processes right child node during SC decoding.
    
    Operations:
        1. Computes LLRs using R function and left sibling bits
        2. For leaf nodes:
            - Frozen bits: force bit=0
            - Non-frozen: create branches for 0 and 1
    
    Args:
        iterator (TreeIterator): Current tree iterator
        frozen_bits (list): Positions of frozen bits
        trees (list): List to store new tree branches
    """
    node = iterator.next_node()
    tree = iterator.tree
    node.value = []
    parent_value = node.get_parent(tree).value

    if node.level != len(tree) - 1:
        stacked = [
            parent_value[: len(parent_value) // 2],
            parent_value[len(parent_value) // 2 :],
        ]

        bits_for_R = tree[node.level][node.position - 1].bits

        for i in range(len(stacked[0])):
            node.value.append(R(stacked[0][i], stacked[1][i], bits_for_R[i]))

        if node.get_left_child(tree).value and node.get_right_child(tree).value:
            node.bits = concat(
                node.get_left_child(tree).bits, node.get_right_child(tree).bits
            )
            parent = node.get_parent(tree)
            parent.bits = concat(
                parent.get_left_child(tree).bits, parent.get_right_child(tree).bits
            )

    else:
        value = R(
            parent_value[0],
            parent_value[1],
            node.get_parent(tree).get_left_child(tree).bits[0],
        )
        node.value = value

        if node.position in frozen_bits:
            node.bits = [0]
            parent = node.get_parent(tree)
            parent.bits = concat(
                parent.get_left_child(tree).bits, parent.get_right_child(tree).bits
            )

        else:
            node.bits = [0]
            parent = node.get_parent(tree)
            parent.bits = concat(
                parent.get_left_child(tree).bits, parent.get_right_child(tree).bits
            )
            trees.append(deepcopy(iterator))
            node.bits = [1]
            parent.bits = concat(
                parent.get_left_child(tree).bits, parent.get_right_child(tree).bits
            )

    return node


def proceed_current_node(iterator, frozen_bits, trees):
    """
    Processes current node based on position (left/right child).
    
    Args:
        iterator (TreeIterator): Current tree iterator
        frozen_bits (list): Positions of frozen bits
        trees (list): List to store new tree branches
    """
    node = iterator.stack[-1]
    if node.position % 2 == 0:
        proceed_left_node(iterator, frozen_bits, trees)
    else:
        proceed_right_node(iterator, frozen_bits, trees)


def get_current_decode(tree):
    """
    Extracts decoded message from leaf nodes and computes error.
    
    Args:
        tree (list): Current decoding tree
        
    Returns:
        tuple: (decoded bits, total error)
    """
    decode = []
    error = 0

    leaves = tree[len(tree) - 1]

    for leave in leaves:
        if leave.value != None:
            decode.append(*leave.bits)

    for i in range(len(decode)):
        error += find_error(leaves[i].value, decode[i])

    return decode, error


def get_tree_size(tree):
    """
    Calculates size of tree (number of defined nodes).
    
    Args:
        tree (list): Decoding tree structure
        
    Returns:
        int: Tree size metric
    """
    size = 0

    for level in tree:
        for leave in level:
            if leave.value != None:
                size += 1
            if leave.bits != None:
                size += 1

    return size


class TreeIterator:
    """
    Depth-first iterator for polar decoding trees.
    
    Attributes:
        tree (list): Tree structure to traverse
        stack (list): Stack of nodes to process
    """
    def __init__(self, tree, root):
        self.tree = tree
        self.stack = [root]

    def next_node(self):
        """
        Gets next node using depth-first traversal.
        
        Returns:
            Node: Next node to process or None
        """
        if not self.stack:
            return None

        node = self.stack.pop()

        if node.level < len(self.tree) - 1:
            right_child = node.get_right_child(self.tree)
            left_child = node.get_left_child(self.tree)

            if node.bits == None:
                if node.position != 0 or node.level != 0:
                    self.stack.append(node)

                if right_child.bits == None:
                    self.stack.append(right_child)

                if left_child.bits == None:
                    self.stack.append(left_child)

        return node


def find_smallest_tree(trees):
    """
    Finds index of smallest tree by size metric.
    
    Args:
        trees (list): List of TreeIterator objects
        
    Returns:
        int: Index of smallest tree, or -1 if empty
    """
    if not trees:
        return -1

    smallest_size = get_tree_size(trees[0].tree)
    smallest_index = 0

    for iter in trees:
        tree = iter.tree
        index = trees.index(iter)
        current_size = get_tree_size(tree)
        if current_size < smallest_size:
            smallest_size = current_size
            smallest_index = index

    return smallest_index


def all_trees_same_size(trees):
    """
    Checks if all trees have identical size metric.
    
    Args:
        trees (list): List of TreeIterator objects
        
    Returns:
        bool: True if all sizes equal, False otherwise
    """
    if not trees:
        return True

    sizes = {get_tree_size(iter.tree) for iter in trees}

    return len(sizes) == 1


def find_best_decodings(noicy_message, possible_trees, frozen_bits):
    """
    Performs list decoding to find top candidate decodings.
    
    Algorithm:
        1. Initialize trees with root node = received signal
        2. Process nodes while maintaining candidate trees
        3. Prune to retain top candidates by error metric
        
    Args:
        noicy_message (list): Received noisy signal
        possible_trees (int): Number of candidates to retain
        frozen_bits (list): Positions of frozen bits
    """
    trees = []
    tree = build_tree(len(noicy_message))
    tree[0][0] = Node(
        noicy_message,
        None,
        0,
        0,
    )

    iterator = TreeIterator(tree, tree[0][0])

    trees.append(iterator)

    smallest_iter = trees[0]

    current_node = smallest_iter.next_node()

    proceed_current_node(iterator, frozen_bits, trees)

    while True:

        smallest_index = find_smallest_tree(trees)

        smallest_iter = trees[smallest_index]

        proceed_current_node(smallest_iter, frozen_bits, trees)

        if all_trees_same_size(trees) and len(trees) != 0:
            trees.sort(key=lambda item: get_current_decode(item.tree)[1])
            for iter in trees:
                tree = iter.tree
                decode, error = get_current_decode(iter.tree)
            trees = trees[:possible_trees]
            if len(smallest_iter.stack) == 0:
                break
        else:
            if len(smallest_iter.stack) == 0:
                break


    for iter in trees:
        decode, error = get_current_decode(iter.tree)
        print(decode, error)        


def main():
    """Main function to execute polar coding simulation."""
    message = list(map(int, input('Enter message of 0s and 1s (with padding) separated by spaces: ').split()))
    original_message = message.copy()
    
    frozen_bits = list(map(int, input('Enter positions of frozen bits (space-separated): ').split()))
    
    encoded = polar_encode(message)
    print('\n' + '='*60)
    print(f'POLAR ENCODED MESSAGE [{len(encoded)} bits]:')
    print(' '.join(map(str, encoded)))
    
    bpsk = BPSK(encoded)
    print('\n' + '='*60)
    print(f'BPSK MODULATED SIGNAL [{len(bpsk)} symbols]:')
    print(' '.join([f'{x:.4f}' for x in bpsk]))
    
    mu, sigma = map(float, input("\nEnter μ and σ for Gaussian noise (space-separated): ").split())
    noisy = add_noice(bpsk, mu, sigma)
    print('\n' + '='*60)
    print(f'SIGNAL WITH GAUSSIAN NOISE (μ={mu}, σ={sigma}):')
    print(' '.join([f'{x:>7.4f}' for x in noisy]))
    
    error_bits = list(map(int, input("\nEnter error positions (space-separated): ").split()))
    errored = add_errors(noisy, error_bits, mu, sigma)
    print('\n' + '='*60)
    print(f'SIGNAL WITH {len(error_bits)} ARTIFICIAL ERRORS:')
    print(' '.join([f'{x:>7.4f}' for x in errored]))
    
    num_trees = int(input('\nEnter number of decoding trees to retain: '))
    print('\n' + '='*60)
    print(f'TOP-{num_trees} DECODING CANDIDATES:')
    find_best_decodings(errored, num_trees, frozen_bits)
    
    print('\n' + '='*60)
    print('ORIGINAL TRANSMITTED MESSAGE:')
    print(' '.join(map(str, original_message)))


if __name__ == '__main__':
    main()