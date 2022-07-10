import numpy as np

class Laplacian():
    """ The Graph to model the skeletons of human body/hand

    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration

        layout (string): must be one of the follow candidates
        - 'hm36_gt' same with ground truth structure of human 3.6 , with 17 joints per frame

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 pad=0):
        self.seqlen = 2 * pad + 1
        self.get_edge()
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.L = self.normalize_digraph(A)

    def __str__(self):
        return self.L

    def graph_link_between_frames(self,base):
        """
        calculate graph link between frames given base nodes and seq_ind
        :param base:
        :return:
        """
        return [((front - 1) + i*self.num_node_each, (back - 1)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]


    def basic_layout(self,neighbour_base):
        """
        for generating basic layout time link selflink etc.
        neighbour_base: neighbour link per frame
        :return: link each node with itself
        """
        self.num_node = self.num_node_each * self.seqlen

        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                     for j in range(self.num_node_each)]

        self_link = [(i, i) for i in range(self.num_node)]

        neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        return self_link, time_link, neighbour_link_all

    def get_edge(self):
        """
        get edge link of the graph
        la,ra: left/right arm
        ll/rl: left/right leg
        cb: center bone
        """
        self.num_node_each = 17

        neighbour_base = [(1, 2), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
                            (8, 1), (9, 8), (10, 9), (11, 10), (12, 9),
                            (13, 12), (14, 13), (15, 9), (16, 15), (17, 16)
                            ]

        self_link, time_link, neighbour_link_all = self.basic_layout(neighbour_base)

        self.la, self.ra =[11, 12, 13], [14, 15, 16]
        self.ll, self.rl = [4, 5, 6], [1, 2, 3]
        self.cb = [0, 7, 8, 9, 10]
        self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

        self.edge = self_link + neighbour_link_all + time_link

        # center node of body/hand
        self.center = 8 - 1

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        D = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                D[i, i] = Dl[i]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            Dn[i, i] = (D[i, i])**(-1)
        return np.dot(np.dot(Dn, A), Dn)

if __name__ == "__main__":
    L = Laplacian(1)
    print(L.L, L.L.shape)