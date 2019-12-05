import torch
import torch.nn as nn
from utils.cuda import GRUUpdate, cuda, move_dgl_to_cuda
from dgl import batch, bfs_edges_generator
import dgl.function as DGLF
import numpy as np

MAX_NB = 8

def level_order(forest, roots):
    try:
        edges = bfs_edges_generator(forest, roots)
        _, leaves = forest.find_edges(edges[-1])
        edges_back = bfs_edges_generator(forest, roots, reverse=True)
        yield from reversed(edges_back)
        yield from edges
    except:
        return None

enc_tree_msg = [DGLF.copy_src(src='m', out='m'), DGLF.copy_src(src='rm', out='rm')]
enc_tree_reduce = [DGLF.sum(msg='m', out='s'), DGLF.sum(msg='rm', out='accum_rm')]
enc_tree_gather_msg = DGLF.copy_edge(edge='m', out='m')
enc_tree_gather_reduce = DGLF.sum(msg='m', out='m')

class EncoderGatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        self.W = nn.Linear(2 * hidden_size, hidden_size).cuda()

    def forward(self, nodes):
        #print(nodes.data.keys() )
        x = nodes.data['x']
        try:
            m = nodes.data['m']
        except:
            m = torch.cuda.FloatTensor(1, self.hidden_size).fill_(0)
        return {
            'h': cuda(torch.relu(self.W(cuda(torch.cat([x, m], 1))))),
        }


class DGLJTNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding=None):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size).cuda()
        else:
            self.embedding = embedding

        self.enc_tree_update = GRUUpdate(hidden_size).cuda()
        self.enc_tree_gather_update = EncoderGatherUpdate(hidden_size)

    def forward(self, mol_trees):
        mol_tree_batch = batch(mol_trees)
        move_dgl_to_cuda(mol_tree_batch)
        
        # Build line graph to prepare for belief propagation
        mol_tree_batch_lg = mol_tree_batch.line_graph(backtracking=False, shared=True)
        move_dgl_to_cuda(mol_tree_batch_lg)

        return self.run(mol_tree_batch, mol_tree_batch_lg)

    def run(self, mol_tree_batch, mol_tree_batch_lg):
        # Since tree roots are designated to 0.  In the batched graph we can
        # simply find the corresponding node ID by looking at node_offset
        node_offset = np.cumsum([0] + mol_tree_batch.batch_num_nodes)
        root_ids = node_offset[:-1]
        n_nodes = mol_tree_batch.number_of_nodes()
        n_edges = mol_tree_batch.number_of_edges()

        # Assign structure embeddings to tree nodes
        x = cuda(self.embedding(cuda(mol_tree_batch.ndata['wid'])))
        h = torch.cuda.FloatTensor(n_nodes, self.hidden_size).fill_(0)
        
        mol_tree_batch.ndata.update({
            'x': x,
            'h': h,
        })
        

        # Initialize the intermediate variables according to Eq (4)-(8).
        # Also initialize the src_x and dst_x fields.
        # TODO: context?
        mol_tree_batch.edata.update({
            's': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
            'm': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
            'r': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
            'z': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
            'src_x': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
            'dst_x': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
            'rm': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
            'accum_rm': torch.cuda.FloatTensor(n_edges, self.hidden_size).fill_(0),
        })

        # Send the source/destination node features to edges
        mol_tree_batch.apply_edges(
            func=lambda edges: {'src_x': edges.src['x'], 'dst_x': edges.dst['x']},
        )

        # Message passing
        # I exploited the fact that the reduce function is a sum of incoming
        # messages, and the uncomputed messages are zero vectors.  Essentially,
        # we can always compute s_ij as the sum of incoming m_ij, no matter
        # if m_ij is actually computed or not.
        for eid in level_order(mol_tree_batch, root_ids):
            #eid = mol_tree_batch.edge_ids(u, v)
            mol_tree_batch_lg.pull(
                eid,
                enc_tree_msg,
                enc_tree_reduce,
                self.enc_tree_update,
            )

        # Readout
        mol_tree_batch.update_all(
            enc_tree_gather_msg,
            enc_tree_gather_reduce,
            self.enc_tree_gather_update,
        )

        root_vecs = mol_tree_batch.nodes[root_ids].data['h']

        return mol_tree_batch, root_vecs