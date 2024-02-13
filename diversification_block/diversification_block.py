
import torch
from torch import nn
import numpy as np

class DiversificationBlock(nn.Module):

    def __init__(self, pk=0.5, r=3, c=4):
        """
        Implement the Diversification Block in the paper by taking three-dimensional feature map as input, 
        and return a numpy list as a mask
        
        :param pk: pk is the probability of random mask in bc'
        :param r: bc'' is divided into several pieces
        :param c: bc'' column is divided into several pieces
        """
        super(DiversificationBlock, self).__init__()
        self.pk = pk
        self.r = r
        self.c = c

    def forward(self, feature_maps):
        
        def peak_supress(feature_map):
            row, col = torch.where(feature_map == torch.max(feature_map))
            b1 = torch.zeros_like(feature_map)
            for i in range(len(row)):
                r, c = int(row[i]), int(col[i])
                b1[r, c] = 1
            mask = torch.bernoulli(torch.full_like(b1, self.pk))
            b1 = b1 * mask
            return b1

        def from_num_to_block(mat, r, c, num):
            assert len(mat.shape) == 2, ValueError("Feature map shape is wrong!")
            res = np.zeros_like(mat)
            row, col = mat.shape
            block_r, block_c = int(row / r), int(col / c)
            index = np.arange(r * c) + 1
            index = index.reshape(r, c)
            index_r, index_c = np.argwhere(index == num)[0]
            if index_c + 1 == c:
                end_c = c + 1
            else:
                end_c = (index_c + 1) * block_c
            if index_r + 1 == r:
                end_r = r + 1
            else:
                end_r = (index_r + 1) * block_r
            res[index_r * block_r: end_r, index_c * block_c:end_c] = 1
            return res

        if len(feature_maps.shape) == 3:
            resb1 = []
            resb2 = []
            feature_maps_list = torch.split(feature_maps, 1)
            for feature_map in feature_maps_list:
                feature_map = feature_map.squeeze()
                tmp = peak_supress(feature_map)
                resb1.append(tmp)
                tmp1 = from_num_to_block(feature_map, self.r, self.c, 3)
                resb2.append(tmp1)

        elif len(feature_maps.shape) == 2:
            tmp = peak_supress(feature_maps)
            tmp1 = from_num_to_block(feature_maps, self.r, self.c, 3)
            resb1 = [tmp]
            resb2 = [tmp1]

        else:
            raise ValueError
        res = [np.clip(resb1[x].numpy() + resb2[x], 0, 1) for x in range(len(resb1))]
        return res


if __name__ == '__main__':
    feature_maps = torch.rand([3,3,4])
    print("Feature maps:\n", feature_maps)
    print("Feature maps shape:\n", feature_maps.shape)
    db = DiversificationBlock()
    res = db(feature_maps)
    print("Result:\n", res)
    print("Result shape:\n", res[0].shape)





