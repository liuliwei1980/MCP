import torch
import readData
class MCPDataset():
    def __init__(self):
        super(MCPDataset, self).__init__()
        graphImg, ecfp, hash,list_of_node,list_of_edge, label = readData.all_data()
        self.Y = label
        dict_data = {'graphImg': graphImg, 'ecfp': ecfp,'hash': hash, "list_of_node":list_of_node,"list_of_edge":list_of_edge,'label': label}
        self.dict_data = dict_data

    def __getitem__(self, idx):
        # item = {key: torch.tensor(value[idx]) for key, value in self.dict_data.items()}
        item = {
            key: torch.tensor(value[idx]) if not isinstance(value[idx], torch.Tensor) else value[idx].clone().detach()
            for key, value in self.dict_data.items()}
        return item
    def __len__(self):
        return len(self.Y)





