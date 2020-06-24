import torch


class TargetList(object):
    def __init__(self, class_ids, domain_ids):
        if not isinstance(class_ids, torch.Tensor):
            self.class_ids = torch.tensor(class_ids, dtype=torch.long)
            self.domain_ids = torch.tensor(domain_ids, dtype=torch.long)
        else:
            self.class_ids = class_ids.long()
            self.domain_ids = domain_ids.long()

    def transpose(self, *args):
        return self

    def resize(self, *args):
        return self

    def __len__(self):
        return self.class_ids.shape[0]

    def to(self, device):
        t = TargetList(self.class_ids, self.domain_ids)
        t.class_ids = t.class_ids.to(device)
        t.domain_ids = t.domain_ids.to(device)
        return t


def concat_target_list(target_lists):
    res_class_ids = []
    res_domain_ids = []

    for target_list in target_lists:
        res_class_ids.append(target_list.class_ids)
        res_domain_ids.append(target_list.domain_ids)

    res_class_ids = torch.cat(res_class_ids, dim=-1)
    res_domain_ids = torch.cat(res_domain_ids, dim=-1)
    return TargetList(res_class_ids, res_domain_ids)
