import torch
import torch.nn as nn
import random


class TRILoss(nn.Module):
    def __init__(self, args):
        super(TRILoss, self).__init__()
        self.loss_function = nn.TripletMarginLoss(margin=args.margin, p=2, eps=1e-6, swap=False, reduction='mean').to(args.device)

    def forward(self, features, labels):
        # left valid labels and features
        labels = labels[labels != 0]
        batch_size = labels.shape[0]
        features = features[:batch_size]

        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # import ipdb; ipdb.set_trace()
        # select pos and neg
        
        loss = 0

        for k in range(5):
            pos_feature = torch.ones_like(anchor_feature)
            neg_feature = torch.ones_like(anchor_feature)
            for i in range(mask.shape[0]):
                pos_indexs = [index for index, value in enumerate(mask[i]) if value == 1]
                neg_indexs = [index for index, value in enumerate(mask[i]) if value == 0]
                pos_feature[i] = contrast_feature[random.choice(pos_indexs)]
                neg_feature[i] = contrast_feature[random.choice(neg_indexs)]    
            loss += self.loss_function(anchor_feature, pos_feature, neg_feature)
        # loss

        return loss
