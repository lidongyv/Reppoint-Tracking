import torch

from mmdet.ops.nms import nms_wrapper


def multiclass_nms_allclasses(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   nms_further_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """

    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)

    # nms_further_cfg_ = nms_further_cfg.copy()

    #------------------- first stage nms --------------------
    for i in range(1, num_classes):

        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    # ------------------- first stage nms --------------------

    #------------------- second stage nms --------------------
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)

        total_index = torch.arange(bboxes.shape[0])
        bboxes, bboxes_selected_index = nms_op(bboxes, 0.85)
        labels = labels[total_index[bboxes_selected_index].long()]
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    # ------------------- second stage nms --------------------

    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
    # if bboxes:
    #     bboxes = torch.cat(bboxes)
    #     labels = torch.cat(labels)
    #     if bboxes.shape[0] > max_num:
    #         _, inds = bboxes[:, -1].sort(descending=True)
    #         inds = inds[:max_num]
    #         bboxes = bboxes[inds]
    #         labels = labels[inds]
    # else:
    #     bboxes = multi_bboxes.new_zeros((0, 5))
    #     labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
    # return bboxes, labels






    # return bboxes_futher, labels_further

    # # ------------------- only single stage nms --------------------
    # max_score = multi_scores.max(1)[0]
    # max_index = multi_scores.max(1)[1]
    # cls_inds = max_score > score_thr
    # total_index = torch.linspace(1, cls_inds.shape[0], steps=cls_inds.shape[0]).int() - 1
    #
    # # if not cls_inds.any():
    # #     continue
    # if multi_bboxes.shape[1] == 4:
    #     _bboxes = multi_bboxes[cls_inds, :]
    # # else:
    # #     _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
    # _scores = max_score[cls_inds]
    # # if score_factors is not None:
    # #     _scores *= score_factors[cls_inds]
    # cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
    #
    # # 将 BBox 和 score 拼接到一起
    # cls_dets_later, cls_dets_later_index = nms_op(cls_dets, **nms_cfg_)
    # cls_labels = max_index[total_index[cls_inds][cls_dets_later_index].long()] - 1
    #
    # bboxes.append(cls_dets_later)
    # labels.append(cls_labels)
    #
    # if bboxes:
    #     bboxes = torch.cat(bboxes)
    #     labels = torch.cat(labels)
    #     if bboxes.shape[0] > max_num:
    #         _, inds = bboxes[:, -1].sort(descending=True)
    #         inds = inds[:max_num]
    #         bboxes = bboxes[inds]
    #         labels = labels[inds]
    # else:
    #     bboxes = multi_bboxes.new_zeros((0, 5))
    #     labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
    # # return bboxes, labels
    # #------------------- only single stage nms  --------------------
    # return bboxes, labels