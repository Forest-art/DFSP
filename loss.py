from torch.nn.modules.loss import CrossEntropyLoss


def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    logits, logits_att, logits_obj, logits_soft_prompt = predict
    loss_logit_df = loss_fn(logits, batch_target)
    loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)
    loss_att = loss_fn(logits_att, batch_attr)
    loss_obj = loss_fn(logits_obj, batch_obj)
    loss = loss_logit_df + config.att_obj_w * (loss_att + loss_obj) + config.sp_w * loss_logit_sp
    return loss