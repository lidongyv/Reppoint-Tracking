    def forward_train(self,
                      imgs,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        all_feat = []
        for img_index in range(0, len(imgs)):
            img = imgs[[img_index]]
            # --- modified ---
            start = time.clock()
            # --- modified ---
            now_img = img[:, :, :384, :]
            former_img = img[:, :, 384:384 * 2, :]
            later_img = img[:, :, 384 * 2:384 * 3, :]
            batch_size, c, im_h_repeat, im_w = img.shape
            im_h = im_h_repeat / 3

            base_feat = self.extract_feat(now_img)
            base_former_feat = self.extract_feat(former_img)
            base_later_feat = self.extract_feat(later_img)

            grid_pixel = set_id_grid(batch_size, int(im_h / 4), int(im_w / 4)).cuda()
            u_grid = grid_pixel[0, 0, :, :].view(-1)
            u_norm = 2 * (u_grid / (int(im_w / 4) -1)) - 1
            v_grid = grid_pixel[0, 1, :, :].view(-1)
            v_norm = 2 * (v_grid / (int(im_h / 4) -1)) - 1
            pixel_coords = torch.stack([u_norm, v_norm], dim=1)
            pixel_coords_base_4 = pixel_coords.view(int(im_h / 4), int(im_w / 4), 2).repeat(batch_size, 1, 1, 1)

            grid_pixel = set_id_grid(batch_size, int(im_h / 8), int(im_w / 8)).cuda()
            u_grid = grid_pixel[0, 0, :, :].view(-1)
            u_norm = 2 * (u_grid / (int(im_w / 8) -1)) - 1
            v_grid = grid_pixel[0, 1, :, :].view(-1)
            v_norm = 2 * (v_grid / (int(im_h / 8) -1)) - 1
            pixel_coords = torch.stack([u_norm, v_norm], dim=1)
            pixel_coords_base_8 = pixel_coords.view(int(im_h / 8), int(im_w / 8), 2).repeat(batch_size, 1, 1, 1)

            grid_pixel = set_id_grid(batch_size, int(im_h / 16), int(im_w / 16)).cuda()
            u_grid = grid_pixel[0, 0, :, :].view(-1)
            u_norm = 2 * (u_grid / (int(im_w / 16) -1)) - 1
            v_grid = grid_pixel[0, 1, :, :].view(-1)
            v_norm = 2 * (v_grid / (int(im_h / 16) -1)) - 1
            pixel_coords = torch.stack([u_norm, v_norm], dim=1)
            pixel_coords_base_16 = pixel_coords.view(int(im_h / 16), int(im_w / 16), 2).repeat(batch_size, 1, 1, 1)

            grid_pixel = set_id_grid(batch_size, int(im_h / 32), int(im_w / 32)).cuda()
            u_grid = grid_pixel[0, 0, :, :].view(-1)
            u_norm = 2 * (u_grid / (int(im_w / 32) -1)) - 1
            v_grid = grid_pixel[0, 1, :, :].view(-1)
            v_norm = 2 * (v_grid / (int(im_h / 32) -1)) - 1
            pixel_coords = torch.stack([u_norm, v_norm], dim=1)
            pixel_coords_base_32 = pixel_coords.view(int(im_h / 32), int(im_w / 32), 2).repeat(batch_size, 1, 1, 1)

            grid_pixel = set_id_grid(batch_size, int(im_h / 64), int(im_w / 64)).cuda()
            u_grid = grid_pixel[0, 0, :, :].view(-1)
            u_norm = 2 * (u_grid / (int(im_w / 64) -1)) - 1
            v_grid = grid_pixel[0, 1, :, :].view(-1)
            v_norm = 2 * (v_grid / (int(im_h / 64) -1)) - 1
            pixel_coords = torch.stack([u_norm, v_norm], dim=1)
            pixel_coords_base_64 = pixel_coords.view(int(im_h / 64), int(im_w / 64), 2).repeat(batch_size, 1, 1, 1)

            grid_pixel = set_id_grid(batch_size, int(im_h / 128), int(im_w / 128)).cuda()
            u_grid = grid_pixel[0, 0, :, :].view(-1)
            u_norm = 2 * (u_grid / (int(im_w / 128) -1)) - 1
            v_grid = grid_pixel[0, 1, :, :].view(-1)
            v_norm = 2 * (v_grid / (int(im_h / 128) -1)) - 1
            pixel_coords = torch.stack([u_norm, v_norm], dim=1)
            pixel_coords_base_128 = pixel_coords.view(int(im_h / 128), int(im_w / 128), 2).repeat(batch_size, 1, 1, 1)

            std_vector = torch.from_numpy(np.array([58.395, 57.12, 57.375])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda().float()
            mean_vector = torch.from_numpy(np.array([123.675, 116.28, 103.53])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda().float()

            now_img = now_img * std_vector + mean_vector
            former_img = former_img * std_vector + mean_vector
            later_img = later_img * std_vector + mean_vector
            # print(now_img.min(), now_img.max())
            # print(former_img.min(), former_img.max())
            # print(later_img.min(), later_img.max())

            # now_img = (now_img - now_img.min()) / (now_img.max() - now_img.min()) * 255
            # former_img = (former_img - former_img.min()) / (former_img.max() - former_img.min()) * 255
            # later_img = (later_img - later_img.min()) / (later_img.max() - later_img.min()) * 255
            former_sample_pair = torch.cat([now_img.unsqueeze(2), former_img.unsqueeze(2)], 2)
            later_sample_pair = torch.cat([now_img.unsqueeze(2), later_img.unsqueeze(2)], 2)
            former_flow = model_and_loss(former_sample_pair).detach()
            later_flow = model_and_loss(later_sample_pair).detach()

            # # for check
            # templete = set_id_grid(batch_size, int(im_h), int(im_w)).cuda()[:, :2, :, :]
            # fore_warp = torch.round(templete + former_flow).long()
            # fore_warp[:, 0, :, :] = torch.clamp(fore_warp[:, 0, :, :], min=0, max=fore_warp.shape[3] - 1)
            # fore_warp[:, 1, :, :] = torch.clamp(fore_warp[:, 1, :, :], min=0, max=fore_warp.shape[2] - 1)
            # print('single stage base 238')
            # from IPython import embed
            # embed()
            # fore_result = torch.where(torch.norm(later_flow[fore_warp[:, 0, :, :], fore_warp[:, 1, :, :]] + former_flow, 2, dim=2) <= 1, \
            #                           self.one, self.zero)

            # # --- 原来大小的 1/16 的图像
            now_img_sample = torch.nn.functional.grid_sample(now_img, pixel_coords_base_16, mode='nearest', padding_mode='zeros')
            former_img_sample = torch.nn.functional.grid_sample(former_img, pixel_coords_base_16, mode='nearest', padding_mode='zeros')
            later_img_sample = torch.nn.functional.grid_sample(later_img, pixel_coords_base_16,  mode='nearest', padding_mode='zeros')

            grid_list = [pixel_coords_base_8, pixel_coords_base_16, pixel_coords_base_32, pixel_coords_base_64, pixel_coords_base_128]
            scale_list = [8, 16, 32, 64, 128]
            aggregated_feat_list = []
            for scale_index, scale_it in enumerate(scale_list):

                former_flow_sample = torch.nn.functional.grid_sample(former_flow, grid_list[scale_index], mode='nearest', padding_mode='zeros')
                former_flow_sample = former_flow_sample / scale_it
                later_flow_sample = torch.nn.functional.grid_sample(later_flow, grid_list[scale_index],  mode='nearest', padding_mode='zeros')
                later_flow_sample = later_flow_sample / scale_it

                temp = 2 * (former_flow_sample) # 这里不要 -1
                u_norm = temp[:, 0, :, :].view(batch_size, -1) / (int(im_w / scale_it) - 1)
                v_norm = temp[:, 1, :, :].view(batch_size, -1) / (int(im_h / scale_it) - 1)
                pixel_coords_bias = torch.stack([u_norm, v_norm], dim=2)
                pixel_coords_bias_view = pixel_coords_bias.view(batch_size, int(im_h / scale_it), int(im_w / scale_it), 2)
                pixel_coords_former2now = grid_list[scale_index] + pixel_coords_bias_view

                base_former_feat_warped = torch.nn.functional.grid_sample(base_former_feat[scale_index], pixel_coords_former2now, padding_mode='zeros')

                # base_former_img_sample_warped = torch.nn.functional.grid_sample(former_img_sample[scale_index], pixel_coords_former2now, padding_mode='zeros')

                temp = 2 * (later_flow_sample)
                u_norm = temp[:, 0, :, :].view(batch_size, -1) / (int(im_w / scale_it) - 1)
                v_norm = temp[:, 1, :, :].view(batch_size, -1) / (int(im_h / scale_it) - 1)
                pixel_coords_bias = torch.stack([u_norm, v_norm], dim=2)
                pixel_coords_bias_view = pixel_coords_bias.view(batch_size, int(im_h / scale_it), int(im_w / scale_it), 2)
                pixel_coords_later2now = grid_list[scale_index] + pixel_coords_bias_view
                # base_later_feat_warped 2 * 512 * 16 * 16
                base_later_feat_warped = torch.nn.functional.grid_sample(base_later_feat[scale_index], pixel_coords_later2now, padding_mode='zeros')

                # base_later_img_sample_warped = torch.nn.functional.grid_sample(later_img_sample[scale_index], pixel_coords_later2now, padding_mode='zeros')

                base_current_feat_warped_embedding = self.get_embednet[scale_index](base_feat[scale_index])  # 1 * 2048 * 16 * 16
                base_former_feat_warped_embedding = self.get_embednet[scale_index](base_former_feat_warped)  # 1 * 2048 * 16 * 16
                base_later_feat_warped_embedding = self.get_embednet[scale_index](base_later_feat_warped)  # 1 * 2048 * 16 * 16

                cos = nn.CosineSimilarity(dim=1, eps=1e-6)

                former_weight = cos(base_current_feat_warped_embedding, base_former_feat_warped_embedding).unsqueeze(1)
                later_weight = cos(base_current_feat_warped_embedding, base_later_feat_warped_embedding).unsqueeze(1)
                former_later_weight = torch.cat([former_weight, later_weight], dim=0)
                former_later_weight_softmax = F.softmax(former_later_weight, dim=0)
                # softmax opperation
                former_weight_tile_softmax = former_later_weight_softmax[[0]].repeat(1, 256, 1, 1)  # repeat in channle
                later_weight_tile_softmax = former_later_weight_softmax[[1]].repeat(1, 256, 1, 1)  # repeat in channle

                aggregated_feat = base_former_feat_warped * former_weight_tile_softmax + base_later_feat_warped * later_weight_tile_softmax
                aggregated_feat_list.append(aggregated_feat)
            a = tuple(aggregated_feat_list)
            all_feat.append(a)


        temp_a = []
        for i in range(0, len(all_feat[0])):
            temp_b = []
            for j in range(0, len(all_feat)):
                temp_b.append(all_feat[j][i])
            temp_a.append(torch.cat(temp_b))


        # # ----- modified -----
        # x = self.extract_feat(img[:, :, :256, :])

        outs = self.bbox_head(tuple(temp_a))

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        end = time.clock()
        # print('total time: %f' % (end - start))
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        im_num = int(img.shape[2] / 384)
        im_h = 384
        im_w = 384 * 4
        batch_size = 1
        std_vector = torch.from_numpy(np.array([58.395, 57.12, 57.375])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda().float()
        mean_vector = torch.from_numpy(np.array([123.675, 116.28, 103.53])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda().float()

        img_list = []
        init_img_list = []
        feat_list = []
        for img_index in range(im_num):
            temp_img = img[:, :, im_h * (img_index):im_h * (img_index + 1), :]  # split each tensor image
            temp_img_feat = self.extract_feat(temp_img)  #
            init_temp_img = temp_img * std_vector + mean_vector

            img_list.append(img[:, :, im_h * (img_index):im_h * (img_index + 1), :])
            feat_list.append(temp_img_feat)
            init_img_list.append(init_temp_img)

        pair_list = []
        flow_list = []
        for pair_index in range(1, im_num):
            temp_pair = torch.cat([init_img_list[0].unsqueeze(2), init_img_list[pair_index].unsqueeze(2)], 2)
            temp_flow = model_and_loss(temp_pair).detach()

            pair_list.append(temp_pair)
            flow_list.append(temp_flow)

        # a = flow_list[0][0].permute(1,2,0)
        # flow_image_former_np = flow_to_image(a)
        # cv2.imwrite('1213_flow_init_former.png', flow_image_former_np)

        scale_list = [8, 16, 32, 64, 128]
        grid_list = []
        for scale_index, scale_it in enumerate(scale_list):
            grid_pixel = set_id_grid(batch_size, int(im_h / scale_it), int(im_w / scale_it)).cuda()
            u_grid = grid_pixel[0, 0, :, :].view(-1)
            u_norm = 2 * (u_grid / (int(im_w / scale_it) -1)) - 1
            v_grid = grid_pixel[0, 1, :, :].view(-1)
            v_norm = 2 * (v_grid / (int(im_h / scale_it) -1)) - 1
            pixel_coords = torch.stack([u_norm, v_norm], dim=1)
            pixel_coords_temp = pixel_coords.view(int(im_h / scale_it), int(im_w / scale_it), 2).repeat(batch_size, 1, 1, 1)
            grid_list.append(pixel_coords_temp)

        aggregated_feat_list = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for scale_index, scale_it in enumerate(scale_list):
            base_current_feat_warped_embedding = self.get_embednet[scale_index](feat_list[0][scale_index])  # 1 * 2048 * 16 * 16
            cos_weight_list = []
            warped_feat_list = []
            for each_flow_index, each_flow in enumerate(flow_list):
                temp_flow_sample = torch.nn.functional.grid_sample(each_flow, grid_list[scale_index], mode='nearest', padding_mode='zeros')
                temp_flow_sample = temp_flow_sample / scale_it
                temp = 2 * (temp_flow_sample) # 这里不要 -1
                u_norm = temp[:, 0, :, :].view(batch_size, -1) / (int(im_w / scale_it) - 1)
                v_norm = temp[:, 1, :, :].view(batch_size, -1) / (int(im_h / scale_it) - 1)
                pixel_coords_bias = torch.stack([u_norm, v_norm], dim=2)
                pixel_coords_bias_view = pixel_coords_bias.view(batch_size, int(im_h / scale_it), int(im_w / scale_it), 2)
                pixel_coords_temp2now = grid_list[scale_index] + pixel_coords_bias_view
                base_temp_feat_warped = torch.nn.functional.grid_sample(feat_list[each_flow_index + 1][scale_index], pixel_coords_temp2now, padding_mode='zeros')
                base_temp_feat_warped_embedding = self.get_embednet[scale_index](base_temp_feat_warped)  # 1 * 2048 * 16 * 16
                temp_weight = cos(base_current_feat_warped_embedding, base_temp_feat_warped_embedding).unsqueeze(1)

                cos_weight_list.append(temp_weight)
                warped_feat_list.append(base_temp_feat_warped)

            weight_cat = torch.cat(cos_weight_list)
            weight_cat_softmax = F.softmax(weight_cat, dim=0)
            softmax_weight_tile_list = []
            for softmax_weight_index in range(weight_cat_softmax.shape[0]):
                temp_weight_tile_softmax = weight_cat_softmax[softmax_weight_index].repeat(1, 256, 1, 1)
                softmax_weight_tile_list.append(temp_weight_tile_softmax)

            weighted_feat_list = []
            for to_aggregate_index in range(len(softmax_weight_tile_list)):
                weighted_feat_list.append(warped_feat_list[to_aggregate_index] * softmax_weight_tile_list[to_aggregate_index])
            aggregated_feat_list.append(torch.cat(weighted_feat_list).sum(0).unsqueeze(0))
        x = tuple(aggregated_feat_list)

        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
