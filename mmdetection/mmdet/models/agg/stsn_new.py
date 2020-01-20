
@AGG.register_module
class STSN_fuse_class(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_fuse_class,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)

        self.relu=nn.LeakyReLU(inplace=True)
        self.offset=[]
        self.mask=[]
        print('init transform kernel')
        self.trans_kernel=torch.from_numpy(np.load('/home/ld/RepPoints/mmdetection/mmdet/ops/dcn/init_kernel.npy'))
        self.trans_kernel=nn.Parameter(self.trans_kernel)
        # self.dcn_kernel = 3
        # self.dcn_pad = 1
        # dcn_base = np.arange(-self.dcn_pad,
        #                      self.dcn_pad + 1).astype(np.float64)
        # dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        # dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        # dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
        #     (-1))
        # self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self.offset_base=torch.from_numpy(np.array([-1, -1, -1,  0, -1,  1,  0, -1,  0,  0,  0,  1,  1, -1,  1,  0,  1, 1])).view(1, -1, 1, 1)
        self.one=torch.ones(1)
        self.zero=torch.zeros(1)
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv11_offset, std=0.01)
        normal_init(self.conv11, std=0.01)
        normal_init(self.conv12_offset, std=0.01)
        normal_init(self.conv12, std=0.01)
        normal_init(self.conv13_offset, std=0.01)
        normal_init(self.conv13, std=0.01)
        normal_init(self.conv14_offset, std=0.01)
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,score_refer,score_support,test=False):
        self.grid_x=torch.arange(0,reference.shape[-2]).view(1,-1,1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        self.grid_y=torch.arange(0,reference.shape[-1]).view(1,1,-1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv11_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv11(feature_f0, offset, mask)

            offset_mask2 = self.conv12_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv12(out, offset, mask)

            offset_mask3 = self.conv13_offset(out)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv13(out, offset, mask)

            offset_mask4 = self.conv14_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # print('mask weight',torch.max(mask,dim=1)[0].mean().item())

            loss=0
            offset2center=offset+self.offset_base
            for i in range(9):
                pos_x=offset2center[:,2*i,:,:]+self.grid_x
                pos_y=offset2center[:,2*i+1,:,:]+self.grid_y
                # pos_x=self.grid_x
                # pos_y=self.grid_y
                pos_x=pos_x.unsqueeze(-1).clamp(min=0,max=reference.shape[-2])/reference.shape[-2]-0.5
                pos_y=pos_y.unsqueeze(-1).clamp(min=0,max=reference.shape[-1])/reference.shape[-1]-0.5
                grid=torch.cat([pos_x,pos_y],dim=-1)
                # print(grid)
                score_warp=grid_sample(score_support,grid,mode='linearized').clamp(min=0,max=1)
                # feature_warp=grid_sample(support,grid,mode='linearized')
                class_check=torch.where(score_warp>0.05,self.one,self.zero)*torch.where(score_refer>0.05,self.one,self.zero)
                score_check=torch.where(score_warp.sum(1)>0.05,self.one,self.zero)*torch.where(score_refer.sum(1)>0.05,self.one,self.zero)
                better_check=torch.where(score_warp>score_refer,self.one,self.zero)
                # score_warp=grid_sample(score_support,grid,mode='linearized').clamp(min=0,max=1)
                # print(torch.max(score_support),torch.max(score_warp))
                # support_class=torch.max(score_warp,dim=1)
                if i==0 and score_support.device==torch.ones(1).cuda(0).device:
                    print('max support score',torch.max(score_support).item(), \
                        'max warp score',torch.max(score_warp).item())
                    print('class check',class_check.sum().item(),'score check',score_check.sum().item(), \
                        'betterclass check',(better_check*class_check).sum().item())
                    print('warp result',(score_warp[0]>0.05).float().sum().item(), \
                        'detection result',(score_refer[0]>0.05).float().sum().item())
                mask_loss=F.binary_cross_entropy(mask[:,i,:,:].sigmoid(),class_check.max(1)[0])
                # print('mask loss',mask_loss)
                mask[:,i,:,:]=torch.where((class_check*better_check).sum(1)>0,self.one,self.zero)*mask[:,i,:,:]
                loss+=((torch.where(score_refer>0.05,self.one,self.zero).float()*(score_refer-score_warp).clamp(min=0)).sum() \
                    /(torch.where(score_refer>0.05,self.one,self.zero).float().sum()+1) \
                    +(torch.where(score_refer<0.05,self.one,self.zero).float()*score_warp.clamp(min=0)).sum() \
                    /(torch.where(score_refer<0.05,self.one,self.zero).float().sum()+1)+mask_loss)/3
                # class_check=refer_class[1]==support_class[1]
                # print('same class',torch.sum((refer_class[1]==support_class[1]).float()))
                # print('better score',torch.sum((refer_class[0]<support_class[0]).float()))
            # print(refer_class[0].shape,refer_class[1].shape)
            # print(grid.shape,score_refer.shape,score_support.shape,score_warp.shape)
            # exit()
            
            mask=torch.where(mask==self.zero,-1e3*self.one,mask).sigmoid()*mask
            mask_softmax=F.softmax(torch.where(mask<0.1*self.one,-1e3*self.one,mask),dim=1)
            mask=torch.where(mask==self.zero,mask,mask_softmax)
            self.mask_weight=torch.sum(mask).item()/mask.shape[0]
            kernel_weight=self.trans_kernel.detach()*9
            self.conv14.weight=nn.Parameter(kernel_weight)
            out = self.conv14(support, offset, mask)
            mask_weight=torch.where(mask.sum(dim=1)>self.zero,self.one,self.zero)
            if test:
                return out,offset,mask
            else:
                return out,loss/9,mask_weight
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            offset1=self.conv11_offset(feature_f0)

            feature_f1=self.conv11(feature_f0,offset1)

            offset2=self.conv12_offset(feature_f1)

            feature_f2=self.conv12(feature_f1,offset2)

            offset3=self.conv13_offset(feature_f2)

            feature_f3=self.conv13(feature_f2,offset3)

            offset4=self.conv14_offset(feature_f3)
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(support,offset4)

        if test:
            return agg_features,offset4
        else:
            return agg_features


    def agg2(self,support,reference,score_refer,score_support,test=False):

        self.grid_x=torch.arange(0,reference.shape[-2]).view(1,-1,1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        self.grid_y=torch.arange(0,reference.shape[-1]).view(1,1,-1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv21_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv21(feature_f0, offset, mask)

            offset_mask2 = self.conv22_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv22(out, offset, mask)

            offset_mask4 = self.conv23_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]

            offset2center=offset+self.offset_base
            loss=0
            for i in range(9):
                pos_x=offset2center[:,2*i,:,:]+self.grid_x
                pos_y=offset2center[:,2*i+1,:,:]+self.grid_y
                pos_x=pos_x.unsqueeze(-1).clamp(min=0,max=reference.shape[-2])/reference.shape[-2]-0.5
                pos_y=pos_y.unsqueeze(-1).clamp(min=0,max=reference.shape[-1])/reference.shape[-1]-0.5
                grid=torch.cat([pos_x,pos_y],dim=-1)
                score_warp=grid_sample(score_support,grid,mode='linearized').clamp(min=0,max=1)
                # feature_warp=grid_sample(support,grid,mode='linearized')
                class_check=torch.where(score_warp>0.05,self.one,self.zero)*torch.where(score_refer>0.05,self.one,self.zero)
                score_check=torch.where(score_warp.sum(1)>0.05,self.one,self.zero)*torch.where(score_refer.sum(1)>0.05,self.one,self.zero)
                better_check=torch.where(score_warp>score_refer,self.one,self.zero)
                mask_loss=F.binary_cross_entropy(mask[:,i,:,:].sigmoid(),class_check.max(1)[0])
                mask[:,i,:,:]=torch.where((class_check*better_check).sum(1)>0,self.one,self.zero)*mask[:,i,:,:]
                loss+=((torch.where(score_refer>0.05,self.one,self.zero).float()*(score_refer-score_warp).clamp(min=0)).sum() \
                    /(torch.where(score_refer>0.05,self.one,self.zero).float().sum()+1) \
                    +(torch.where(score_refer<0.05,self.one,self.zero).float()*score_warp.clamp(min=0)).sum() \
                    /(torch.where(score_refer<0.05,self.one,self.zero).float().sum()+1)+mask_loss)/3
            mask=torch.where(mask==self.zero,-1e3*self.one,mask).sigmoid()*mask
            mask_softmax=F.softmax(torch.where(mask<0.1*self.one,-1e3*self.one,mask),dim=1)
            mask=torch.where(mask==self.zero,mask,mask_softmax)
            self.mask_weight=torch.max(mask,dim=1)[0].mean().item()
            kernel_weight=self.trans_kernel.detach()*9
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv23(support, offset, mask)
            mask_weight=torch.where(mask.sum(dim=1)>self.zero,self.one,self.zero)
            if test:
                return out,offset,mask
            else:
                return out,loss/9,mask_weight
        else:
            offset1=self.conv21_offset(feature_f0)

            feature_f1=self.conv21(feature_f0,offset1)

            offset2=self.conv22_offset(feature_f1)

            feature_f2=self.conv22(feature_f1,offset2)

            offset3=self.conv23_offset(feature_f2)
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg3(self,support,reference,score_refer,score_support,test=False):

        self.grid_x=torch.arange(0,reference.shape[-2]).view(1,-1,1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        self.grid_y=torch.arange(0,reference.shape[-1]).view(1,1,-1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv31_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv31(feature_f0, offset, mask)

            offset_mask2 = self.conv32_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv32(out, offset, mask)

            offset_mask4 = self.conv33_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            offset2center=offset+self.offset_base
            loss=0
            for i in range(9):
                pos_x=offset2center[:,2*i,:,:]+self.grid_x
                pos_y=offset2center[:,2*i+1,:,:]+self.grid_y
                pos_x=pos_x.unsqueeze(-1).clamp(min=0,max=reference.shape[-2])/reference.shape[-2]-0.5
                pos_y=pos_y.unsqueeze(-1).clamp(min=0,max=reference.shape[-1])/reference.shape[-1]-0.5
                grid=torch.cat([pos_x,pos_y],dim=-1)
                score_warp=grid_sample(score_support,grid,mode='linearized').clamp(min=0,max=1)
                # feature_warp=grid_sample(support,grid,mode='linearized')
                class_check=torch.where(score_warp>0.05,self.one,self.zero)*torch.where(score_refer>0.05,self.one,self.zero)
                score_check=torch.where(score_warp.sum(1)>0.05,self.one,self.zero)*torch.where(score_refer.sum(1)>0.05,self.one,self.zero)
                better_check=torch.where(score_warp>score_refer,self.one,self.zero)
                mask_loss=F.binary_cross_entropy(mask[:,i,:,:].sigmoid(),class_check.max(1)[0])
                mask[:,i,:,:]=torch.where((class_check*better_check).sum(1)>0,self.one,self.zero)*mask[:,i,:,:]
                loss+=((torch.where(score_refer>0.05,self.one,self.zero).float()*(score_refer-score_warp).clamp(min=0)).sum() \
                    /(torch.where(score_refer>0.05,self.one,self.zero).float().sum()+1) \
                    +(torch.where(score_refer<0.05,self.one,self.zero).float()*score_warp.clamp(min=0)).sum() \
                    /(torch.where(score_refer<0.05,self.one,self.zero).float().sum()+1)+mask_loss)/3
            mask=torch.where(mask==self.zero,-1e3*self.one,mask).sigmoid()*mask
            mask_softmax=F.softmax(torch.where(mask<0.1*self.one,-1e3*self.one,mask),dim=1)
            mask=torch.where(mask==self.zero,mask,mask_softmax)
            self.mask_weight=torch.max(mask,dim=1)[0].mean().item()
            kernel_weight=self.trans_kernel.detach()*9
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv33(support, offset, mask)
            mask_weight=torch.where(mask.sum(dim=1)>self.zero,self.one,self.zero)
            if test:
                return out,offset,mask
            else:
                return out,loss/9,mask_weight
        else:
            offset1=self.conv31_offset(feature_f0)

            feature_f1=self.conv31(feature_f0,offset1)

            offset2=self.conv32_offset(feature_f1)

            feature_f2=self.conv32(feature_f1,offset2)

            offset3=self.conv33_offset(feature_f2)
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg4(self,support,reference,score_refer,score_support,test=False):
        

        self.grid_x=torch.arange(0,reference.shape[-2]).view(1,-1,1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        self.grid_y=torch.arange(0,reference.shape[-1]).view(1,1,-1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        feature_f0=torch.cat([support,reference],dim=1)


        if self.with_modulated_dcn:
            offset_mask1 = self.conv41_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv41(feature_f0, offset, mask)

            offset_mask4 = self.conv42_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            offset2center=offset+self.offset_base
            loss=0
            for i in range(9):
                pos_x=offset2center[:,2*i,:,:]+self.grid_x
                pos_y=offset2center[:,2*i+1,:,:]+self.grid_y
                pos_x=pos_x.unsqueeze(-1).clamp(min=0,max=reference.shape[-2])/reference.shape[-2]-0.5
                pos_y=pos_y.unsqueeze(-1).clamp(min=0,max=reference.shape[-1])/reference.shape[-1]-0.5
                grid=torch.cat([pos_x,pos_y],dim=-1)
                score_warp=grid_sample(score_support,grid,mode='linearized').clamp(min=0,max=1)
                # feature_warp=grid_sample(support,grid,mode='linearized')
                class_check=torch.where(score_warp>0.05,self.one,self.zero)*torch.where(score_refer>0.05,self.one,self.zero)
                score_check=torch.where(score_warp.sum(1)>0.05,self.one,self.zero)*torch.where(score_refer.sum(1)>0.05,self.one,self.zero)
                better_check=torch.where(score_warp>score_refer,self.one,self.zero)
                mask_loss=F.binary_cross_entropy(mask[:,i,:,:].sigmoid(),class_check.max(1)[0])
                mask[:,i,:,:]=torch.where((class_check*better_check).sum(1)>0,self.one,self.zero)*mask[:,i,:,:]
                loss+=((torch.where(score_refer>0.05,self.one,self.zero).float()*(score_refer-score_warp).clamp(min=0)).sum() \
                    /(torch.where(score_refer>0.05,self.one,self.zero).float().sum()+1) \
                    +(torch.where(score_refer<0.05,self.one,self.zero).float()*score_warp.clamp(min=0)).sum() \
                    /(torch.where(score_refer<0.05,self.one,self.zero).float().sum()+1)+mask_loss)/3
            mask=torch.where(mask==self.zero,-1e3*self.one,mask).sigmoid()*mask
            mask_softmax=F.softmax(torch.where(mask<0.1*self.one,-1e3*self.one,mask),dim=1)
            mask=torch.where(mask==self.zero,mask,mask_softmax)
            self.mask_weight=torch.max(mask,dim=1)[0].mean().item()
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv42.weight=nn.Parameter(kernel_weight)
            mask_weight=torch.where(mask.sum(dim=1)>self.zero,self.one,self.zero)
            out = self.conv42(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out,loss/9,mask_weight
        else:
            offset1=self.conv41_offset(feature_f0)

            feature_f1=self.conv41(feature_f0,offset1)

            offset2=self.conv42_offset(feature_f1)
            self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv42(support,offset2)

        if test:
            return agg_features,offset2
        else:
            return agg_features
    def agg5(self,support,reference,score_refer,score_support,test=False):
        

        self.grid_x=torch.arange(0,reference.shape[-2]).view(1,-1,1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        self.grid_y=torch.arange(0,reference.shape[-1]).view(1,1,-1).expand(reference.shape[0],reference.shape[-2],reference.shape[-1]).to(reference.device).float()
        feature_f0=torch.cat([support,reference],dim=1)


        if self.with_modulated_dcn:
            offset_mask1 = self.conv51_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv51(feature_f0, offset, mask)

            offset_mask4 = self.conv52_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            offset2center=offset+self.offset_base
            loss=0
            for i in range(9):
                pos_x=offset2center[:,2*i,:,:]+self.grid_x
                pos_y=offset2center[:,2*i+1,:,:]+self.grid_y
                pos_x=pos_x.unsqueeze(-1).clamp(min=0,max=reference.shape[-2])/reference.shape[-2]-0.5
                pos_y=pos_y.unsqueeze(-1).clamp(min=0,max=reference.shape[-1])/reference.shape[-1]-0.5
                grid=torch.cat([pos_x,pos_y],dim=-1)
                score_warp=grid_sample(score_support,grid,mode='linearized').clamp(min=0,max=1)
                # feature_warp=grid_sample(support,grid,mode='linearized')
                class_check=torch.where(score_warp>0.05,self.one,self.zero)*torch.where(score_refer>0.05,self.one,self.zero)
                score_check=torch.where(score_warp.sum(1)>0.05,self.one,self.zero)*torch.where(score_refer.sum(1)>0.05,self.one,self.zero)
                better_check=torch.where(score_warp>score_refer,self.one,self.zero)
                mask_loss=F.binary_cross_entropy(mask[:,i,:,:].sigmoid(),class_check.max(1)[0])
                mask[:,i,:,:]=torch.where((class_check*better_check).sum(1)>0,self.one,self.zero)*mask[:,i,:,:]
                loss+=((torch.where(score_refer>0.05,self.one,self.zero).float()*(score_refer-score_warp).clamp(min=0)).sum() \
                    /(torch.where(score_refer>0.05,self.one,self.zero).float().sum()+1) \
                    +(torch.where(score_refer<0.05,self.one,self.zero).float()*score_warp.clamp(min=0)).sum() \
                    /(torch.where(score_refer<0.05,self.one,self.zero).float().sum()+1)+mask_loss)/3
            mask=torch.where(mask==self.zero,-1e3*self.one,mask).sigmoid()*mask
            mask_softmax=F.softmax(torch.where(mask<0.1*self.one,-1e3*self.one,mask),dim=1)
            mask=torch.where(mask==self.zero,mask,mask_softmax)
            self.mask_weight=torch.max(mask,dim=1)[0].mean().item()
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv52.weight=nn.Parameter(kernel_weight)
            mask_weight=torch.where(mask.sum(dim=1)>self.zero,self.one,self.zero)
            out = self.conv52(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out,loss/9,mask_weight
        else:
            offset1=self.conv51_offset(feature_f0)

            feature_f1=self.conv51(feature_f0,offset1)

            offset2=self.conv52_offset(feature_f1)
            self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv52(support,offset2)

        if test:
            return agg_features,offset2
        else:
            return agg_features
    def forward(self,datas,class_score,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        self.one=self.one.to(datas[0].device)
        self.zero=self.zero.to(datas[0].device)
        self.offset_base=self.offset_base.to(datas[0].device).float()
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        if datas[0].device=='cuda:0':
            print('fuse channel')
        refer_out=[]
        agg_out=[]
        support_out=[]
        support_count=4
        for i in range(support_count):
            support_out.append([])
        out=[]
        # for i in range(len(datas)):
        #     print(class_score[i].shape,datas[i].shape)
        losses=0
        for i in range(len(datas)):
            reference=datas[i]+0
            score_refer=class_score[i].sigmoid()+0
            refer_out.append(reference)
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            feature=reference.unsqueeze(1)
            for j in range(support_count):
                shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
                shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
                support=datas[i][shuffle_id,:,:,:]+0
                score_support=class_score[i][shuffle_id,:,:,:].sigmoid()+0
                # print(reference.shape,score_support.shape)
                tk_feature,loss,mask_weight=self.agg[i](support,reference,score_refer,score_support)
                # print('loss check',loss.item(),i,j)
                weight=(torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1)*mask_weight).unsqueeze(1).unsqueeze(1)
                    
                
                weight0=torch.cat([weight0,weight],dim=1)
                feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
                support_out[j].append(tk_feature)

                losses+=loss
            weight0=torch.where(weight0==self.zero,-1e-3*self.one,weight0)
            weight=torch.nn.functional.softmax(weight0,dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_out.append(agg_feature)
            if i==0 and weight.device==torch.ones(1).cuda(0).device:
                print('max_weight:',torch.max(weight).item(),'mean_max_weight:',torch.max(weight,dim=1)[0].mean().item())
                print('mask weight',self.mask_weight)
        losses=losses/len(datas)/support_count
        for i in range(support_count):
            support_out[i]=tuple(support_out[i])
        out=[tuple(refer_out),tuple(agg_out)]+support_out

        return out,losses
    def fuse(self,datas,score,test=False):
        reference=datas[0]
        support=datas[2:]
        score_refer=score[0]
        score_support=score[1:]
        agg_out=[]
        for j in range(len(support[0])):
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference[j],reference[j],dim=1).unsqueeze(1).unsqueeze(1))
            feature=reference[j].unsqueeze(1)
            for i in range(len(support)):
                class_check=(score_refer[j].sigmoid()>0.05).float()*(score_support[i][j].sigmoid()>0.05).float()
                better_check=(score_support[i][j].sigmoid()>score_refer[j].sigmoid()).float()
                score_weight=((class_check*better_check).sum(1)>0).float()
                weight=(torch.nn.functional.cosine_similarity(reference[j],support[i][j],dim=1)*score_weight).unsqueeze(1).unsqueeze(1)
                weight0=torch.cat([weight0,weight],dim=1)
                feature=torch.cat([feature,support[i][j].unsqueeze(1)],dim=1)
            weight=torch.nn.functional.softmax(weight0,dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_out.append(agg_feature)
        return tuple(agg_out)
    def forward_test(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn test')
        self.offset=[]
        self.mask=[]
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0

            tk_feature,soffset4,mask4=self.agg[i](support,reference,test)
            self.offset.append(soffset4)
            self.mask.append(mask4)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)
    def forward_eval(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn eval')
        self.offset=[]
        self.mask=[]
        refer_out=[]
        agg_out=[]
        support_out=[]
        self.offset=[]
        self.mask=[]
        for i in range(datas[0].shape[0]-1):
            support_out.append([])
            self.offset.append([])
            self.mask.append([])
        out=[]
        

        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            refer_out.append(reference)
            if datas[i].shape[0]>1:
                support=datas[i][1:,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][:1,:,:,:]+0
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            feature=reference.unsqueeze(1)
            for j in range(support.shape[0]):
                tk_feature,offset,mask=self.agg[i](support[j:j+1,...],reference,test)
                weight=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
                weight0=torch.cat([weight0,weight],dim=1)
                feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
                support_out[j].append(tk_feature)
                self.offset[j].append(offset)
                self.mask[j].append(mask)
            weight=torch.nn.functional.softmax(weight0,dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_out.append(agg_feature)
        for i in range(datas[0].shape[0]-1):
            support_out[i]=tuple(support_out[i])
        out=[tuple(refer_out),tuple(agg_out)]+support_out

        return out
