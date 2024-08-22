import torch.nn.functional as F
import torch
import torch.nn as nn
import functools

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)





################  changed for local, global attention and spatial attention ################################################

class attn_module(nn.Module):
    '''Auxiliary attention module for parallel trainable attention network'''


    def __init__(self, in_ch, out_ch, s1, s2):  #(32->(20,20) for image size 160,(8,8) for image size 60
        self.s1, self.s2 = s1, s2
        super(attn_module, self).__init__()
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (128 * 128) -> (64 * 64)
        self.d1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (64 * 64) -> (32 * 32)
        self.d2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (32 * 32) -> (16 * 16)

        self.mid = nn.Sequential(*[
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        ])

        self.u2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.u1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        
        if self.s2 == (20,20):
            self.last = nn.Sequential(*[
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
                nn.Sigmoid()
            ])
        elif self.s2 == (8,8):
            self.last = nn.Sequential(*[
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 2, 1, bias=False),  #kernel 1-->2
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 1, 1, bias=False),
                nn.Sigmoid()
            ])
            

    def forward(self, x):
#         print('attention module')
#         print('input shape in attention module', x.shape)
    
        out = F.relu(self.d1(self.mp1(x)))
#         print('m1 shape', self.mp1(x).shape)
#         print(' d1 out1', out.shape)
#         print('m2 ', self.mp2(out).shape)
        out = F.relu(self.d2(self.mp2(out)))
        
#         print('d2 out2', out.shape)
        skip2 = F.relu(self.skip2(out))
#         print('skip2', skip2.shape)
        out = self.mid(out)
#         print('mid out3', out.shape)
        out = F.interpolate(out, size=self.s2, mode='bilinear', align_corners=True) + skip2

#         out = F.interpolate(out, size=self.s2, mode='bilinear', align_corners=True) + skip2
            
#         print('out4', out.shape)
        # (14 * 14 -> 28 * 28)
        out = self.last(self.u2(out))
#         print('out5', out.shape)

        return out

class AttnDiscriminator(nn.Module):
    # The size of input is assumed to be 256 * 256
    def __init__(self, in_ch, int_ch,img_size=160, n_layers=3, mask_shape=160, norm_layer=nn.BatchNorm2d, inner_rescale=True,
                 inner_s1=None, inner_s2=None):
        super(AttnDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        if img_size == 160:
            mask_shape = 160
        elif img_size == 60:
            mask_shape = 60

        kw, padw = 4, 1
        self.fe = nn.Sequential(*[
            nn.Conv2d(in_ch, int_ch, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(.2, True)
        ])

        trunk_model = list()
        nf_mult, nf_mult_prev = 1, 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)

            trunk_model += [
                nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(int_ch * nf_mult),
                nn.LeakyReLU(.2, True)
            ]

        self.trunk_brunch = nn.Sequential(*trunk_model)
#         if not inner_s1 and not inner_s2:
#             self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, s1 =(64,64), s2=(8,8))
#         else:
#             self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, inner_s1, inner_s2)

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
        
        if img_size == 160:
            
            if not inner_s1 and not inner_s2:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, s1 =(64,64), s2=(20,20))
            else:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, inner_s1, inner_s2)

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            
            self.fin_phase = nn.Sequential(*[
                nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(int_ch * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(int_ch * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                nn.Flatten(),                                #adding the last layer according to the reconglgan
                nn.Linear(1*18*18,64),
                nn.LeakyReLU(0.2, True)
            ])
            self.inner_rescale = inner_rescale
            self.mask_shape = mask_shape
            
        elif img_size == 60:
            
            if not inner_s1 and not inner_s2:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, s1 =(64,64), s2=(8,8))
            else:
                self.mask_brunch = attn_module(int_ch, int_ch * nf_mult, inner_s1, inner_s2)

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            
            self.fin_phase = nn.Sequential(*[
                nn.Conv2d(int_ch * nf_mult_prev, int_ch * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(int_ch * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(int_ch * nf_mult, 32, kernel_size=kw, stride=1, padding=padw),
                nn.Flatten(),                                #adding the last layer to make it to same size[1,64]
                nn.Linear(1*5*5*32,512),
                nn.LeakyReLU(0.2, True),
                nn.Linear(512,64),
                nn.LeakyReLU(0.2, True)
            ])
            self.inner_rescale = inner_rescale
            self.mask_shape = mask_shape
            

    def forward(self, x):
#         print('input size', x.shape)
        feature = self.fe(x)
#         print('feature', feature.shape)
        trunk = self.trunk_brunch(feature)
#         print('trunk', trunk.shape)
        mask = self.mask_brunch(feature)
#         print('mask', mask.shape)

        expand_mask = self._mask_rescale(mask, self.inner_rescale)
        # print('expand mask', expand_mask.shape)
        
        final = self.fin_phase((mask+1)*trunk)
#         print('final', final.shape)
#         print('mask+1', mask+1)

        return self.fin_phase((mask + 1) * trunk), expand_mask

    def _mask_rescale(self, mask_tensor, final_scale=False):
        mask_tensor = torch.mean(mask_tensor, dim=1).unsqueeze(1)

        if final_scale:
            t_max, t_min = torch.max(mask_tensor), torch.min(mask_tensor)
            mask_tensor = (mask_tensor - t_min) / (t_max - t_min)
            return F.interpolate(mask_tensor, (self.mask_shape, self.mask_shape), mode='bilinear')
        else:
            return mask_tensor

class Concatenate_attn(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate_attn, self).__init__()
        self.dim = dim

    def forward(self, x):
#         print('input shape in concatenate',x.shape)
        return torch.cat(x, dim=self.dim)


class ContextDiscriminator_attention(nn.Module):
    def __init__(self):
        super(ContextDiscriminator_attention, self).__init__()
        self.model_gd = AttnDiscriminator(in_ch =1, int_ch =64, img_size=160)
        self.model_ld = AttnDiscriminator(in_ch = 1, int_ch =64, img_size=60)
        # self.model_ld = NLayerDiscriminator_local(2) #LDiscriminator()  #2 specifies the number of output channels(size=60)
        # self.model_gd = NLayerDiscriminator_global(2) #Discriminator()  size=160
        # output_shape: (None, 1)
        self.concat1 = Concatenate_attn(dim=-1)
        self.linear1 = nn.Linear(128, 1)

    def forward(self, x_local, x_global):
        x_ld, local_attn = self.model_ld(x_local)
        # print('---local shape---', x_ld.shape)
        x_gd, global_attn = self.model_gd(x_global)
        # print('----global shape-----', x_gd.shape)
        x = self.linear1(self.concat1([x_ld, x_gd]))

        #local attn embedding onto gloabl attn
        new_attn = global_attn.detach().cpu().numpy()
        new_attn[:,:,50:110,60:120] = local_attn.detach().cpu().numpy()
        tensor_new_attn = torch.from_numpy(new_attn).to(global_attn.device)
        # print('attn type===========', type(tensor_new_attn))
        
        #intersection of global and local attn maps
        
        # local_attn_pad = F.pad(local_attn, (50,50,50,50),value =0)  #padding attention map from 60 to 160
        # intersection_attn = global_attn*local_attn_pad  #finding the intersection of the local and global attention
        
        return x, tensor_new_attn #new_attn #, intersection_attn


