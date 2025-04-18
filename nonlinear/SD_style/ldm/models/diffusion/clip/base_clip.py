import torch
import torch.nn as nn
from .clip import clip
# from clip import clip
import torchvision
from PIL import Image

model_name = "ViT-B/16"
# model_name = "ViT-B/32"


def load_clip_to_cpu():
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CLIPEncoder(nn.Module):
    def __init__(self, need_ref=False, ref_path=None):
        super().__init__()
        self.clip_model = load_clip_to_cpu()
        self.clip_model.requires_grad = True
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
            (0.26862954*2, 0.26130258*2, 0.27577711*2)
        )
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        if need_ref:
            img = Image.open(ref_path).convert('RGB')
            image = img.resize((224, 224), Image.Resampling.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            self.ref = img

    def get_residual(self, image, text):
        text = clip.tokenize(text).cuda()
        image = torch.nn.functional.interpolate(image, size=224, mode='bicubic')
        image = self.preprocess(image)
        image_feature, _ = self.clip_model.encode_image_with_features(image)
        text_feature = self.clip_model.encode_text(text)
        text_feature = text_feature.repeat(image.shape[0], 1)
        return text_feature - image_feature


    def get_clip_score_manual(self, image, text, normalize=True):
        text = clip.tokenize(text).cuda()
        image = torch.nn.functional.interpolate(image, size=224, mode='bicubic')
        image = self.preprocess(image)
        image_feature, _ = self.clip_model.encode_image_with_features(image)  # (b, d) with d=512
        text_feature = self.clip_model.encode_text(text)  # (d,) with d=512
        text_feature = text_feature.repeat(image.shape[0], 1)  # (b, d) with d=512
        # compute inner product
        if normalize:
            normalized_image_feature = image_feature / torch.norm(image_feature, dim=1, keepdim=True)  # (b, d)
            normalized_text_feature = text_feature / torch.norm(text_feature, dim=1, keepdim=True)  # (b, d)
            normalized_clip_score = torch.bmm(normalized_image_feature.unsqueeze(1), normalized_text_feature.unsqueeze(2))  # (b, 1, d) * (b, d, 1) = (b, 1, 1)
            normalized_clip_score = normalized_clip_score.squeeze(1).squeeze(1)  # (b,)
            return normalized_clip_score
        else:
            clip_score = torch.bmm(image_feature.unsqueeze(1), text_feature.unsqueeze(2))  # (b, 1, d) * (b, d, 1) = (b, 1, 1)
            clip_score = clip_score.squeeze(1).squeeze(1)  # (b,)
            return clip_score

    
    def calc_ref_feat(self, ref_path):
        img = Image.open(ref_path).convert('RGB')
        image = img.resize((224, 224), Image.Resampling.BILINEAR)
        img = self.to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        self.ref = img

    
    # def get_gram_matrix_residual(self, im1):
    #     im1 = torch.nn.functional.interpolate(im1, size=(224, 224), mode='bicubic')
    #     im1 = self.preprocess(im1)

    #     f1, feats1 = self.clip_model.encode_image_with_features(im1)
    #     f2, feats2 = self.clip_model.encode_image_with_features(self.ref)
        
    #     feat1 = feats1[2][1:, 0, :]
    #     feat2 = feats2[2][1:, 0, :]
    #     gram1 = torch.mm(feat1.t(), feat1)
    #     gram2 = torch.mm(feat2.t(), feat2)
    #     return gram1 - gram2


    def get_gram_matrix_residual(self, im1):
        im1 = torch.nn.functional.interpolate(im1, size=(224, 224), mode='bicubic')
        im1 = self.preprocess(im1)

        f1, feats1 = self.clip_model.encode_image_with_features(im1)
        f2, feats2 = self.clip_model.encode_image_with_features(self.ref)
        
        feat1 = feats1[2][1:, :, :]
        feat2 = feats2[2][1:, :, :]

        feat1_mat = feat1.permute(1, 0, 2)
        feat2_mat = feat2.permute(1, 0, 2)

        feat1_mat_t = feat1_mat.permute(0, 2, 1)
        feat2_mat_t = feat2_mat.permute(0, 2, 1)

        # multiply the two matrices broadcasting the dimension
        gram1 = torch.bmm(feat1_mat_t, feat1_mat)
        gram2 = torch.bmm(feat2_mat_t, feat2_mat)
        return gram1 - gram2


    def get_gram_matrix_residual_im1_im2(self, im1, im2):
        im1 = torch.nn.functional.interpolate(im1, size=(224, 224), mode='bicubic')
        im1 = self.preprocess(im1)

        f1, feats1 = self.clip_model.encode_image_with_features(im1)
        f2, feats2 = self.clip_model.encode_image_with_features(im2)
        
        feat1 = feats1[2][1:, :, :]
        feat2 = feats2[2][1:, :, :]

        feat1_mat = feat1.permute(1, 0, 2)
        feat2_mat = feat2.permute(1, 0, 2)

        feat1_mat_t = feat1_mat.permute(0, 2, 1)
        feat2_mat_t = feat2_mat.permute(0, 2, 1)

        # multiply the two matrices broadcasting the dimension
        gram1 = torch.bmm(feat1_mat_t, feat1_mat)
        gram2 = torch.bmm(feat2_mat_t, feat2_mat)
        return gram1 - gram2
    
    def get_clip_score(self, image, text):
        return self.clip_model(image, text)



if __name__ == "__main__":
    m = CLIPEncoder().cuda()
    im1 = torch.randn((1, 3, 224, 224)).cuda()
    im2 = torch.randn((2, 3, 224, 224)).cuda()
    im3 = torch.zeros_like(im2).cuda()
    gram_res = m.get_gram_matrix_residual_im1_im2(im1, im2)
    print('gram res shape: ', gram_res.shape)

    text = 'random noise'
    text_res_1 = m.get_residual(im1, text)
    print('text res 1 shape: ', text_res_1.shape)

    text_res_2 = m.get_residual(im2, text)
    print('text res 2 shape: ', text_res_2.shape)

    clip_score_2, normalized_clip_score_2 = m.get_clip_score_manual(image=im2, text='a black image')
    print('clip score 2: ', clip_score_2)
    print('normalized clip score 2 (random noise): ', normalized_clip_score_2)

    clip_score_3, normalized_clip_score_3 = m.get_clip_score_manual(image=im3, text='a black image')
    print('clip score 3: ', clip_score_3)
    print('normalized clip score 3 (black): ', normalized_clip_score_3)

