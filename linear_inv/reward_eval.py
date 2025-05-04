from abc import ABC, abstractmethod
import torch
import numpy as np
from torchvision import transforms


from face_detection import FaceRecognition

from AdaFace import inference
from AdaFace.face_alignment import align, mtcnn
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from typing import Tuple, List

__REWARD_METHOD__ = {}


def register_reward_method(name: str):
    def wrapper(cls):
        if __REWARD_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __REWARD_METHOD__[name] = cls
        return cls
    return wrapper


def get_reward_eval(name: str, **kwargs):
    if __REWARD_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __REWARD_METHOD__[name](**kwargs)


class Reward(ABC):
    """
    Abstract base class for all reward functions used in guided diffusion or sampling.

    Subclasses should implement custom logic to compute a reward signal for a given input
    (e.g., image, text, or other modality) that can be used for steering diffusion sampling
    via gradient-based or search-based methods.

    Note:
        This base class is designed to be flexible for multiple types of guidance:
        - Face recognition similarity
        - Style transfer alignment
        - Text-to-image alignment
        - Any task-specific reward signal

    To implement a custom reward:
        1. Inherit from this class.
        2. Implement the `get_reward` method.
        3. Optionally implement any setup methods like `set_gt_embeddings`.

    Methods:
        get_reward(**kwargs): Abstract method to compute and return a reward score.
    """

    def __init__(self, **kwargs):
        """
        Optional constructor for reward classes. Accepts arbitrary keyword arguments
        for flexibility and downstream configuration.
        """
        pass

    @abstractmethod
    def get_reward(self, particles, **kwargs) -> torch.Tensor:
        """
        Compute and return the reward signal given inputs.

        Args:
            particles: The particles that you want to find the reward for
            **kwargs: Task-specific keyword arguments such as 'images', 'text', etc.

        Returns:
            A torch.Tensor representing the reward(s).
        """
        pass

    @abstractmethod
    def set_ref_embeddings(self, **kwargs):
        pass


@register_reward_method(name='facenet')
class FacenetReward(Reward):
    """
    Reward function based on FaceNet embeddings.

    This class computes a reward signal based on the similarity of face embeddings
    using a pre-trained FaceNet model. The reward is computed as the cosine similarity
    between the input image's embedding and a reference embedding.

    Args:
        ref_embedding (torch.Tensor): Reference embedding for comparison.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get('device', 'cuda')
        self.facenet = FaceRecognition(mtcnn_face=True, norm_order=2).to(self.device)
        self.ref_embd = None

    def set_ref_embeddings(self, ref):
        self.ref_embd = self.facenet(ref)

    def get_reward(self, x):
        """
        Compute the reward based on the cosine similarity of face embeddings.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Reward signal based on cosine similarity.
        """
        # Compute the embedding for the input image

        embd = []

        embd_dim = self.ref_embd.shape[1]
        
        for n, xn in enumerate(x):
            with torch.no_grad():
                zn = self.facenet(xn.unsqueeze(0))

            if zn is not None:
                embd.append(zn.squeeze(0))
            else:
                embd.append(torch.zeros(embd_dim).to(self.device))

        embd = torch.stack(embd, dim=0)  # (1, 512) -> (N, 512)

        difference = embd - self.ref_embd  # (N, 512)
        loss = torch.linalg.norm(difference, dim=-1, ord=2) ** 2
        return -loss


@register_reward_method('adaface')
class AdaFaceReward(Reward):
    """
    Reward function based on AdaFace facial embeddings.

    This class computes the similarity between a generated face image and a reference
    face (additional image of the same person) using embeddings from a pretrained AdaFace model.

    The reward can be used for guiding diffusion models in tasks like face reconstruction,
    identity-preserving generation, and image alignment.

    Attributes:
        files (List[Path]): List of all image file paths in the dataset directory.
        model: Pretrained AdaFace embedding model.
        gt_embeddings (torch.Tensor): Ground-truth face embedding.
        device (str): Computation device, e.g., 'cuda' or 'cpu'.
        mtcnn_model: Face detector and aligner (MTCNN).
        res (int): Target image resolution for preprocessing.
    """
    def __init__(self, pretrained_model: str, resolution: int = 256, device: str = 'cuda:0', **kwargs):
        """
        Initializes the AdaFaceReward class.

        Args:
            data_path (str): Path to the directory containing face images.
            pretrained_model (str): Name of the pretrained AdaFace model to load.
            resolution (int): Target resolution to resize and center crop images.
            device (str): Torch device for inference, default is 'cuda:0'.
            **kwargs: Additional unused keyword arguments.
        """
        super().__init__(**kwargs)
        self.device = device
        self.model = inference.load_pretrained_model(pretrained_model).to(self.device)
        self.mtcnn_model = mtcnn.MTCNN(device=self.device, crop_size=(112, 112))
        self.ref_embd = None
        self.res = resolution
        self.name = 'adaface'

    def get_reward(self, x, **kwargs) -> torch.Tensor:
        """
        Computes the negative L2 distance between the embeddings of given images
        and the stored ground-truth embedding.

        Args:
            images (torch.Tensor): Input batch of images (B, C, H, W) in [-1, 1].

        Returns:
            torch.Tensor: A tensor of shape B containing reward values.
        """
        embd = self._embeddings(x)
        difference = embd - self.ref_embd  # (N, 512)
        loss = torch.linalg.norm(difference, dim=-1, ord=2) ** 2
        return -loss

    def set_ref_embeddings(self, ref) -> None:
        """
        Sets the ground-truth embedding by loading and embedding the additional image
        at the given index in the dataset.

        Args:
            index (int): Index of the reference image in the dataset list.
        """
        
        # # Load and preprocess image
        # img = Image.open(self.files[index])
        # trans = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(self.res),
        #     transforms.CenterCrop(self.res)
        # ])
        # img_tensor = (trans(img) * 2 - 1).to(self.device)
        # if img_tensor.shape[0] == 1:
        #     img_tensor = img_tensor.expand(3, -1, -1)

        # Set gt embedding
        self.ref_embd = self._embeddings(ref).detach()

    def _embeddings(self, tensor_images: torch.Tensor) -> torch.Tensor:
        """
        Computes AdaFace embeddings for a batch of images.

        Each image is aligned using MTCNN and passed through the pretrained model.

        Args:
            tensor_images (torch.Tensor): Batch of images (B, C, H, W) in [-1, 1].

        Returns:
            torch.Tensor: A tensor of shape (B, D) with D-dimensional embeddings.
        """
        tensor_images = ((tensor_images + 1) / 2 * 255).clamp(0, 255).byte()
        to_pil = transforms.ToPILImage()

        aligned_images, failed_indices = [], []
        for i in range(tensor_images.size(0)):
            try:
                img = to_pil(tensor_images[i])
                aligned = align.get_aligned_face('', rgb_pil_image=img)
                aligned_images.append(inference.to_input(aligned).to(self.device))
            except Exception as e:
                print('Error in face alignment at index {0}, adding fallback embedding.'.format(i), flush=True)
                failed_indices.append(i)
                aligned_images.append(torch.randn((1, 3, 112, 112), device=self.device))

        batch_input = torch.cat(aligned_images, dim=0)  # Assuming dim=0 is batch
        embeddings, _ = self.model(batch_input)
        if failed_indices:
            fallback = torch.ones((len(failed_indices), self.ref_embd.shape[1]), device=embeddings.device) * 1e3
            embeddings[torch.tensor(failed_indices, device=embeddings.device)] = fallback

        return embeddings


        