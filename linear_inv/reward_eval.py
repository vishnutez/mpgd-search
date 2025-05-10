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

class Reward:
    """
        Evaluation module for computing evaluation metrics.
    """

    def __init__(self, reward_fn_list, config_list):
        """
            Initializes the evaluator with the ground truth and measurement.

            Parameters:
                eval_fn_list (tuple): List of evaluation functions to use.
        """
        super().__init__()
        self.reward_fn = {}
        for reward_fn in reward_fn_list:
            self.reward_fn[reward_fn.name] = reward_fn(**config_list[reward_fn.name])

    def set_ref_embeddings(self, ref, **kwargs):
        for reward_fn_name, reward_fn in self.reward_fn.items():
            if hasattr(reward_fn, 'set_ref_embeddings'):
                reward_fn.set_ref_embeddings(ref, **kwargs)  # set ref embeddings


    def __call__(self, x):
        """
            Computes evaluation metrics for the given input.

            Parameters:
                x (torch.Tensor): Input tensor.
                reduction (str): Reduction method ('mean' or 'none').

            Returns:
                dict: Dictionary of evaluation results.
        """
        reward = 0
        for reward_fn_name, reward_fn in self.reward_fn.items():
            if hasattr(reward_fn, 'get_reward'):
                reward += reward_fn.scale * reward_fn.get_reward(x)
        return reward


class RewardFn(ABC):
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
    def get_reward(self, **kwargs) -> torch.Tensor:
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


@register_reward_method(name='measurement')
class Measurement(RewardFn):
    """
    Reward function based on measurement similarity.

    This class computes a reward signal based on the similarity of measurements
    between the input and a reference measurement. The reward is computed as the
    negative L2 distance between the two measurements.

    Args:
        ref_measurement (torch.Tensor): Reference measurement for comparison.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get('device', 'cuda')
        self.scale = kwargs.get('scale', 1.0)

    def set_ref_embeddings(self, measurement, operator, **kwargs):
        self.measurement = measurement
        self.operator = operator

    def get_reward(self, x):
        """
        Compute the reward based on the negative L2 distance to the reference measurement.

        Args:
            x (torch.Tensor): Input measurement tensor.

        Returns:
            torch.Tensor: Reward signal based on negative L2 distance.
        """
        difference = self.operator.forward(x) - self.measurement
        difference_vec = difference.reshape(x.shape[0], -1)
        loss = torch.linalg.norm(difference_vec, axis=-1, ord=2) ** 2
        return -loss
    

@register_reward_method(name='genie')
class GenieReward(RewardFn):
    """
    Reward function based on Genie embeddings.

    This class computes a reward signal based on the similarity of Genie embeddings
    between the input and a reference measurement. The reward is computed as the
    negative L2 distance between the two measurements.

    Args:
        ref_measurement (torch.Tensor): Reference measurement for comparison.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get('device', 'cuda')
        self.scale = kwargs.get('scale', 1.0)

    def set_ref_embeddings(self, ref, **kwargs):
        self.ref = ref

    def get_reward(self, x):
        """
        Compute the reward based on the negative L2 distance to the reference measurement.

        Args:
            x (torch.Tensor): Input measurement tensor.

        Returns:
            torch.Tensor: Reward signal based on negative L2 distance.
        """
        difference = self.ref - x
        difference_vec = difference.reshape(x.shape[0], -1)
        loss = torch.linalg.norm(difference_vec, axis=-1, ord=2) ** 2
        return -loss


@register_reward_method(name='facenet')
class FacenetReward(RewardFn):
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
        self.scale = kwargs.get('scale', 1.0)

    def set_ref_embeddings(self, ref, **kwargs):
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
class AdaFaceReward(RewardFn):
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
    def __init__(self, pretrained_model: str = 'ir_50', resolution: int = 256, device: str = 'cuda:0', **kwargs):
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
        self.scale = kwargs.get('scale', 1.0)
        self.gradient = kwargs.get('gradient', False)
        

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
    

    def get_reward_and_gradient(self, x, **kwargs):
        """
        Computes the loss between the generated image and the ground truth.

        Args:
            x (torch.Tensor): Generated image tensor.

        Returns:
            torch.Tensor: Computed reward value.
            torch.Tensor: Gradient of the reward with respect to the input image.
        """

        x.requires_grad_()        
        reward = self.get_reward(x, **kwargs)
        grad = torch.autograd.grad(reward.sum(), x)[0]
        x.detach_()
        return reward, grad
    

    def set_ref_embeddings(self, ref, **kwargs) -> None:
        """
        Sets the ground-truth embedding by loading and embedding the additional image
        at the given index in the dataset.

        Args:
            ref (torch.Tensor): Reference image tensor (B, C, H, W) in [-1, 1].
        """
       
        # Set ref embedding
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
    

    def compute_loss(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between the generated image and the ground truth.

        Args:
            x (torch.Tensor): Generated image tensor.
            gt (torch.Tensor): Ground truth image tensor.

        Returns:
            torch.Tensor: Computed loss value.
        """
        
        gt_embd = self._embeddings(gt)
        embd = self._embeddings(x)
        difference = embd - gt_embd  # (N, 512)
        loss = torch.linalg.norm(difference, dim=-1, ord=2)  # compute l2 norm
        return loss
    

    
        


        