import yaml
import argparse

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


parser = argparse.ArgumentParser(description="Test reward config")
parser.add_argument(
    "--reward_eval_config",
    type=str,
    default="configs/rewards_adaface_measurement.yaml",
    help="Path to the reward evaluation config file",
)

parser.add_argument(
    "--ref_faces_path",
    type=str,
    default="data/ref-face-images",
    help="Path to the reference faces directory",
)


args = parser.parse_args()
reward_eval_config = load_yaml(args.reward_eval_config)

print("Reward evaluation config:")
print(reward_eval_config)

# rewards = reward_eval_config['rewards']

# print('Rewards:', rewards)

all_rewards = {}

from reward_eval import get_reward_eval
for reward_config in reward_eval_config:
    reward = get_reward_eval(**reward_config)
    all_rewards[reward_config['name']] = reward
print('All rewards:', all_rewards['adaface'])
print('All rewards:', all_rewards['measurement'])

from glob import glob
import os
from PIL import Image
from torchvision import transforms
import torch

img_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])


i = 0
ref_faces = sorted(glob(os.path.join(args.ref_faces_path + '/*.png')))

ref_face_img = Image.open(ref_faces[i]).convert('RGB')
ref_face_img = transform(ref_face_img)
ref_face_img = ref_face_img.to(device)
ref_face_img = ref_face_img.unsqueeze(0)


    
    
