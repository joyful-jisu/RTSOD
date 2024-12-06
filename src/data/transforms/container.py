import torchvision
import torchvision.transforms.v2 as T

from typing import Any

torchvision.disable_beta_transforms_warning()


class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        transforms = ops
        super().__init__(transforms=transforms)
        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy['name'])(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_epoch = self.policy['epoch']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample

    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
