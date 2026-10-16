
import numpy as np

class TransformPipeline:

    def __init__(self, transforms=None):
        self.transforms = transforms or []

    def __call__(self, x):
        if isinstance(x, list):
            out = []
            for item in x:
                for t in self.transforms:
                    item = t(item)
                out.append(item)
            return out

        for t in self.transforms:
            x = t(x)
        return x

    def add(self, t):
        self.transforms.append(t)

    def extend(self, ts):
        self.transforms.extend(ts)

    def __repr__(self):
        names = [t.__class__.__name__ for t in self.transforms]
        return f"TransformPipeline({names})"


class Normalize:
    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return (x - mean) / std


class RandomFlip:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            return np.flip(x, axis=0).copy()
        return x


class RandomGaussianNoise:
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, x):
        return x + np.random.randn(*x.shape) * self.sigma
