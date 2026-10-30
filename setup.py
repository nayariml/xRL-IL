from setuptools import setup, find_packages

setup(
    name="xRL",
    version="0.1.0",
    description="Cross-Embodiment Reinforcement Learning with Priors",
    author="Nayari Lessa",
    license="MIT",
    packages=find_packages(include=["hydra_plugins", "hydra_plugins.*"]),
    install_requires=[
        "torch",
        "numpy",
        "gymnasium",
        "hydra-core",
        "loguru",
        "wandb",
    ],
    python_requires=">=3.8",
)
