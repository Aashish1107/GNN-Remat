from setuptools import setup, find_packages

setup(
    name="gnn-remat",
    version="0.3.0",
    description=(
        "Aggregation-granular rematerialization for PyTorch Geometric, "
        "with SAR-inspired destination-node chunked propagation for "
        "large-graph memory efficiency."
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torch_geometric>=2.3",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
    },
)
