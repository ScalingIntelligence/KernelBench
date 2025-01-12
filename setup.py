from setuptools import setup

if __name__ == "__main__":
    setup(
        name="src",
        version="0.0.1",
        packages=["src"],
        include_package_data=True,
        package_data={"src": ["KernelBench/*"]}
    )
